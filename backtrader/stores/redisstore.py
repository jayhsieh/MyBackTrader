import asyncio
import collections
import heapq
import itertools
import threading
import time
from datetime import timedelta, datetime

import aioredis
from backtrader import date2num
from backtrader.metabase import MetaParams
from backtrader.utils.checker import check_trade_hour
from backtrader.utils.config import cfg
from backtrader.utils.py3 import (queue, with_metaclass)

redis_cfg = cfg["redis"]
MAX_SIZE = cfg["queue_size"]  # max tick per day is about 60000
MAX_PUT_SIZE = int(0.9 * MAX_SIZE)
started = False
Second, Minute, Hour, Day = range(4)


def _get_timeframe(f: str):
    """
    Gets timeframe of resample.
    :param f: String of frequency, e.g. 5S, 15M, 1H, 1D
    :return: Time frame and time compression
    """
    f = f.upper()
    if f.endswith("S"):
        tf = Second
        comp = int(f.replace("S", ""))
    elif f.endswith("M"):
        tf = Minute
        comp = int(f.replace("M", ""))
    elif f.endswith("H"):
        tf = Hour
        comp = int(f.replace("H", ""))
    elif f.endswith("D"):
        tf = Day
        comp = int(f.replace("D", ""))
    else:
        msg = "Cannot read this timeframe." \
              " Only 'S', 'M', 'H', 'D' is allowed."
        raise Exception(msg)

    return tf, comp


def _get_period(date: datetime, freq: str):
    """
    Gets the trade period of the bar.
    :param date: DateTime of the tick
    :param freq: String of frequency, e.g. 5S, 15M, 1H, 1D
    :return: Start date and end date of the bar.
    """
    sd, ed = None, None

    tf, comp = _get_timeframe(freq)

    if tf == Second:
        sec = date.second // comp * comp
        sd = date.replace(second=sec, microsecond=0)
        ed = sd + timedelta(seconds=comp)
    elif tf == Minute:
        minute = date.minute // comp * comp
        sd = date.replace(minute=minute, second=0,
                          microsecond=0)
        ed = sd + timedelta(minutes=comp)
    elif tf == Hour:
        hr = date.hour // comp * comp
        sd = date.replace(hour=hr, minute=0,
                          second=0, microsecond=0)
        ed = sd + timedelta(hours=comp)
    elif tf == Day:
        day = date.day // comp * comp
        sd = date.replace(day=day, hour=0, minute=0,
                          second=0, microsecond=0)
        ed = sd + timedelta(days=comp)

    return sd, ed


class Streamer(threading.Thread):

    def __init__(self, url: str, qs, subscription: list, run_status: list):
        """
        Initializes a streamer to collect messages.
        :param url: Redis url with database number.
        :param qs: Heap queues for collecting data by subscriptions
        :param subscription: [(ticker, freq)]
            List of K-bars subscriptions with ticker and freq
            where freq is like '15M', '1H', '5D'.
        """
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()

        self.qs = qs
        """Heap queues to hold data requests from Redis"""
        self.qs_key = self.qs.keys()
        """Keys of queues in ``self.qs``."""

        self.subscription = subscription
        """Subscription list of datas"""

        self.url = url
        """URL of Redis database"""
        self._keys = []
        """Used keys of Redis data"""

        self.run_status = run_status
        """
        Status of streamer. 
        To return by address, we use list to package the boolean.\n
        :type list[bool] with length 1
        """

    def stop(self):
        self.stop_event.set()

    async def consume(self, data_src):
        """
        Consumes messages from Redis
        :param data_src: Data source name, like 'HSBC'.
        """
        redis = await aioredis.from_url(self.url)
        async with redis.client() as conn:
            # Scan the whole database
            cur = b"0"
            while cur:
                cur, keys = await conn.scan(cur, match=f"{data_src}:*", count=10000)

                # Skip for no keys
                if not len(keys):
                    continue

                for k in keys:
                    # Skipped for used key
                    if k in self._keys:
                        continue

                    # Check subscription of key
                    k_split = k.split(b":")
                    ticker = k_split[1].decode("utf-8")
                    freq = k_split[2].decode("utf-8")
                    k_sub = (ticker, freq)
                    if k_sub not in self.subscription:
                        continue

                    msg = await conn.hgetall(k)

                    await self._deal_tick(msg, k_sub)

                    # Append used key
                    self._keys.append(k)

            self.run_status[0] = True
            self.sleep()

    async def _deal_tick(self, msg, key):
        """
        Analyzes bar from Redis and push into ``self.qs``
        :param msg: Message from Redis
        :param key: Key of data
        """
        bar = dict()
        dt = datetime.strptime(msg[b'time'].decode("utf-8"), "%Y-%m-%d %H:%M:%S")

        # Check if data is in market hour
        if not check_trade_hour(dt):
            return

        # Check subscription
        if key not in self.qs_key:
            if key in self.subscription:
                self.qs[key] = []
                self.qs_key = self.qs.keys()
                print(self.qs_key)
            else:
                return

        # Analyzes tick from Redis message
        bar['datetime'] = dt
        bar['open'] = float(msg[b'open'] or 0)
        bar['high'] = float(msg[b'high'] or 0)
        bar['low'] = float(msg[b'low'] or 0)
        bar['close'] = float(msg[b'close'] or 0)
        bar['volume'] = int(msg[b'volume'] or 0)
        bar['open_interest'] = int(msg[b'open_interest'] or 0)
        bar['DT'] = date2num(dt)

        try:
            while len(self.qs[key]) > MAX_PUT_SIZE:
                heapq.heappop(self.qs[key])

            val = (bar['DT'], bar)
            if val not in self.qs[key]:
                heapq.heappush(self.qs[key], val)

        except Exception as e:
            print('Redis Error:' + str(e))
            print(bar)

    def sleep(self):
        """Sleep until next time of bar"""
        next_time = None
        dt_now = datetime.now()
        for s in self.subscription:
            _, nt = _get_period(dt_now, s[1])
            if next_time is None:
                next_time = nt
            else:
                next_time = min(next_time, nt)

        sec = max((next_time - dt_now).seconds - 1, 0)
        time.sleep(sec)

    def run(self):
        """Runs an async loop to consume messages"""
        data_src = redis_cfg["data_src"]

        # 建立一個Event Loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        loop = asyncio.get_event_loop()

        # 建立一個任務列表
        tasks = [
            asyncio.ensure_future(self.consume(data_src))
        ]

        # 開始執行
        loop.run_until_complete(asyncio.gather(*tasks))


class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(cls, name, bases, dct):
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = (
                super(MetaSingleton, cls).__call__(*args, **kwargs))

        return cls._singleton


class RedisStore(with_metaclass(MetaSingleton, object)):
    REQIDBASE = 0x01000000
    BrokerCls = None
    """broker class will autoregister"""
    DataCls = None
    """data class will auto register"""

    params = (
        ('host', ''),
        ('post', ''),
        ('market', ''),
        ('order', ''),  # account balance refresh timeout
        ('subscription', []),  # subscription list of data
        ('run_status', [False])  # store run status and return
    )

    @classmethod
    def getdata(cls, *args, **kwargs):
        """Returns ``DataCls`` with args, kwargs"""
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        """Returns broker with *args, **kwargs from registered ``BrokerCls``"""
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self):
        super(RedisStore, self).__init__()

        self._lock_q = threading.Lock()
        """sync access to _tickerId/Queues"""

        self.notifs = queue.Queue()
        """store notifications for cerebro"""

        self._env = None
        """reference to cerebro for general notifications"""
        self.broker = None
        """broker instance"""
        self.datas = list()
        """datas that have registered over start"""
        self.ccount = 0
        """requests to start (from cerebro or datas)"""

        self._lock_tmoffset = threading.Lock()
        self.tmoffset = timedelta()
        """to control time difference with server"""

        self.notifs = collections.deque()
        """store notifications for cerebro"""

        # Structures to hold datas requests
        self.qs = collections.OrderedDict()
        """Heap queues to store prices"""
        self.ts = collections.OrderedDict()
        """key: queue -> tickerId"""

        self._tickerId = itertools.count(self.REQIDBASE)
        """unique tickerIds"""

        self._cash = 0.0
        self._value = 0.0

        self._cancel_flag = False

        self.debug = True

    def subscribe(self, data_name, freq):
        """
        Subscribe data by name
        :param data_name: Name of data
        :param freq: Frequency of K-bar data.
            Can be '1M', '5M', '10M', '15M', '30M', '1H'.
        """
        key = (data_name, freq)
        if key not in self.p.subscription:
            self.p.subscription.append(key)

    def start(self, data=None, broker=None):
        """
        Starts running Redis store and its streamer.
        :param data:
        :param broker:
        :return:
        """
        global started

        # Data require some processing to kickstart data reception
        if data is None and broker is None:
            self.cash = None
            return

        if data is not None:
            self._env = data._env
            # For datas simulate a queue with None to kickstart co
            self.datas.append(data)

            if self.broker is not None:
                self.broker.data_started(data)

        elif broker is not None:
            self.broker = broker
            self.broker_threads()
            self.streaming_events()

        kwargs = {
            'tmout': redis_cfg["time_out"],
            'url': redis_cfg["url"],
            'qs': self.qs,
            'subscription': self.p.subscription,
            'run_status': self.p.run_status
        }

        # Only use one thread to collect from Redis,
        # no matter how many redis data are used
        if not started:
            t = threading.Thread(target=self._t_streaming_prices, kwargs=kwargs)
            t.daemon = True
            t.start()
            started = True

    def stop(self):
        """Signal end of thread"""
        if self.broker is not None:
            self.q_ordercreate.put(None)
            self.q_orderclose.put(None)

    def put_notification(self, msg, *args, **kwargs):
        self.notifs.append((msg, args, kwargs))

    def get_notifications(self):
        """Return the pending "store" notifications"""
        self.notifs.append(None)  # put a mark / threads could still append
        return [x for x in iter(self.notifs.popleft, None)]

    def streaming_events(self, tmout=None):
        q = queue.Queue()
        kwargs = {'q': q, 'tmout': tmout}

        t = threading.Thread(target=self._t_streaming_listener, kwargs=kwargs)
        t.daemon = True
        t.start()

        t = threading.Thread(target=self._t_streaming_events, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    def streaming_prices(self, symbol: str, freq: str):
        """
        Gets heap queue of the price from streamer.
        :param symbol: Symbol of product
        :param freq: Frequency of K-bar data.
            Can be '1M', '5M', '10M', '15M', '30M', '1H'.
        :return: Heap queues of price bar
        """
        k = (symbol, freq)
        while k not in self.qs.keys():
            pass
        else:
            return self.qs[k]

    def _t_streaming_prices(self, tmout, url, qs, subscription, run_status):
        """
        Runs a streaming process to collect prices from Redis
        :param url: Redis url with database number.
        :param qs: Heap queues for collecting data by subscriptions
        :param tmout: Run streamer then sleep 'tmout' seconds.
        :param subscription: [(ticker, freq)]
            List of K-bars subscriptions with ticker and freq
            where freq is like '15M', '1H', '5D'.
        :param run_status: Running status of streamer.
        """
        streamer = Streamer(url, qs, subscription, run_status)
        streamer.run()

        if tmout is not None:
            time.sleep(tmout)

    def streaming_status(self):
        """
        Returns streaming status whether it has run at least once
        :return: Running status of streamer
        """
        return self.p.run_status[0]
