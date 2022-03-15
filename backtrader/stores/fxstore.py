import asyncio
import collections
import heapq
import itertools
import json
import sys
import threading
import traceback
from datetime import datetime, timedelta

from aiokafka import AIOKafkaProducer
from kafka import KafkaConsumer

from backtrader.utils import date2num, str2dt
from backtrader.metabase import MetaParams
from backtrader.utils.config import cfg
from backtrader.utils.py3 import (queue, with_metaclass)
from backtrader.utils.logger import log_setup

kafka_cfg = cfg["kafka"]
MAX_SIZE = cfg["queue_size"]  # max tick per day is about 60000
MAX_PUT_SIZE = int(0.9 * MAX_SIZE)
started = False


class FXStreamer(threading.Thread):
    def __init__(self, qs, subscription):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()

        # The last quotes keeper
        self.dic_quotes = {}

        # Structures to hold datas requests
        self.qs = qs
        self.qskey = self.qs.keys()

        # Subscription list of datas
        self.subscription = subscription

        self.logger = log_setup(self.__class__.__name__)
        """Logger to record any log"""

    def stop(self):
        self.stop_event.set()

    async def consume(self):
        """
        Consumes data from Kafka with topic.\n
        """
        if cfg["env"] == "DEV":
            consumer = KafkaConsumer(
                kafka_cfg["tick_topic"],
                bootstrap_servers=kafka_cfg["ip"],
                auto_offset_reset=kafka_cfg["offset"],
                connections_max_idle_ms=kafka_cfg["idle"],
            )
        elif cfg["env"] == "SIT":
            consumer = KafkaConsumer(
                kafka_cfg["tick_topic"],
                bootstrap_servers=kafka_cfg["ip"],
                auto_offset_reset=kafka_cfg["offset"],
                sasl_mechanism=kafka_cfg["sasl_mechanism"],
                sasl_plain_username=kafka_cfg["username"],
                sasl_plain_password=kafka_cfg["password"],
                ssl_cafile=kafka_cfg["ca_file"],
                client_id=kafka_cfg["username"],
                group_id=kafka_cfg["group"],
                connections_max_idle_ms=kafka_cfg["idle"]
            )
        else:
            msg = f"Did not set configuration in environment {cfg['env']}"
            raise EnvironmentError(msg)

        try:
            # Consume messages
            for msg in consumer:
                tick = json.loads(msg.value)
                ticker = tick['FullSymbol']
                datetime_tick = str2dt(tick['Date'])

                # Check if data is in market hour
                if datetime_tick.weekday() == 6:
                    continue
                elif datetime_tick.weekday() == 5 and datetime_tick.hour > 6:
                    continue
                elif datetime_tick.weekday() == 0 and datetime_tick.hour < 6:
                    continue

                # Check subscription
                if ticker not in self.qskey:
                    if ticker in self.subscription:
                        self.qs[ticker] = []
                        self.qskey = self.qs.keys()
                        print(self.qskey)
                    else:
                        continue

                tick['BidPrice'] = float(tick['BidPrice'] or 0)
                tick['AskPrice'] = float(tick['AskPrice'] or 0)
                tick['TradePrice'] = (tick['BidPrice'] + tick['AskPrice']) * 0.5
                tick['DT'] = date2num(datetime_tick)
                tick['StreamerTime'] = datetime.now()

                try:
                    while len(self.qs[ticker]) > MAX_PUT_SIZE:
                        heapq.heappop(self.qs[ticker])

                    val = (tick['DT'], tick['BidPrice'], tick)
                    if val not in self.qs[ticker]:
                        heapq.heappush(self.qs[ticker], val)
                        dt_now = tick['StreamerTime']
                        dt = (dt_now - datetime_tick).total_seconds()
                        self.logger.info(f"DataTime: {datetime_tick.isoformat()}, "
                                         f"ReceiveTime: {dt_now.isoformat()}, "
                                         f"DiffTime: {dt}/sec, "
                                         f"Ccy: {ticker}, "
                                         f"Bid: {tick['BidPrice']}, "
                                         f"Ask: {tick['AskPrice']}, "
                                         f"Mid: {tick['TradePrice']}")

                except Exception as e:
                    msg = f"Kafka Error: {e}\n{tick}"
                    print(msg)
                    self.logger.exception(e)

        except Exception as e:
            error_class = e.__class__.__name__
            detail = e.args[0]

            # Get last call stack
            cl, exc, tb = sys.exc_info()
            last_call_stack = traceback.extract_tb(tb)[-1]

            # Get information of error
            file_name = last_call_stack[0]
            line_num = last_call_stack[1]
            func_name = last_call_stack[2]

            err_msg = f"File \"{file_name}\", line {line_num}, in {func_name}: " \
                      f"[{error_class}] {detail}"
            print(err_msg)
            raise Exception(err_msg)

    def run(self):
        """
        Runs Kafka consumer.\n
        """
        # 建立一個Event Loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        loop = asyncio.get_event_loop()

        # 建立一個任務列表
        tasks = [
            asyncio.ensure_future(self.consume())
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


class FXStore(with_metaclass(MetaSingleton, object)):
    REQIDBASE = 0x01000000
    BrokerCls = None  # broker class will autoregister
    DataCls = None  # data class will auto register

    params = (
        ('host', ''),
        ('post', ''),
        ('market', ''),
        ('order', ''),  # account balance refresh timeout
        ('subscription', [])
    )

    @classmethod
    def getdata(cls, *args, **kwargs):
        """Returns ``DataCls`` with args, kwargs"""
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        """Returns broker with *args, **kwargs from registered ``BrokerCls``"""
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self, host='localhost'):
        super(FXStore, self).__init__()

        self.cash = None
        self._lock_q = threading.Lock()  # sync access to _tickerId/Queues

        self.notifs = queue.Queue()  # store notifications for cerebro

        self._env = None  # reference to cerebro for general notifications
        self.broker = None  # broker instance
        self.datas = list()  # datas that have registered over start
        self.ccount = 0  # requests to start (from cerebro or datas)

        self._lock_tmoffset = threading.Lock()
        self.tmoffset = timedelta()  # to control time difference with server

        self.notifs = collections.deque()  # store notifications for cerebro

        # self._orders = collections.OrderedDict()  # map order.ref to oid
        # self._ordersrev = collections.OrderedDict()  # map oid to order.ref
        # self._orders_type = dict()  # keeps order types

        # Structures to hold datas requests
        self.qs = collections.OrderedDict()  # key: tickerId -> queues
        self.ts = collections.OrderedDict()  # key: queue -> tickerId

        self._tickerId = itertools.count(self.REQIDBASE)  # unique tickerIds

        self._cash = 0.0
        self._value = 0.0

        self._cancel_flag = False

        self.debug = True

    def subscribe(self, dataname):
        """
        Subscribe data by name
        :param dataname: Name of data
        :return:
        """
        if dataname not in self.p.subscription:
            self.p.subscription.append(dataname)

    def start(self, data=None, broker=None):
        global started

        # Data require some processing to kickstart data reception
        if data is None and broker is None:
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
        tmout = None
        kwargs = {'qs': self.qs, 'tmout': tmout, 'subscription': self.p.subscription}

        # Only use one thread to collect from Kafka, no matter how many FX data are used
        if not started:
            t = threading.Thread(target=self._t_streaming_prices, kwargs=kwargs)
            t.daemon = True
            t.start()
            started = True

    def stop(self):
        # signal end of thread
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

    def streaming_prices(self, symbol):
        while symbol not in self.qs.keys():
            pass
        else:
            return self.qs[symbol]

    def _t_streaming_prices(self, qs, tmout, subscription):
        # if tmout is not None:
        #     _time.sleep(tmout)

        streamer = FXStreamer(qs, subscription)
        streamer.start()
