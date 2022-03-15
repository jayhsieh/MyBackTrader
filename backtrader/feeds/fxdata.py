from datetime import datetime, timedelta
from heapq import heappop

from backtrader import date2num, num2date
from backtrader.feed import DataBase
from backtrader.stores import fxstore
from backtrader.utils.py3 import (queue, with_metaclass)
from backtrader.utils.logger import log_setup


class MetaFXData(DataBase.__class__):
    def __init__(cls, name, bases, dct):
        """Class has already been created ... register"""
        # Initialize the class
        super(MetaFXData, cls).__init__(name, bases, dct)

        # Register with the store
        fxstore.FXStore.DataCls = cls


class FXData(with_metaclass(MetaFXData, DataBase)):
    """FX Data Feed.

    Params:

      - ``qcheck`` (default: ``0.5``)

        Time in seconds to wake up if no data is received to give a chance to
        resample/replay packets properly and pass notifications up the chain

      - ``historical`` (default: ``False``)

        If set to ``True`` the data feed will stop after doing the first
        download of data.

        The standard data feed parameters ``fromdate`` and ``todate`` will be
        used as reference.

        The data feed will make multiple requests if the requested duration is
        larger than the one allowed by IB given the timeframe/compression
        chosen for the data.

      - ``backfill_start`` (default: ``True``)

        Perform backfilling at the start. The maximum possible historical data
        will be fetched in a single request.

      - ``backfill`` (default: ``True``)

        Perform backfilling after a disconnection/reconnection cycle. The gap
        duration will be used to download the smallest possible amount of data

      - ``backfill_from`` (default: ``None``)

        An additional data source can be passed to do an initial layer of
        backfilling. Once the data source is depleted and if requested,
        backfilling from IB will take place. This is ideally meant to backfill
        from already stored sources like a file on disk, but not limited to.

      - ``bidask`` (default: ``True``)

        If ``True``, then the historical/backfilling requests will request
        bid/ask prices from the server

        If ``False``, then *midpoint* will be requested

      - ``useask`` (default: ``False``)

        If ``True`` the *ask* part of the *bidask* prices will be used instead
        of the default use of *bid*

      - ``includeFirst`` (default: ``True``)

        Influence the delivery of the 1st bar of a historical/backfilling
        request by setting the parameter directly to the Oanda API calls

      - ``reconnect`` (default: ``True``)

        Reconnect when network connection is down

      - ``reconnections`` (default: ``-1``)

        Number of times to attempt reconnections: ``-1`` means forever

      - ``reconntimeout`` (default: ``5.0``)

    """
    params = (
        ('qcheck', 0.5),
        ('historical', False),  # do backfilling at the start
        ('backfill_start', False),  # do backfilling at the start
        ('backfill', False),  # do backfilling when reconnecting
        ('backfill_from', None),  # additional data source to do backfill from
        ('bidask', True),
        ('useask', False),
        ('includeFirst', True),
        ('reconnect', True),
        ('reconnections', -1),  # forever
        ('reconntimeout', 5.0),
        ('timeframe', 2),  # microsecond
    )

    _store = fxstore.FXStore

    # States for the Finite State Machine in _load
    _ST_FROM, _ST_START, _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(5)

    _TOFFSET = timedelta()

    def _timeoffset(self):
        # Effective way to overcome the non-notification?
        return self._TOFFSET

    def islive(self):
        """
        True notifies `Cerebro` that `preloading` and `runonce`
        should be deactivated
        """
        return True

    def __init__(self, **kwargs):
        self._statelivereconn = False  # if reconnecting in live state
        self._storedmsg = dict()  # keep pending live message (under None)
        self.qlive = []  # heap-queue
        self.o = self._store(**kwargs)
        # self._candleFormat = 'bidask' if self.p.bidask else 'midpoint'

        # Subscribe data
        self.o.subscribe(self.p.dataname)

        ccy = self.p.dataname[:3] + self.p.dataname[4:]

        self.logger = log_setup(f"{self.__class__.__name__}_{ccy}")
        """Logger to record any log"""

    def setenvironment(self, env):
        """
        Receives an environment (cerebro) and passes it over to the store it
        belongs to
        """
        super(FXData, self).setenvironment(env)
        env.addstore(self.o)

    def start(self):
        """
        Starts the FX connection and gets the real contract and
        contract details if it exists
        """
        super(FXData, self).start()

        # Create attributes as soon as possible
        self._state = self._ST_OVER

        # Kickstart store and get queue to wait on
        self.o.start(data=self)

        # Configure server script symbol and time frame
        # Error will be raised if params are not supported
        # ram self.o.config_server(self.p.dataname, data_tf)

        # Backfill from external data feed
        if self.p.backfill_from is not None:
            self._state = self._ST_FROM
            self.p.backfill_from._start()
        else:
            self._start_finish()
            # initial state for _load
            self._state = self._ST_START
            self._st_start()

    def _st_start(self, instart=True, tmout=None):
        if self.p.historical:
            self.put_notification(self.DELAYED)
            dtend = None
            if self.todate < float('inf'):
                dtend = num2date(self.todate)

            dtbegin = None
            if self.fromdate > float('-inf'):
                dtbegin = num2date(self.fromdate)

            self.qhist = self.o.candles(
                self.p.dataname, dtbegin, dtend,
                self._timeframe, self._compression,
                candleFormat=self._candleFormat,
                includeFirst=self.p.includeFirst)

            self._state = self._ST_HISTORBACK
            return True

        self.qlive = self.o.streaming_prices(self.p.dataname)
        if instart:
            self._statelivereconn = self.p.backfill_start
        else:
            self._statelivereconn = self.p.backfill

        if self._statelivereconn:
            self.put_notification(self.DELAYED)

        self._state = self._ST_LIVE
        if instart:
            self._reconns = self.p.reconnections

        return True  # no return before - implicit continue

    def stop(self):
        """Stops and tells the store to stop"""
        super(FXData, self).stop()
        self.o.stop()

    def haslivedata(self):
        return bool(self.qlive)  # do not return the obj

    def _load(self):
        if self._state == self._ST_OVER:
            return False

        while True:
            if self._state == self._ST_LIVE:
                try:
                    msg = (self._storedmsg.pop(None, None) or
                           heappop(self.qlive)[-1])
                except queue.Empty:
                    return None  # indicate timeout situation
                except IndexError:
                    return None  # No index in deque
                except TypeError as e:
                    if e.args[0] == "\'<\' not supported between instances of \'dict\' and \'dict\'":
                        continue
                    else:
                        raise e

                if msg is None:  # Conn broken during historical/backfilling
                    self.put_notification(self.CONNBROKEN)
                    # Try to reconnect
                    if not self.p.reconnect or self._reconns == 0:
                        # Can no longer reconnect
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False  # failed

                    self._reconns -= 1
                    self._st_start(instart=False)
                    continue

                self._reconns = self.p.reconnections

                # Process the message according to expected return type
                if not self._statelivereconn:
                    if self._laststatus != self.LIVE:
                        if len(self.qlive) <= 1:  # very short live queue
                            self.put_notification(self.LIVE)

                    ret = self._load_tick(msg)
                    if ret:
                        return True

                    # could not load bar ... go and get new one
                    continue

                # Fall through to processing reconnect - try to backfill
                self._storedmsg[None] = msg  # keep the msg

                # else do a backfill
                if self._laststatus != self.DELAYED:
                    self.put_notification(self.DELAYED)

                dtend = None
                if len(self) > 1:
                    # len == 1 ... forwarded for the 1st time
                    dtbegin = self.datetime.datetime(-1)
                elif self.fromdate > float('-inf'):
                    dtbegin = num2date(self.fromdate)
                else:  # 1st bar and no begin set
                    # passing None to fetch max possible in 1 request
                    dtbegin = None

                # self._state = self._ST_HISTORBACK
                self._statelivereconn = False  # no longer in live
                continue

            elif self._state == self._ST_HISTORBACK:
                msg = self.qhist.get()
                if msg is None:  # Conn broken during historical/backfilling
                    # Situation not managed. Simply bail out
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False  # error management cancelled the queue

                elif 'code' in msg:  # Error
                    self.put_notification(self.NOTSUBSCRIBED)
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False

                if msg:
                    if self._load_history(msg):
                        return True  # loading worked

                    continue  # not loaded ... date may have been seen
                else:
                    # End of histdata
                    if self.p.historical:  # only historical
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False  # end of historical

                # Live is also wished - go for it
                self._state = self._ST_LIVE
                continue

            elif self._state == self._ST_FROM:
                if not self.p.backfill_from.next():
                    # additional data source is consumed
                    self._state = self._ST_START
                    continue

                # copy lines of the same name
                for alias in self.lines.getlinealiases():
                    lsrc = getattr(self.p.backfill_from.lines, alias)
                    ldst = getattr(self.lines, alias)

                    ldst[0] = lsrc[0]

                return True

            elif self._state == self._ST_START:
                if not self._st_start(instart=False):
                    self._state = self._ST_OVER
                    return False

    def _load_tick(self, msg):
        if msg['DT'] <= self.lines.datetime[-1]:
            return False  # time already seen

        # Common fields
        self.lines.datetime[0] = msg['DT']

        # Put the prices into the bar
        tick = msg['AskPrice'] if self.p.useask else msg['BidPrice']
        self.lines.open[0] = tick
        self.lines.high[0] = tick
        self.lines.low[0] = tick
        self.lines.close[0] = tick
        self.lines.volume[0] = 0.0
        self.lines.openinterest[0] = 0.0

        dt_data = self.datetime.datetime()
        dt_streamer = msg['StreamerTime']
        dt_now = datetime.now()
        dt = (dt_now - dt_streamer).total_seconds()

        self.logger.info(f"DataTime: {dt_data.isoformat()}, "
                         f"StreamerTime: {dt_streamer.isoformat()}, "
                         f"ReceiveTime: {dt_now.isoformat()}, "
                         f"DiffTime: {dt}/sec, "
                         f"Bid: {msg['BidPrice']}, "
                         f"Ask: {msg['AskPrice']}, "
                         f"Mid: {msg['TradePrice']}")

        return True

    # TODO
    def _load_history(self, msg):
        dtobj = datetime.utcfromtimestamp(int(msg['time']) / 10 ** 6)
        dt = date2num(dtobj)
        if dt <= self.lines.datetime[-1]:
            return False  # time already seen

        # Common fields
        self.lines.datetime[0] = dt
        self.lines.volume[0] = float(msg['volume'])
        self.lines.openinterest[0] = 0.0

        # Put the prices into the bar
        if self.p.bidask:
            if not self.p.useask:
                self.lines.open[0] = float(msg['openBid'])
                self.lines.high[0] = float(msg['highBid'])
                self.lines.low[0] = float(msg['lowBid'])
                self.lines.close[0] = float(msg['closeBid'])
            else:
                self.lines.open[0] = float(msg['openAsk'])
                self.lines.high[0] = float(msg['highAsk'])
                self.lines.low[0] = float(msg['lowAsk'])
                self.lines.close[0] = float(msg['closeAsk'])
        else:
            self.lines.open[0] = float(msg['openMid'])
            self.lines.high[0] = float(msg['highMid'])
            self.lines.low[0] = float(msg['lowMid'])
            self.lines.close[0] = float(msg['closeMid'])

        return True
