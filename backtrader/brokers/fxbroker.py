import collections
import math
import queue
from datetime import datetime

import numpy as np

from backtrader.brokers.bbroker import BackBroker
from backtrader.order import Order, BuyOrder, SellOrder
from backtrader.utils import num2date
from backtrader.utils.db_conn import get_tick
from backtrader.utils.ordermisc import factor_down, factor_up

LIMIT_HIGH = 1e6
"""Default highest limit price for buy market order"""
LIMIT_LOW = 0
"""Default lowest limit price for sell market order"""


def price_reverse(price, size, ccy, is_cash_fix):
    """
    Reverses price and size to its reciprocal if foreign currency is USD.\n
    :param price: Trade price.
    :param size: Trade size
    :param ccy: Currency pair in XXXYYY.
    :param is_cash_fix: Flag if the cash amount of the transaction is fixed.
        If true, size will be adjusted by its price. This is especially for
        USD as foreign currency.
    :return: (price, size) Adjusted price and size in tuple.
    """
    if ccy[:3] == "USD":
        # Reverse price if ccy1 is USD
        price = 1 / price

    if is_cash_fix:
        # If cash amount is fixed, change size divided by price
        size = -size / price

    return price, size


def get_tick_price(ccy: str, start: datetime, end: datetime, price: float, is_buy: bool, is_limit: bool):
    """
    Gets tick price given the time period and the condition of order.\n
    :param ccy: Currency pair in XXXYYY.
    :param start: Start time period.
    :param end: End time period.
    :param price: Price of the order.
    :param is_buy: Flag if the order is to buy.
    :param is_limit: Flag if the order is limit order or stop order.
    :return: Tradable tick price, max/min price of the time period, datetime of tick.
    """
    # Import and filter data from csv
    df = get_tick(ccy, start, end, "HSBC")
    df = df.reset_index(drop=True)

    if is_buy:
        df = df.loc[:, ["DateTime", "offer"]]
        pmax = df["offer"].max()

        # Get tick which is the first one match the condition
        if is_limit:
            ticks = df[df["offer"] <= price]
        else:
            ticks = df[df["offer"] >= price]

        if len(ticks) == 0:
            # Return if no tick is valid in the time period
            return None, None, None

        tick = ticks.iloc[0]
        dt = tick["DateTime"].to_pydatetime()
        tick = tick["offer"]

        return tick, pmax, dt
    else:
        df = df.loc[:, ["DateTime", "bid"]]
        pmax = df["bid"].min()

        # Get tick which is the first one match the condition
        if is_limit:
            ticks = df[df["bid"] >= price]
        else:
            ticks = df[df["bid"] <= price]

        if len(ticks) == 0:
            # Return if no tick is valid in the time period
            return None, None, None

        tick = ticks.iloc[0]
        dt = tick["DateTime"].to_pydatetime()
        tick = tick["bid"]

        return tick, pmax, dt


class FXBroker(BackBroker):
    """FX Broker Simulator

      The simulation not only includes all functions and data types in ``BackBroker``
      but also supports FX trading.

      Improved features in ``BackBroker``:

        1. In any order, we set property ``nccy`` to reflect the notional currency
           of the order size

        2. In ``_execute``, we adjust the price and the size to convert
            currency pair from *USD/XXX* to *XXX/USD* and to reflect the
            notional currency

        3. In ``check_submitted``, we allow order if any position does not exceed the
            trading limit size.

        4. In ``next``, we cancel order when it is expired

        5. In ``_try_exec*``, we use tick data instead of bar data

      TODO: Transaction for cross currencies need to be considered in the future.
    """

    def __init__(self, limit_size: float, use_tick: bool = True):
        """
        Initializes FX broker.\n
        :param limit_size: Trading limit size. Cannot be negative.
        :param use_tick: Flag if tick data is used when execution is made.
        """
        super(FXBroker, self).__init__()

        self.print_position = None
        self._print_ccy = []
        self._last_min = -1
        self.timediff = queue.Queue(50)
        for i in range(50):
            self.timediff.put(float('nan'))

        self._last_len = -1  # last length of all data
        self._last_dt = datetime(1970, 1, 1)

        if limit_size < 0:
            raise ValueError(f"Trading limit size cannot be negative: {limit_size}")
        self.limit = limit_size
        """Trading limit size"""

        if not use_tick:
            # If tick data is not used, use bar data to make execution only
            self._try_exec_market = super(FXBroker, self)._try_exec_market
            self._try_exec_limit = super(FXBroker, self)._try_exec_limit
            self._try_exec_stop = super(FXBroker, self)._try_exec_stop
            self._try_exec_stoplimit = super(FXBroker, self)._try_exec_stoplimit
            self._try_exec_close = super(FXBroker, self)._try_exec_close
            self._try_exec_historical = super(FXBroker, self)._try_exec_historical

    def start(self):
        super(FXBroker, self).start()

        # Determinate print method by live
        if self.cerebro.datas[0].islive():
            self.print_position = self.print_live
        else:
            self.print_position = self.print_history

        # List ccy pairs to print outstanding positions
        for d in self.cerebro.datas:
            c = d.p.name
            if c[:6] in self._print_ccy:
                continue
            self._print_ccy.append(c[:6])

    def buy(self, owner, data,
            size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            parent=None, transmit=True,
            histnotify=False, _checksubmit=True,
            nccy=None,
            **kwargs):

        if nccy is None:
            # Default is ccy1
            ccys = data.p.name
            nccy = ccys[:3]

        order = BuyOrder(owner=owner, data=data,
                         size=size, price=price, pricelimit=plimit,
                         exectype=exectype, valid=valid, tradeid=tradeid,
                         trailamount=trailamount, trailpercent=trailpercent,
                         parent=parent, transmit=transmit,
                         histnotify=histnotify)

        order.addinfo(nccy=nccy)
        order.addinfo(**kwargs)
        self._ocoize(order, oco)

        return self.submit(order, check=_checksubmit)

    def sell(self, owner, data,
             size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             parent=None, transmit=True,
             histnotify=False, _checksubmit=True,
             nccy=None,
             **kwargs):

        if nccy is None:
            # Default is ccy1
            ccys = data.p.name
            nccy = ccys[:3]

        order = SellOrder(owner=owner, data=data,
                          size=size, price=price, pricelimit=plimit,
                          exectype=exectype, valid=valid, tradeid=tradeid,
                          trailamount=trailamount, trailpercent=trailpercent,
                          parent=parent, transmit=transmit,
                          histnotify=histnotify)

        order.addinfo(nccy=nccy)
        order.addinfo(**kwargs)
        self._ocoize(order, oco)

        return self.submit(order, check=_checksubmit)

    def cancel(self, order, bracket=False):
        """
        Cancels order if it is in pending list.
        This check the pending list and submitted list.
        This also check its oco and bracket if asked.\n
        :param order: Order to be cancelled.
        :param bracket: Flag whether to check bracket.
        :return: Boolean status if order is in pending list.
        """
        try:
            self.pending.remove(order)
        except ValueError:
            try:
                self.submitted.remove(order)
            except ValueError:
                # If the list didn't have the element we didn't cancel anything
                return False

        order.cancel()
        self.notify(order)
        self._ococheck(order)
        if not bracket:
            self._bracketize(order, cancel=True)
        return True

    def cancel_order(self, order):
        """
        Cancels order and checks its oco and bracket.
        This is different from ``BackBroker.cancel`` since
        this will not check pending list.\n
        :param order: Order to be cancelled.
        """
        try:
            self.pending.remove(order)
        except ValueError:
            pass

        try:
            self.submitted.remove(order)
        except ValueError:
            pass

        order.cancel()
        self.notify(order)
        self._ococheck(order)
        self._bracketize(order, cancel=True)

    def check_submitted(self):
        while self.submitted:
            cash = self.cash
            order = self.submitted.popleft()

            if self._take_children(order) is None:  # children not taken
                continue

            comminfo = self.getcommissioninfo(order.data)
            position = self.positions[order.data].clone()

            # pseudo-execute the order to get the remaining cash after exec
            cash = self._execute(order, cash=cash, position=position)

            if abs(cash - self.startingcash) < self.limit:
                self.submit_accept(order)
                continue

            # Cancel order if cash or position exceeds limit
            self.cancel_order(order)

    def _try_exec_historical(self, order):
        self._execute(order, ago=0, price=order.created.price)

    def _try_exec_market(self, order, popen, phigh, plow):
        is_buy = order.isbuy()
        is_limit = True

        if np.isnan(popen):
            # Return if no valid data
            return

        data = order.data

        if order.isbuy():
            p, pmax, dt = get_tick_price(data.p.name[:6],
                                         data.datetime.datetime(0),
                                         data.datetime.datetime(1),
                                         LIMIT_HIGH, is_buy, is_limit)
            p = self._slip_up(pmax, p, doslip=self.p.slip_open)
        else:
            p, pmin, dt = get_tick_price(data.p.name[:6],
                                         data.datetime.datetime(0),
                                         data.datetime.datetime(1),
                                         LIMIT_LOW, is_buy, is_limit)
            p = self._slip_down(pmin, p, doslip=self.p.slip_open)

        self._execute(order, ago=0, price=p, dtcoc=dt)

    def _try_exec_close(self, order, pclose):
        # pannotated allows to keep track of the closing bar if there is no
        # information which lets us know that the current bar is the closing
        # bar (like matching end of session bar)
        # The actual matching will be done one bar afterwards but using the
        # information from the actual closing bar

        dt0 = order.data.datetime[0]
        # don't use "len" -> in replay the close can be reached with same len
        if dt0 > order.created.dt:  # can only execute after creation time
            # or (self.p.eosbar and dt0 == order.dteos):
            if dt0 >= order.dteos:
                # past the end of session or right at it and eosbar is True
                if order.pannotated and dt0 > order.dteos:
                    ago = -1
                    execprice = order.pannotated
                else:
                    ago = 0
                    execprice = pclose

                self._execute(order, ago=ago, price=execprice)
                return

        # If no exexcution has taken place ... annotate the closing price
        order.pannotated = pclose

    def _try_exec_limit(self, order, popen, phigh, plow, plimit):
        data = order.data
        is_buy = order.isbuy()
        is_limit = True

        if np.isnan(popen):
            # Return if no valid data
            return

        if is_buy:
            if plimit < plow:
                # Return if bar low is above req price
                return

            # Check tick
            p, pmax, dt = get_tick_price(data.p.name[:6],
                                         data.datetime.datetime(0),
                                         data.datetime.datetime(1),
                                         plimit, is_buy, is_limit)

            if p is None:
                # Return if the order is actually not completed
                return

            # Execute with slipped price
            p = self._slip_up(pmax, p, doslip=self.p.slip_open,
                              lim=True)
            self._execute(order, ago=0, price=p, dtcoc=dt)

        else:  # Sell
            if plimit > phigh:
                # Return if bar high is below req price
                return

            # Check tick
            p, pmin, dt = get_tick_price(data.p.name[:6],
                                         data.datetime.datetime(0),
                                         data.datetime.datetime(1),
                                         plimit, is_buy, is_limit)

            if p is None:
                # Return if the order is actually not completed
                return

            # Execute with slipped price
            p = self._slip_down(pmin, p, doslip=self.p.slip_open,
                                lim=True)
            self._execute(order, ago=0, price=p, dtcoc=dt)

    def _try_exec_stop(self, order, popen, phigh, plow, pcreated, pclose):
        data = order.data
        is_buy = order.isbuy()
        is_limit = False

        if np.isnan(popen):
            # Return if no valid data
            return

        if order.isbuy():
            if phigh >= pcreated:
                # If bar high is equal to or above req price

                # Check tick
                p, pmax, dt = get_tick_price(data.p.name[:6],
                                             data.datetime.datetime(0),
                                             data.datetime.datetime(1),
                                             pcreated, is_buy, is_limit)

                if p is not None:
                    # Execute if the order is actually completed with slipped price
                    p = self._slip_up(pmax, p)
                    self._execute(order, ago=0, price=p, dtcoc=dt)

        else:  # Sell
            if plow <= pcreated:
                # If bar low is equal to or below req price

                # Check tick
                p, pmin, dt = get_tick_price(data.p.name[:6],
                                             data.datetime.datetime(0),
                                             data.datetime.datetime(1),
                                             pcreated, is_buy, is_limit)

                if p is not None:
                    # Execute if the order is actually completed with slipped price
                    p = self._slip_down(pmin, p)
                    self._execute(order, ago=0, price=p, dtcoc=dt)

        # not (completely) executed and trailing stop
        if order.alive() and order.exectype == Order.StopTrail:
            order.trailadjust(pclose)

    def _try_exec_stoplimit(self, order,
                            popen, phigh, plow, pclose,
                            pcreated, plimit):

        if np.isnan(popen):
            # Return if no valid data
            return

        if order.isbuy():
            if popen >= pcreated:
                order.triggered = True
                self._try_exec_limit(order, popen, phigh, plow, plimit)

            elif phigh >= pcreated:
                # price penetrated upwards during the session
                order.triggered = True
                # can calculate execution for a few cases - datetime is fixed
                if popen > pclose:
                    if plimit >= pcreated:  # limit above stop trigger
                        p = self._slip_up(phigh, pcreated, lim=True)
                        self._execute(order, ago=0, price=p)
                    elif plimit >= pclose:
                        self._execute(order, ago=0, price=plimit)
                else:  # popen < pclose
                    if plimit >= pcreated:
                        p = self._slip_up(phigh, pcreated, lim=True)
                        self._execute(order, ago=0, price=p)
        else:  # Sell
            if popen <= pcreated:
                # price penetrated downwards with an open gap
                order.triggered = True
                self._try_exec_limit(order, popen, phigh, plow, plimit)

            elif plow <= pcreated:
                # price penetrated downwards during the session
                order.triggered = True
                # can calculate execution for a few cases - datetime is fixed
                if popen <= pclose:
                    if plimit <= pcreated:
                        p = self._slip_down(plow, pcreated, lim=True)
                        self._execute(order, ago=0, price=p)
                    elif plimit <= pclose:
                        self._execute(order, ago=0, price=plimit)
                else:
                    # popen > pclose
                    if plimit <= pcreated:
                        p = self._slip_down(plow, pcreated, lim=True)
                        self._execute(order, ago=0, price=p)

        # not (completely) executed and trailing stop
        if order.alive() and order.exectype == Order.StopTrailLimit:
            order.trailadjust(pclose)

    def _execute(self, order, ago=None, price=None, cash=None, position=None,
                 dtcoc=None):
        """
        Executes or pseudo executes an order.\n
        :param order: Order information.
        :param ago: None if it is for pseudo execution.
        :param price: Price with slippage effect.
        :param cash: Cash in hand.
        :param position: Outstanding positions.
        :param dtcoc: Datetime for cheat-on-close setting
        """
        # ago = None is used a flag for pseudo execution
        if ago is not None and price is None:
            return  # no psuedo exec no price - no execution

        if self.p.filler is None or ago is None:
            # Order gets full size or pseudo-execution
            size = order.executed.remsize
        else:
            # Execution depends on volume filler
            size = self.p.filler(order, price, ago)
            if order.issell():
                size = -size

        if ago is not None:
            # Round price by factor of currency pair
            if order.ordtype:
                price = factor_down(price, order.p.data.p.name)
            else:
                price = factor_up(price, order.p.data.p.name)

        # Get comminfo object for the data
        comminfo = self.getcommissioninfo(order.data)

        # Check if something has to be compensated
        if order.data._compensate is not None:
            data = order.data._compensate
            cinfocomp = self.getcommissioninfo(data)  # for actual commission
        else:
            data = order.data
            cinfocomp = comminfo

        ccy_pair = data.p.name

        # USD cash amount is the notional and hence is fixed, e.g. USDJPY
        is_cash_fix = order.info.nccy == "USD"

        old_price = price
        old_size = size

        # Adjust position with operation size
        if ago is not None:
            price, size = price_reverse(price, size, ccy_pair, is_cash_fix)

            # Real execution with date
            position = self.positions[data]
            pprice_orig = position.price

            psize, pprice, opened, closed = position.pseudoupdate(size, price)

            # if part/all of a position has been closed, then there has been
            # a profitandloss ... record it
            pnl = comminfo.profitandloss(-closed, pprice_orig, price)
            cash = self.cash
        else:
            pnl = 0
            if not self.p.coo:
                price = pprice_orig = order.created.price
            else:
                # When doing cheat on open, the price to be considered for a
                # market order is the opening price and not the default closing
                # price with which the order was created
                if order.exectype == Order.Market:
                    price = pprice_orig = order.data.open[0]
                else:
                    price = pprice_orig = order.created.price

            price, size = price_reverse(price, size, ccy_pair, is_cash_fix)

            psize, pprice, opened, closed = position.update(size, price)

        # "Closing" totally or partially is possible. Cash may be re-injected
        if closed:
            # Adjust to returned value for closed items & acquired opened items
            if self.p.shortcash:
                closedvalue = comminfo.getvaluesize(-closed, price)
            else:
                closedvalue = comminfo.getoperationcost(closed, price)

            closecash = closedvalue
            if closedvalue > 0:  # long position closed
                closecash /= comminfo.get_leverage()  # inc cash with lever

            # Update cash after transaction
            cash += closecash

            # Calculate and substract commission
            closedcomm = comminfo.getcommission(closed, price)
            cash -= closedcomm

            if ago is not None:
                # Cashadjust closed contracts: prev close vs exec price
                # The operation can inject or take cash out
                cash += comminfo.cashadjust(-closed,
                                            position.adjbase,
                                            price)

                # Update system cash
                self.cash = cash
        else:
            closedvalue = closedcomm = 0.0

        popened = opened
        if opened:
            if self.p.shortcash:
                openedvalue = comminfo.getvaluesize(opened, price)
            else:
                openedvalue = comminfo.getoperationcost(opened, price)

            opencash = openedvalue
            if openedvalue > 0:  # long position being opened
                opencash /= comminfo.get_leverage()  # dec cash with level

            cash -= opencash  # original behavior

            openedcomm = cinfocomp.getcommission(opened, price)
            cash -= openedcomm

            if ago is not None:  # real execution
                if abs(psize) > abs(opened):
                    # some futures were opened - adjust the cash of the
                    # previously existing futures to the operation price and
                    # use that as new adjustment base, because it already is
                    # for the new futures At the end of the cycle the
                    # adjustment to the close price will be done for all open
                    # futures from a common base price with regards to the
                    # close price
                    adjsize = psize - opened
                    cash += comminfo.cashadjust(adjsize,
                                                position.adjbase, price)

                # record adjust price base for end of bar cash adjustment
                position.adjbase = price

                # update system cash - checking if opened is still != 0
                self.cash = cash
        else:
            openedvalue = openedcomm = 0.0

        if ago is None:
            # return cash from pseudo-execution
            return cash

        execsize = closed + opened

        if execsize:
            # Confimrm the operation to the comminfo object
            comminfo.confirmexec(execsize, price)

            # do a real position update if something was executed
            position.update(execsize, price, data.datetime.datetime())

            if closed and self.p.int2pnl:  # Assign accumulated interest data
                closedcomm += self.d_credit.pop(data, 0.0)

            # Revert original
            if is_cash_fix:
                # If cash amount is fixed, change size divided by price
                opened = -opened * price
                closed = -closed * price
                execsize = old_size

            if ccy_pair[:3] == "USD":
                # Reverse price if ccy1 is USD
                price = old_price

            # Execute and notify the order
            order.execute(dtcoc or data.datetime.datetime(ago),
                          execsize, price,
                          closed, closedvalue, closedcomm,
                          opened, openedvalue, openedcomm,
                          comminfo.margin, pnl,
                          psize, pprice)

            order.addcomminfo(comminfo)

            self.notify(order)
            self._ococheck(order)

            self.check_after_exec()

    def check_after_exec(self):
        """
        Checks position limit after any order is executed.
        """
        pending = collections.deque()
        """Add deque to temporarily store orders"""

        while self.pending:
            cash = self.cash
            order = self.pending.popleft()

            if order is None:
                # Skip for None
                pending.append(order)
                continue
            elif self._take_children(order) is None:
                # Children not taken
                pending.append(order)
                continue
            elif order.p.parent is not None:
                # Consider order with parent is completed
                if order.p.parent.status != Order.Completed:
                    pending.append(order)
                    continue

            comminfo = self.getcommissioninfo(order.data)
            position = self.positions[order.data].clone()

            # pseudo-execute the order to get the remaining cash after exec
            cash = self._execute(order, cash=cash, position=position)

            if abs(cash - self.startingcash) < self.limit:
                pending.append(order)
                continue

            # Cancel order if cash or position exceeds limit
            self.cancel_order(order)

        # Move order to self.pending
        while pending:
            order = pending.popleft()
            self.pending.append(order)

    def _get_value(self, datas=None, lever=False):
        pos_value = 0.0
        pos_value_unlever = 0.0
        unrealized = 0.0

        while self._cash_addition:
            c = self._cash_addition.popleft()
            self._fundshares += c / self._fundval
            self.cash += c

        for data in datas or self.positions:
            comminfo = self.getcommissioninfo(data)
            position = self.positions[data]

            # Get latest non-null close price
            j = 0
            close = data.close[j]
            data_len = len(data)
            while math.isnan(close) and (-j < data_len):
                j -= 1
                close = data.close[j]

            # Take reciprocal of close price for USD as foreign currency
            if data.p.name[:3] == "USD":
                close = 1 / close

            # use valuesize:  returns raw value, rather than negative adj val
            if not self.p.shortcash:
                dvalue = comminfo.getvalue(position, close)
            else:
                dvalue = comminfo.getvaluesize(position.size, close)

            dunrealized = comminfo.profitandloss(position.size, position.price, close)

            if datas and len(datas) == 1:
                if lever and dvalue > 0:
                    dvalue -= dunrealized
                    return (dvalue / comminfo.get_leverage()) + dunrealized
                return dvalue  # raw data value requested, short selling is neg

            if not self.p.shortcash:
                dvalue = abs(dvalue)  # short selling adds value in this case

            pos_value += dvalue
            unrealized += dunrealized

            if dvalue > 0:  # long position - unlever
                dvalue -= dunrealized
                pos_value_unlever += (dvalue / comminfo.get_leverage())
                pos_value_unlever += dunrealized
            else:
                pos_value_unlever += dvalue

        if not self._fundhist:
            self._value = self.cash + pos_value_unlever
            self._fundval = self._value / self._fundshares  # update fundvalue
        else:
            # Try to fetch a value
            fval, fvalue = self._process_fund_history()

            self._value = fvalue
            self.cash = fvalue - pos_value_unlever
            self._fundval = fval
            self._fundshares = fvalue / fval
            lev = pos_value / (pos_value_unlever or 1.0)

            # update the calculated values above to the historical values
            pos_value_unlever = fvalue
            pos_value = fvalue * lev

        self._valuemkt = pos_value_unlever

        self._valuelever = self.cash + pos_value
        self._valuemktlever = pos_value

        self._leverage = pos_value / (pos_value_unlever or 1.0)
        self._unrealized = unrealized

        return self._value if not lever else self._valuelever

    def next(self):
        super(FXBroker, self).next()

        dt = num2date(max(d.datetime[0] for d in self.cerebro.datas if len(d) > 0))

        self.print_position(dt)

    def _operate_order(self):
        """
        Operates each pending order.\n
        :return: Status if there are some pending orders.
        """
        order = self.pending.popleft()
        if order is None:
            return False

        if order.expire():
            # Quant: Delete order by BackTrader when it is expired
            self.cancel_order(order)

        elif not order.active():
            self.pending.append(order)  # cannot yet be processed

        else:
            self._try_exec(order)
            if order.alive():
                self.pending.append(order)

            elif order.status == Order.Completed:
                # a bracket parent order may have been executed
                self._bracketize(order)

        return True

    def print_live(self, dt):
        """
        Prints position messages for live data
        :return:
        """
        # Use queue to monitor latency
        dt_now = datetime.now()
        diff_t = (dt_now - dt).total_seconds()
        _ = self.timediff.get()
        self.timediff.put(diff_t)
        mean_lag = np.nanmean(self.timediff.queue)

        if diff_t > 10000:
            # If latency is too large caused by loading data,
            # print for every minute only
            if self._last_min == dt.minute:
                return
            else:
                self._last_min = dt.minute

        # Return if length of data does not increase
        len_data = 0
        for d in self.cerebro.datas:
            len_data += len(d)

        if len_data == self._last_len:
            return

        self._last_len = len_data

        len_msg = 0
        for q in self.cerebro.datas[0].o.qs.values():
            len_msg += len(q)

        msg = f"\r{dt_now} | {dt} | {self._last_len} | {len_msg} | {round(diff_t, 2)} | " \
              f"{round(mean_lag, 3)} | USD: {self.cash:,.2f} | "
        for c in self._print_ccy:
            p = self.get_print_position(c)
            msg += f"{c}: {p:,.2f} | "
        print(msg, end="\x1b[1K")  # print and carriage return

    def print_history(self, dt):
        """
        Prints position message for historical data
        :return:
        """

        if dt.minute:
            return

        # Return if length of data does not increase
        len_data = 0
        for d in self.cerebro.datas:
            len_data += len(d)

        if len_data == self._last_len:
            return

        self._last_len = len_data

        dt_now = datetime.now()
        msg = f"\r{dt_now} | {dt} | USD: {self.cash:,.2f} | "
        for c in self._print_ccy:
            p = self.get_print_position(c)
            msg += f"{c}: {p:,.2f} | "
        print(msg, end="\x1b[1K")  # print and carriage return

    def get_print_position(self, ccy):
        """
        Gets the whole position of the specified currency pair.\n
        :param ccy: Currency pair in XXXYYY.
        :return: All position in the broker.
        """
        d_ccys = [v for k, v in self.cerebro.datasbyname.items() if ccy in k]
        p = 0
        for d in d_ccys:
            p += self.getposition(d).size

        return p
