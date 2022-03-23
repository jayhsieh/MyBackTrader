import datetime
import queue
import math
import numpy as np
from backtrader.order import Order, BuyOrder, SellOrder
from .bbroker import BackBroker


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

        3. In ``check_submitted``, we always allow order even if negative cash

      TODO: Transaction for cross currencies need to be considered in the future.
    """

    def __init__(self):
        super(FXBroker, self).__init__()

        self._last_min = -1
        self.timediff = queue.Queue(50)
        for i in range(50):
            self.timediff.put(float('nan'))

    def start(self):
        super(FXBroker, self).start()

        # List ccy pairs to print outstanding positions
        self._print_ccy = []
        for d in self.cerebro.datas:
            c = d._name
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
            ccys = data._name
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
            ccys = data._name
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

    def check_submitted(self):
        cash = self.cash
        positions = dict()

        while self.submitted:
            order = self.submitted.popleft()

            if self._take_children(order) is None:  # children not taken
                continue

            comminfo = self.getcommissioninfo(order.data)

            position = positions.setdefault(
                order.data, self.positions[order.data].clone())

            # pseudo-execute the order to get the remaining cash after exec
            cash = self._execute(order, cash=cash, position=position)

            # Always accept order even if cash is negative
            self.submit_accept(order)
            continue

    def _execute(self, order, ago=None, price=None, cash=None, position=None,
                 dtcoc=None):
        # ago = None is used a flag for pseudo execution
        if ago is not None and price is None:
            return  # no psuedo exec no price - no execution

        if self.p.filler is None or ago is None:
            # Order gets full size or pseudo-execution
            size = order.executed.remsize
        else:
            # Execution depends on volume filler
            size = self.p.filler(order, price, ago)
            if not order.isbuy():
                size = -size

        # Get comminfo object for the data
        comminfo = self.getcommissioninfo(order.data)

        # Check if something has to be compensated
        if order.data._compensate is not None:
            data = order.data._compensate
            cinfocomp = self.getcommissioninfo(data)  # for actual commission
        else:
            data = order.data
            cinfocomp = comminfo

        ccy_pair = data._name
        is_cash_fix = order.info.nccy == "USD"

        old_price = price
        old_size = size

        # Adjust position with operation size
        if ago is not None:
            if ccy_pair[:3] == "USD":
                # Reverse price if ccy1 is USD
                price = 1 / price

            if is_cash_fix:
                # If cash amount is fixed, change size divided by price
                size = -size / price

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

            if ccy_pair[:3] == "USD":
                # Reverse price if ccy1 is USD
                price = 1 / price

            if is_cash_fix:
                # If cash amount is fixed, change size divided by price
                size = -size / price

            psize, pprice, opened, closed = position.update(size, price)

        # "Closing" totally or partially is possible. Cash may be re-injected
        if closed:
            # Adjust to returned value for closed items & acquired opened items
            if self.p.shortcash:
                closedvalue = comminfo.getvaluesize(-closed, pprice_orig)
            else:
                closedvalue = comminfo.getoperationcost(closed, pprice_orig)

            closecash = closedvalue
            if closedvalue > 0:  # long position closed
                closecash /= comminfo.get_leverage()  # inc cash with lever

            cash += closecash + pnl * comminfo.stocklike
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

            # Revert origional
            if is_cash_fix:
                # If cash amount is fixed, change size divided by price
                opened = -opened * price
                closed = -closed * price
                execsize = old_size

            if ccy_pair[:3] == "USD":
                # Reverse price if ccy1 is USD
                price = old_price

            # Execute and notify the order
            order.execute(dtcoc or data.datetime[ago],
                          execsize, price,
                          closed, closedvalue, closedcomm,
                          opened, openedvalue, openedcomm,
                          comminfo.margin, pnl,
                          psize, pprice)

            order.addcomminfo(comminfo)

            self.notify(order)
            self._ococheck(order)

        if popened and not opened:
            # opened was not executed - not enough cash
            order.margin()
            self.notify(order)
            self._ococheck(order)
            self._bracketize(order, cancel=True)

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
            while math.isnan(close) and (-j < len(data)):
                j -= 1
                close = data.close[j]

            # Take reciprocal of close price for USD as foreign currency
            if data._name[:3] == "USD":
                close = 1 / close

            # use valuesize:  returns raw value, rather than negative adj val
            if not self.p.shortcash:
                dvalue = comminfo.getvalue(position, close)
            else:
                dvalue = comminfo.getvaluesize(position.size, close)

            dunrealized = comminfo.profitandloss(position.size, position.price,
                                                 close)
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
            self._value = v = self.cash + pos_value_unlever
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

        # Set order for the first time in second
        dt = max(d.datetime.datetime() for d in self.cerebro.datas if len(d) > 0)
        self._last_min = dt.minute

        # Use queue to monitor latency
        dt_now = datetime.datetime.now()
        _ = self.timediff.get()
        self.timediff.put((dt_now - dt).total_seconds())
        mean_lag = np.nanmean(self.timediff.queue)

        msg = f"\r{dt_now} | {dt} | {round(mean_lag, 3)} | USD: {self.cash:,.2f} | "
        for c in self._print_ccy:
            p = self.getposition(self.cerebro.datasbyname[c])
            msg += f"{c}: {p.size:,.2f} | "
        print(msg, end="\x1b[1K")
