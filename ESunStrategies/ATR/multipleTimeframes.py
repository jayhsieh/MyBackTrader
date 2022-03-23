import math
import datetime
import backtrader as bt
import pandas as pd
import configparser


class Intra15MinutesReverseStrategy(bt.Strategy):
    params = dict(
        last_min=-1,
        last_day=-1,
        is_new_day=False,
        # order parameters
        trigger=2,
        limit=5,
        factor=0,
    )

    def __init__(self, name=None):
        if name is None:
            name = self.getdatanames()[0]

        self.name = name
        # data = [d for d in self.datas if name in d._name]
        self.trade_data = self.datas[0]     # 1m
        self.index_data = self.datas[1]     # 15m

        self.cnt = {self.trade_data._name: 0,
                    self.index_data._name: 0}

        self.atr = bt.indicators.ATR(self.index_data)
        self.order_buy = list()
        self.order_sell = list()
        self.order_buy_close = list()
        self.order_sell_close = list()
        self.order_close = list()

        # order parameters
        self.p.factor = 0.00001 * self.get_factor(self.name)

        self.transaction = pd.DataFrame(columns=['amount', 'price', 'value', 'atr'])

        # List ccy pairs to print outstanding positions
        self._print_ccy = []
        for c in self.positionsbyname.keys():
            if c[:6] in self._print_ccy:
                continue
            self._print_ccy.append(c[:6])

        # End date of strategy
        # today = datetime.datetime.today()
        # self.end_date = today + datetime.timedelta(days=6 - today.weekday())
        # self.end_date = datetime.datetime.combine(self.end_date, datetime.datetime.min.time())

        # self.end_date = datetime.datetime.now() + datetime.timedelta(minutes=4)

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or max(d.datetime.datetime() for d in self.datas if len(d) > 0)
            print(f'\r{dt.isoformat()}, {txt}')

    @staticmethod
    def get_factor(target):
        config = configparser.ConfigParser()
        config.read('settings.ini')
        if target in config['factor']:
            return float(config['factor'][target])
        else:
            return 1

    def notify_order(self, order):
        if order.status in [order.Accepted]:
            pass

        if order.status in [order.Submitted]:
            ord_type = order.OrdTypes[order.ordtype]
            exec_type = order.ExecTypes[order.exectype]

            msg = f'Order Record: Ref: {order.ref:0>4}, Ccy: {order.data._name}, ' \
                  f'Type: {ord_type: <4}, Execute Type: {exec_type}'

            if order.exectype in [order.Market]:
                msg += f', Amount: {order.p.size:,}'
            elif order.exectype in [order.Limit]:
                deadline = bt.num2date(order.valid)
                msg += f', Limit Price: {order.price:.5f}, ' \
                       f'Amount: {order.p.size:,}, Deadline: {deadline}'
            elif order.exectype in [order.StopLimit]:
                deadline = bt.num2date(order.valid)
                msg += f', Limit Price: {order.plimit:.5f}, '
                if order.price is not None:
                    msg += f'Init Price: {order.price:.5f}, '
                msg += f'Amount: {order.p.size:,}, Deadline: {deadline}'

            self.log(msg, doprint=True)

        elif order.status in [order.Completed]:
            if order.exectype in [order.StopTrail, order.StopTrailLimit]:
                print('\rExecute stop order')
                return

            msg = 'Order Executed: '
            tr = None
            if order.isbuy():
                val = -order.executed.size * order.executed.price
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.trade_data.datetime.datetime(0)])
                msg += "Buy "
            elif order.issell():
                val = -order.executed.size * order.executed.price
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.trade_data.datetime.datetime(0)])
                msg += "Sell "

            msg += f"Order: Ref: {order.ref:0>4}, Ccy: {order.data._name}, " \
                   f"Exec Price: {round(order.executed.price, 8)}, Size: {order.executed.size:,.2f}"
            if order.exectype == 4 and isinstance(order.price, float):
                atr = abs(order.price - order.plimit) / (self.p.limit - self.p.trigger)
                msg += f", ATR: {atr:>6f}"
                tr.loc[self.trade_data.datetime.datetime(0), 'atr'] = atr

            self.log(msg, doprint=True)
            self.transaction = self.transaction.append(tr)

        elif order.status in [order.Canceled]:
            self.log(f"Order Canceled: Ref: {order.ref:0>4}, Limit Price: {order.plimit}", doprint=True)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'Trade Record: Ccy: {self.name}, P&L: {trade.pnl:,.2f}, Net P&L: {trade.pnlcomm:,.2f}, '
                 f'Market Value: {self.broker.getvalue():,.2f}',
                 doprint=True)
        self.log('===========================================================================', doprint=True)

    # def prenext(self):
    #     self._print_data()
    #
    # def nextstart(self):
    #     self._print_data()

    def next(self):
        self._print_data()

        # Set order for the first time in second
        dt = max(d.datetime.datetime() for d in self.datas if len(d) > 0)

        # TODO: something wrong here, it always false except the first in the loop
        if dt.minute != self.p.last_min:
            dt = dt.replace(microsecond=0)
            dt = dt.replace(second=0)

            # self._print_position(dt)

            self._order(dt)
            self.p.last_min = dt.minute

            # if dt > self.end_date:
            #     self.cerebro.runstop()

    def _print_data(self):
        """
        Print data when broker receive the new one.
        :return:
        """
        for d in [self.trade_data, self.index_data]:
            name = d._name
            if len(d) == 0:
                continue
            elif len(d) == self.cnt[name]:
                continue

            msg = f'{name}, {len(d)}, {bt.num2date(d.datetime[0])},' \
                  f' {d.open[0]}, {d.high[0]}, {d.low[0]}, {d.close[0]}'
            self.log(msg, doprint=True)
            self.cnt[name] = len(d)

    def _print_position(self, dt):
        """
        Prints outstanding position with datetime now.
        param dt: Datetime now
        :return:
        """
        msg = f"\r{dt} | USD: {self.broker.cash:,.2f} | "
        for c in self._print_ccy:
            p = self.positionsbyname[c]
            msg += f"{c}: {p.size:,.2f} | "
        print(msg)

    def _order(self, when):
        """
        Sets order determined with time
        param when: Time to set order
        return:
        """
        # if when.second:
        #     return

        # Skip for the first atr value
        if math.isnan(self.atr[-1]):
            return

        size = self.getposition(self.trade_data).size
        criteria = abs(self.broker.getvalue(datas=[self.trade_data]) / self.broker.startingcash)
        is_buy = (size < 0) ^ (self.name[:3] == "USD")

        # Stop trading and clean position in the last 30 minutes
        if (when.hour == 23) and (when.minute >= 30):
            if when.day != self.p.last_day:
                self.p.last_day = when.day
                self.p.is_new_day = True
                if abs(size) > 0:
                    ccy = self.name.replace("USD", "")
                    self.log(f"Clean all position at day end: {size:,.2f} {ccy}", doprint=True)
                    if size < 0:
                        self.order_buy = self.buy(data=self.trade_data, exectype=bt.Order.Market,
                                                  size=abs(size), nccy=ccy)
                    else:
                        self.order_sell = self.sell(data=self.trade_data, exectype=bt.Order.Market,
                                                    size=abs(size), nccy=ccy)
            return
        elif (when.hour == 0) and (when.minute < 30):
            if self.p.is_new_day:
                self.p.is_new_day = False
                if len(self._trades[self.trade_data][0]):
                    self._trades[self.trade_data][0].pop()
                    self.log('===========================================================================',
                             doprint=True)
            return

        if when.minute % 15 == 10:
            # For the last 5 min, create close order for outstanding position
            valid2 = when + datetime.timedelta(minutes=5)

            if criteria < 0.5:
                return

            if is_buy:
                self.order_sell_close = self.buy(data=self.trade_data, exectype=bt.Order.StopLimit,
                                                 plimit=self.order_sell_close.plimit, valid=valid2)
            else:
                self.order_buy_close = self.sell(data=self.trade_data, exectype=bt.Order.StopLimit,
                                                 plimit=self.order_buy_close.plimit, valid=valid2)
            return
        elif when.minute % 15:
            return

        valid1 = when + datetime.timedelta(minutes=10)
        if criteria < 0.5:
            # 賺取向下偏離過多的reversion
            buy_trigger_price = math.floor(
                (self.index_data.low[-1] - self.p.trigger * self.atr[-1]) / self.p.factor) * self.p.factor
            buy_limit_price = math.ceil(
                (self.index_data.low[-1] - self.p.limit * self.atr[-1]) / self.p.factor) * self.p.factor

            self.order_buy = self.buy(data=self.trade_data, exectype=bt.Order.StopLimit,
                                      price=buy_trigger_price, plimit=buy_limit_price, valid=valid1)
            self.order_buy_close = self.sell(data=self.trade_data, exectype=bt.Order.StopLimit,
                                             price=buy_limit_price, plimit=buy_trigger_price, valid=valid1)

            # 賺取向上偏離過多的reversion
            sell_trigger_price = math.ceil(
                (self.index_data.high[-1] + self.p.trigger * self.atr[-1]) / self.p.factor) * self.p.factor
            sell_limit_price = math.floor(
                (self.index_data.high[-1] + self.p.limit * self.atr[-1]) / self.p.factor) * self.p.factor
            self.order_sell = self.sell(data=self.trade_data, exectype=bt.Order.StopLimit,
                                        price=sell_trigger_price, plimit=sell_limit_price, valid=valid1)
            self.order_sell_close = self.buy(data=self.trade_data, exectype=bt.Order.StopLimit,
                                             price=sell_limit_price, plimit=sell_trigger_price, valid=valid1)
        else:
            # Close position
            if size != 0:
                if size > 0:
                    self.log(
                        f"Did not touch buy trigger price: {round(self.order_buy_close.plimit, 8)}",
                        doprint=True)
                elif size < 0:
                    self.log(
                        f"Did not touch sell trigger price: {round(self.order_sell_close.plimit, 8)}",
                        doprint=True)

                self.log("Close position on next open market price", doprint=True)

                if is_buy:
                    self.order_buy = self.buy(data=self.trade_data, exectype=bt.Order.Market)
                else:
                    self.order_sell = self.sell(data=self.trade_data, exectype=bt.Order.Market)
