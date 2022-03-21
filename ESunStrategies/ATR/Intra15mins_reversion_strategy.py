import math
import datetime
import backtrader as bt
import configparser
import pandas as pd


class Intra15MinutesReverseStrategy(bt.Strategy):
    params = dict(
        last_sec=-1,
        # order parameters
        trigger=2,
        limit=5,
        factor=0,
    )

    def __init__(self, name=None):
        if name is None:
            name = self.getdatanames()[0].split('_')[0]

        self.name = name
        data = [d for d in self.datas if name in d._name]
        self.trade_data = data[0]
        self.index_data = data[1]

        self.cnt = {self.trade_data._name: 0,
                    self.index_data._name: 0}

        self.atr = bt.indicators.ATR(self.index_data)
        self.order_buy = list()
        self.order_sell = list()
        self.order_buy_close = list()
        self.order_sell_close = list()
        self.order_close = list()

        # order parameters
        self.config = configparser.ConfigParser()
        self.p.factor = 100000 * int(self.config['factor'][name])
        self.transaction = pd.DataFrame(columns=['amount', 'price', 'value', 'atr'])

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or max(d.datetime.datetime() for d in self.datas if len(d) > 0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Accepted]:
            pass

        if order.status in [order.Submitted]:
            ord_type = order.OrdTypes[order.ordtype]
            exec_type = order.ExecTypes[order.exectype]

            msg = f'下單紀錄: 編號: {order.ref:0>4}, 幣別: {order.data._name}, ' \
                  f'買賣: {ord_type: <4}, 類型: {exec_type}'

            if order.exectype in [order.StopLimit]:
                deadline = bt.num2date(order.valid)
                msg += f', 限價價格: {order.plimit:.5f}, '
                if order.price is not None:
                    msg += f'啟動價格: {order.price:.5f}, '
                msg += f'數量: {order.p.size:,}, 期限； {deadline}'

            self.log(msg, doprint=True)

        elif order.status in [order.Completed]:
            if order.exectype in [order.StopTrail, order.StopTrailLimit]:
                print('執行停損單')
                return

            msg = ''
            tr = None
            if order.isbuy():
                val = -order.executed.size * order.executed.price
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.trade_data.datetime.datetime(0)])
                msg = "買"
            elif order.issell():
                val = -order.executed.size * order.executed.price
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.trade_data.datetime.datetime(0)])
                msg = "賣"

            msg += f"單執行: 編號: {order.ref:0>4}, 幣別: {order.data._name}, " \
                   f"執行價格: {order.executed.price}, 部位大小: {order.executed.size}"
            if order.exectype == 4 and isinstance(order.price, float):
                atr = abs(order.price - order.plimit) / (self.p.limit - self.p.trigger)
                msg = msg + ", ATR: {}".format(atr)
                tr.loc[self.trade_data.datetime.datetime(0), 'atr'] = atr

            self.log(msg, doprint=True)
            self.transaction = self.transaction.append(tr)

        elif order.status in [order.Canceled]:
            self.log(f"Order Canceled: 編號: {order.ref:0>4}, 限價價格: {order.plimit}",
                     doprint=True)

    # def prenext(self):
    #     self.next()
    #
    # def nextstart(self):
    #     self.is_started = True
    #     self.next()

    def next(self):
        # self._print_data()

        # Set order for the first time in second
        dt = max(d.datetime.datetime() for d in self.datas if len(d) > 0)
        if dt.second != self.p.last_sec:
            dt = dt.replace(microsecond=0)
            self._order(dt)
            self.p.last_sec = dt.second

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'交易紀錄: 毛利：{trade.pnl:.2f}, 淨利: {trade.pnlcomm:.2f}, '
                 f'市值: {self.broker.getvalue():.2f}',
                 doprint=True)
        self.log('===========================================================================', doprint=True)

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

    def _order(self, when):
        """
        Sets order determined with time
        :param when: Time to set order
        :return:
        """
        # if when.second:
        #     return

        size = self.getposition(self.trade_data).size

        if when.second % 15 == 10:
            # For the last 5 min, create close order for outstanding position
            valid2 = when + datetime.timedelta(seconds=5)

            if size > 0:
                self.order_buy_close = self.sell(data=self.trade_data, exectype=bt.Order.StopLimit,
                                                 plimit=self.order_buy_close.plimit, valid=valid2,
                                                 size=abs(size))
            elif size < 0:
                self.order_sell_close = self.buy(data=self.trade_data, exectype=bt.Order.StopLimit,
                                                 plimit=self.order_sell_close.plimit, valid=valid2,
                                                 size=abs(size))
            return
        elif when.second % 15:
            return

        valid1 = when + datetime.timedelta(seconds=10)

        if not size:
            # Skip for the first atr value
            if math.isnan(self.atr[-1]):
                return

            # 賺取向下偏離過多的reversion
            buy_trigger_price = math.floor(
                (self.index_data.low[-1] - self.p.trigger * self.atr[-1]) * self.p.factor) / self.p.factor
            buy_limit_price = math.ceil(
                (self.index_data.low[-1] - self.p.limit * self.atr[-1]) * self.p.factor) / self.p.factor

            self.order_buy = self.buy(data=self.trade_data, exectype=bt.Order.StopLimit,
                                      price=buy_trigger_price, plimit=buy_limit_price, valid=valid1)
            self.order_buy_close = self.sell(data=self.trade_data, exectype=bt.Order.StopLimit,
                                             price=buy_limit_price, plimit=buy_trigger_price, valid=valid1)

            # 賺取向上偏離過多的reversion
            sell_trigger_price = math.ceil(
                (self.index_data.high[-1] + self.p.trigger * self.atr[-1]) * self.p.factor) / self.p.factor
            sell_limit_price = math.floor(
                (self.index_data.high[-1] + self.p.limit * self.atr[-1]) * self.p.factor) / self.p.factor
            self.order_sell = self.sell(data=self.trade_data, exectype=bt.Order.StopLimit,
                                        price=sell_trigger_price, plimit=sell_limit_price, valid=valid1)
            self.order_sell_close = self.buy(data=self.trade_data, exectype=bt.Order.StopLimit,
                                             price=sell_limit_price, plimit=sell_trigger_price, valid=valid1)
        else:
            # Close position
            if size != 0:
                if size > 0:
                    self.log(
                        "Did not touch buy trigger price: {}".format(self.order_buy_close.plimit),
                        doprint=True)
                elif size < 0:
                    self.log(
                        "Did not touch sell trigger price: {}".format(self.order_sell_close.plimit),
                        doprint=True)

                self.log("Close position on next open market price", doprint=True)
                self.order_close = self.close(data=self.trade_data)
