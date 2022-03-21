import math
import datetime
import backtrader as bt
import configparser


class TestStrategy(bt.Strategy):
    params = dict(
        last_sec=-1,
        last_period=-1,
        period_len=15,
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

        self.order_buy = list()
        self.order_sell = list()

        self.config = configparser.ConfigParser()
        self.p.factor = 100000 * int(self.config['factor'][name])

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

            if order.exectype in [order.Market]:
                msg += f', 數量: {order.p.size:,}'
            elif order.exectype in [order.Limit]:
                deadline = bt.num2date(order.valid)
                msg += f', 限價價格: {order.price:.5f}, ' \
                       f'數量: {order.p.size:,}, 期限； {deadline}'
            elif order.exectype in [order.StopLimit]:
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
            if order.isbuy():
                msg = "買"
            elif order.issell():
                msg = "賣"

            msg += f"單執行: 編號: {order.ref:0>4}, 幣別: {order.data._name}, " \
                   f"執行價格: {order.executed.price}, 部位大小: {order.executed.size:,}"

            self.log(msg, doprint=True)

        elif order.status in [order.Canceled]:
            self.log(f"Order Canceled: 編號: {order.ref:0>4}, 限價價格: {order.plimit}",
                     doprint=True)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'交易紀錄: 毛利：{trade.pnl:.2f}, 淨利: {trade.pnlcomm:.2f}, '
                 f'市值: {self.broker.getvalue():.2f}',
                 doprint=True)
        self.log('===========================================================================', doprint=True)

    def prenext(self):
        self._print_data()

    def nextstart(self):
        self._print_data()

    def next(self):
        self._print_data()

        # Set order for the first time in second
        dt = max(d.datetime.datetime() for d in self.datas if len(d) > 0)
        if dt.second != self.p.last_sec:
            dt = dt.replace(microsecond=0)
            self._order(dt)
            self.p.last_sec = dt.second

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
        period = when.second % self.p.period_len
        if period == self.p.last_period:
            return

        self.p.last_period = period

        if (when.second % self.p.period_len) / self.p.period_len > 0.67:
            return

        valid1 = when + datetime.timedelta(seconds=10)

        size = self.getposition(self.trade_data).size
        criteria = abs(self.broker.getvalue(datas=[self.trade_data]) / self.broker.startingcash)
        if criteria < 0.5:
            buy_price = math.floor(self.index_data.low[-1] * 1.1 * self.p.factor) / self.p.factor
            self.order_buy = self.buy(data=self.trade_data, exectype=bt.Order.Limit,
                                      price=buy_price, valid=valid1)

            # sell_price = math.ceil(self.index_data.high[-1] * 0.9 * self.p.factor) / self.p.factor
            # self.order_sell = self.sell(data=self.trade_data, exectype=bt.Order.Limit,
            #                             price=sell_price, size=1000000, valid=valid1)
        else:
            # Close position
            if size != 0:
                self.log("Close position on next open market price", doprint=True)
                is_buy = (size < 0) ^ (self.trade_data._name[:3] == "USD")
                if is_buy:
                    self.order_buy = self.buy(data=self.trade_data, exectype=bt.Order.Market)
                else:
                    self.order_sell = self.sell(data=self.trade_data, exectype=bt.Order.Market)
