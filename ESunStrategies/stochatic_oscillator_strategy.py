import math
import datetime
import backtrader as bt
import pandas as pd
import configparser


class StochasticOscillatorStrategy(bt.Strategy):

    def __init__(self):
        self.so = bt.indicators.Stochastic(self.data1, period=21)
        self.hi = bt.indicators.Highest(self.data1, period=21)
        self.lo = bt.indicators.Lowest(self.data1, period=21)
        self.order_buy = list()
        self.order_sell = list()
        self.order_buy_close = list()
        self.order_sell_close = list()
        self.order_close = list()

        # order parameters
        self.factor = 100000 * self.get_factor(self.data1._name.split('_')[0])
        self.upper_barrier = 95
        self.lower_barrier = 5
        self.holding_cnt = 0
        self.holding_limit = 13

        self.transaction = pd.DataFrame(columns=['amount', 'price', 'value'])

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('%s, %s' % (dt.isoformat(), txt))

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
            txt = "買" if order.isbuy() else "賣"
            # self.log("order.Submitted {}單執行: 訂單編號: {}".format(txt, order.ref), doprint=True)  # for debug

        if order.status in [order.Completed]:
            if order.exectype in [order.StopTrail, order.StopTrailLimit]:
                print('執行停損單')
            if order.isbuy():
                val = -order.executed.size * order.executed.price
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val]],
                                  columns=self.transaction.columns,
                                  index=[self.data0.datetime.datetime(0)])
                msg = "買單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(
                    order.ref, order.executed.price, order.executed.comm, order.executed.size)

                self.log(msg, doprint=True)
                self.transaction = self.transaction.append(tr)
            elif order.issell():
                val = -order.executed.size * order.executed.price
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val]],
                                  columns=self.transaction.columns,
                                  index=[self.data0.datetime.datetime(0)])
                msg = "賣單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(
                    order.ref, order.executed.price, order.executed.comm, order.executed.size)

                self.log(msg, doprint=True)
                self.transaction = self.transaction.append(tr)
        elif order.status in [order.Canceled]:
            self.log("Order Canceled: 訂單編號: {}, 限價價格: {}".format(order.ref, order.plimit), doprint=True)

    def next(self):
        time_minute = self.data0.datetime.datetime(0).minute % 15
        if time_minute not in [0, 14]:
            return

        # skip for the first stochastic oscillator value
        if math.isnan(self.so[-2]):
            return

        if time_minute == 0:
            size = self.getposition().size
            if abs(size) > 0:
                self.holding_cnt += 1
            else:
                self.holding_cnt = 0

            valid1 = datetime.timedelta(minutes=self.holding_limit * 15 - 1)
            middle = (self.hi[-1] + self.lo[-1]) * 0.5
            if self.so[-1] > self.lower_barrier > self.so[-2]:
                # Buy if it breaks through the lower barrier
                if size < 0:
                    self.order_close = self.close()

                if size <= 0:
                    self.order_buy = self.buy(exectype=bt.Order.Market)
                    self.holding_cnt = 1

                    self.order_buy_close = self.sell(exectype=bt.Order.Limit, price=middle,
                                                     size=self.order_buy.size, valid=valid1)

            elif self.so[-1] < self.upper_barrier < self.so[-2]:
                # Sell if it falls from the upper barrier
                if size > 0:
                    self.order_close = self.close()

                if size >= 0:
                    self.order_sell = self.sell(exectype=bt.Order.Market)
                    self.holding_cnt = 1

                    self.order_sell_close = self.buy(exectype=bt.Order.Limit, price=middle,
                                                     size=self.order_sell.size, valid=valid1)
        elif time_minute == 14:
            size = self.getposition().size
            if abs(size) == 0:
                self.holding_cnt = 0
                return

            if self.holding_cnt >= self.holding_limit:
                # Close if it is held for a long time
                self.log("Close position on next open market price", doprint=True)
                self.order_close = self.close()
                self.holding_cnt = 0

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('交易紀錄： 毛利：%.4f, 淨利：%.4f, 手續費：%.4f, 市值：%.2f, 現金：%.2f' %
                 (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()),
                 doprint=True)
        self.log('===========================================================================', doprint=True)
