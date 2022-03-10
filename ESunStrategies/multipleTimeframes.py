import math
import datetime
import backtrader as bt


class Intra15MinutesReverseStrategy(bt.Strategy):

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data1)
        self.order_buy = list()
        self.order_sell = list()
        self.order_buy_close = list()
        self.order_sell_close = list()
        self.order_close = list()

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('%s, %s' % (dt.isoformat(), txt))

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
                self.log("買單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(order.ref, order.executed.price,
                         order.executed.comm, order.executed.size), doprint=True)
            elif order.issell():
                self.log("賣單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(order.ref, order.executed.price,
                         order.executed.comm, order.executed.size), doprint=True)
        elif order.status in [order.Canceled]:
            self.log("Order Canceled: 訂單編號: {}, 限價價格: {}".format(order.ref, order.plimit), doprint=True)

    def next(self):
        if self.data0.datetime.datetime(0).minute % 15:
            return

        # print_str = 'Open: {}, High: {}, Low: {}, Close: {}, ATR: {:5f}, index: {} '.format(
        #     self.data1.open[0], self.data1.high[0], self.data1.low[0], self.data1.close[0], self.atr[0], len(self))
        # self.log(print_str, doprint=True)  # for debug

        valid1 = datetime.timedelta(minutes=10)
        multiplier = 100000
        trigger_param = 2
        limit_param = 5

        if not self.position:
            # skip for the first atr value
            if math.isnan(self.atr[-1]):
                return

            # 賺取向下偏離過多的reversion
            buy_trigger_price = math.floor(
                (self.data1.low[-1] - trigger_param * self.atr[-1]) * multiplier) / multiplier
            buy_limit_price = math.ceil(
                (self.data1.low[-1] - limit_param * self.atr[-1]) * multiplier) / multiplier

            self.order_buy = self.buy(exectype=bt.Order.StopLimit, price=buy_trigger_price,
                                      plimit=buy_limit_price, valid=valid1)
            self.order_buy_close = self.sell(exectype=bt.Order.StopLimit, price=buy_limit_price,
                                             plimit=buy_trigger_price, valid=valid1)

            # 賺取向上偏離過多的reversion
            sell_trigger_price = math.ceil(
                (self.data1.high[-1] + trigger_param * self.atr[-1]) * multiplier) / multiplier
            sell_limit_price = math.floor(
                (self.data1.high[-1] + limit_param * self.atr[-1]) * multiplier) / multiplier
            self.order_sell = self.sell(exectype=bt.Order.StopLimit, price=sell_trigger_price,
                                        plimit=sell_limit_price, valid=valid1)
            self.order_sell_close = self.buy(exectype=bt.Order.StopLimit, price=sell_limit_price,
                                             plimit=sell_trigger_price, valid=valid1)
        else:
            position_size = self.getposition().size
            if position_size != 0:
                self.log("Close position on next open market price", doprint=True)
                self.order_close = self.close()

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('交易紀錄： 毛利：%.4f, 淨利：%.4f, 手續費：%.4f, 市值：%.2f, 現金：%.2f' %
                 (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()),
                 doprint=True)
        self.log('===========================================================================', doprint=True)
