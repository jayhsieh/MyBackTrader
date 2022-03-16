import math
import datetime
import backtrader as bt
import pandas as pd
import configparser


class Intra15MinutesReverseStrategy(bt.Strategy):

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data1)
        self.order_buy = list()
        self.order_sell = list()
        self.order_buy_close = list()
        self.order_sell_close = list()
        self.order_close = list()

        # order parameters
        self.factor = 100000 * self.get_factor(self.data1._name.split('_')[0])
        self.trigger_param = 2
        self.limit_param = 5
        self.transaction = pd.DataFrame(columns=['amount', 'price', 'value', 'atr'])

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
                val = -order.executed.price * order.executed.size
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.data0.datetime.datetime(0)])
                msg = "買單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(
                    order.ref, order.executed.price, order.executed.comm, order.executed.size)
                if order.exectype == 4 and isinstance(order.price, float):
                    # ExecTypes 0: Market, 1: Close, 2: Limit, 3: Stop, 4: StopLimit, 5: StopTrail, 6: StopTrailLimit,
                    # 7: Historical
                    atr = abs(order.price - order.plimit) / (self.limit_param - self.trigger_param)
                    msg = msg + ", ATR: {}".format(atr)
                    tr.loc[self.data0.datetime.datetime(0), 'atr'] = atr

                self.log(msg, doprint=True)
                self.transaction = self.transaction.append(tr)
            elif order.issell():
                val = -order.executed.price * order.executed.size
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.data0.datetime.datetime(0)])
                msg = "賣單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(
                    order.ref, order.executed.price, order.executed.comm, order.executed.size)
                if order.exectype == 4 and isinstance(order.price, float):
                    atr = abs(order.price - order.plimit) / (self.limit_param - self.trigger_param)
                    msg = msg + ", ATR: {}".format(atr)
                    tr.loc[self.data0.datetime.datetime(0), 'atr'] = atr

                self.log(msg, doprint=True)
                self.transaction = self.transaction.append(tr)
        elif order.status in [order.Canceled]:
            self.log("Order Canceled: 訂單編號: {}, 限價價格: {}".format(order.ref, order.plimit), doprint=True)

    def next(self):
        for d in self.datas:
            msg = f'{d._name}, {len(d)}, {bt.num2date(d.datetime[0])},' \
                  f' {d.open[0]}, {d.high[0]}, {d.low[0]}, {d.close[0]}'
            self.log(msg, doprint=True)

        if self.data0.datetime.datetime(0).minute % 15 == 10:
            # For the last 5 min, create close order for outstanding position
            position_size = self.getposition().size
            valid2 = datetime.timedelta(minutes=5)

            if position_size > 0:
                self.order_buy_close = self.sell(exectype=bt.Order.StopLimit, plimit=self.order_buy_close.plimit,
                                                 valid=valid2, size=abs(position_size))
            elif position_size < 0:
                self.order_sell_close = self.buy(exectype=bt.Order.StopLimit, plimit=self.order_sell_close.plimit,
                                                 valid=valid2, size=abs(position_size))
            return
        elif self.data0.datetime.datetime(0).minute % 15:
            return

        # print_str = 'Open: {}, High: {}, Low: {}, Close: {}, ATR: {:5f}, index: {} '.format(
        #     self.data1.open[0], self.data1.high[0], self.data1.low[0], self.data1.close[0], self.atr[0], len(self))
        # self.log(print_str, doprint=True)  # for debug

        valid1 = datetime.timedelta(minutes=10)

        if not self.position:
            # skip for the first atr value
            if math.isnan(self.atr[-1]):
                return

            # 賺取向下偏離過多的reversion
            buy_trigger_price = math.floor(
                (self.data1.low[-1] - self.trigger_param * self.atr[-1]) * self.factor) / self.factor
            buy_limit_price = math.ceil(
                (self.data1.low[-1] - self.limit_param * self.atr[-1]) * self.factor) / self.factor

            self.order_buy = self.buy(exectype=bt.Order.StopLimit, price=buy_trigger_price,
                                      plimit=buy_limit_price, valid=valid1)
            self.order_buy_close = self.sell(exectype=bt.Order.StopLimit, price=buy_limit_price,
                                             plimit=buy_trigger_price, valid=valid1)

            # 賺取向上偏離過多的reversion
            sell_trigger_price = math.ceil(
                (self.data1.high[-1] + self.trigger_param * self.atr[-1]) * self.factor) / self.factor
            sell_limit_price = math.floor(
                (self.data1.high[-1] + self.limit_param * self.atr[-1]) * self.factor) / self.factor
            self.order_sell = self.sell(exectype=bt.Order.StopLimit, price=sell_trigger_price,
                                        plimit=sell_limit_price, valid=valid1)
            self.order_sell_close = self.buy(exectype=bt.Order.StopLimit, price=sell_limit_price,
                                             plimit=sell_trigger_price, valid=valid1)
        else:
            position_size = self.getposition().size
            if position_size != 0:
                if position_size > 0:
                    self.log(
                        "Did not touch buy trigger price: {}".format(self.order_buy_close.plimit),
                        doprint=True)
                elif position_size < 0:
                    self.log(
                        "Did not touch sell trigger price: {}".format(self.order_sell_close.plimit),
                        doprint=True)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('交易紀錄： 毛利：%.4f, 淨利：%.4f, 手續費：%.4f, 市值：%.2f, 現金：%.2f' %
                 (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()),
                 doprint=True)
        self.log('===========================================================================', doprint=True)
