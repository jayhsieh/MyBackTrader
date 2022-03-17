import math
import datetime
import backtrader as bt
import pandas as pd
import configparser


class Intra15MinutesReverseStrategy(bt.Strategy):

    def __init__(self, name=None):
        if name is None:
            name = self.getdatanames()[0]

        self.atr = bt.indicators.ATR(self.data1)
        self.order_buy = list()
        self.order_sell = list()
        self.order_buy_close = list()
        self.order_sell_close = list()
        self.order_close = list()

        # order parameters
        self.factor = 100000 * self.get_factor(name)
        self.trigger_param = 2
        self.limit_param = 5

        self.transaction = pd.DataFrame(columns=['amount', 'price', 'value', 'atr'])

        self.add_timer(
            when=bt.timer.SESSION_START,
            repeat=datetime.timedelta(seconds=5),
            cheat=True
        )

        self.is_started = False
        self.trade_data = self.getdatabyname(name)

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
            ord_type = order.OrdTypes[order.ordtype]
            exec_type = order.ExecTypes[order.exectype]

            msg = f'下單紀錄: 編號: {order.ref:0>4}, 幣別: {order.data._name}, ' \
                  f'買賣: {ord_type: <4}, 類型: {exec_type}'

            if order.exectype in [order.StopLimit]:
                deadline = bt.num2date(order.valid)
                msg += f', 限價價格: {order.plimit}, 啟動價格: {order.price}, ' \
                       f'期限； {deadline}'

            self.log(msg, doprint=True)

        elif order.status in [order.Completed]:
            if order.exectype in [order.StopTrail, order.StopTrailLimit]:
                print('執行停損單')
                return

            msg = ''
            tr = None
            if order.isbuy():
                val = -order.executed.price * order.executed.size
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.data0.datetime.datetime(0)])
                msg = "買"
            elif order.issell():
                val = -order.executed.price * order.executed.size
                tr = pd.DataFrame([[order.executed.size, order.executed.price, val, float('nan')]],
                                  columns=self.transaction.columns,
                                  index=[self.data0.datetime.datetime(0)])
                msg = "賣"

            msg += f"單執行: 編號: {order.ref:0>4}, 幣別: {order.data._name}, " \
                   f"執行價格: {order.executed.price}, 部位大小: {order.executed.size}"
            if order.exectype == 4 and isinstance(order.price, float):
                atr = abs(order.price - order.plimit) / (self.limit_param - self.trigger_param)
                msg = msg + ", ATR: {}".format(atr)
                tr.loc[self.data0.datetime.datetime(0), 'atr'] = atr

            self.log(msg, doprint=True)
            self.transaction = self.transaction.append(tr)

        elif order.status in [order.Canceled]:
            self.log(f"Order Canceled: 編號: {order.ref:0>4}, 限價價格: {order.plimit}", doprint=True)

    def prenext(self):
        self.next()

    def nextstart(self):
        self.is_started = True
        self.next()

    def next(self):
        self.log(datetime.datetime.now(), doprint=True)
        for d in self.datas:
            if len(d) == 0:
                continue
            msg = f'{d._name}, {len(d)}, {bt.num2date(d.datetime[0])},' \
                  f' {d.open[0]}, {d.high[0]}, {d.low[0]}, {d.close[0]}'
            self.log(msg, doprint=True)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'交易紀錄: 毛利：{trade.pnl:.2f}, 淨利: {trade.pnlcomm:.2f}, '
                 f'市值: {self.broker.getvalue():.2f}',
                 doprint=True)
        self.log('===========================================================================', doprint=True)

    def notify_timer(self, timer, when, *args, **kwargs):
        if not self.is_started:
            return

        # if when.second:
        #     return

        if when.second % 15 == 10:
            # For the last 5 min, create close order for outstanding position
            position_size = self.getposition().size
            valid2 = when + datetime.timedelta(seconds=5)

            if position_size > 0:
                self.order_buy_close = self.sell(exectype=bt.Order.StopLimit, plimit=self.order_buy_close.plimit,
                                                 valid=valid2, size=abs(position_size))
            elif position_size < 0:
                self.order_sell_close = self.buy(exectype=bt.Order.StopLimit, plimit=self.order_sell_close.plimit,
                                                 valid=valid2, size=abs(position_size))
            return
        elif when.second % 15:
            return

        valid1 = when + datetime.timedelta(seconds=10)

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

                self.log("Close position on next open market price", doprint=True)
                self.order_close = self.close()
