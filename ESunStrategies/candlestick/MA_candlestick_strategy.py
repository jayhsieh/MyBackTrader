import datetime
import math
import os
import sys
import backtrader as bt


class MACandlestick_LongStrategy(bt.Strategy):

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data1, period=6)
        self.sma_slow = bt.indicators.SMA(self.data1, period=12)
        self.atr = bt.indicators.ATR(self.data1)
        self.count_golden_cross = 0
        self.order_buy = list()
        self.order_buy_close_sl = list()
        self.order_buy_close_tp = list()
        self.order_buy_close_sl_trail = list()
        self.order_close = list()
        self.long_tp_target = None
        self.long_sl_target = None
        self.factor = 100000 * self.get_factor(self.data1._name.split('_')[0])

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        # 訂單已被處理
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 訂單完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("買單執行: 訂單名稱: {}, 訂單編號: {}, 執行價格: {}, 部位大小: {}".format(order.info['name'],
                                                                               order.ref, order.executed.price,
                                                                               order.executed.size), doprint=True)

            elif order.issell():
                self.log("賣單執行: 訂單名稱: {}, 訂單編號: {}, 執行價格: {}, 部位大小: {}".format(order.info['name'],
                                                                               order.ref, order.executed.price,
                                                                               order.executed.size), doprint=True)

        elif order.status in [order.Canceled]:
            if order.exectype in [order.Limit, order.Stop]:
                self.log("Order Canceled: 訂單名稱: {}, 訂單編號: {}, 限價價格: {}".format(order.info['name'], order.ref,
                                                                               order.price), doprint=True)
            elif order.exectype in [order.StopTrail]:
                self.log(
                    "Order Canceled: 訂單名稱: {}, 訂單編號: {}, 滾動停損百分比: {}".format(order.info['name'], order.ref,
                                                                             order.trailpercent), doprint=True)
        elif order.status in [order.Expired]:
            self.log("Order Expired: 訂單名稱: {}, 訂單編號: {}".format(order.info['name'], order.ref), doprint=True)

    def next(self):

        valid1 = datetime.timedelta(minutes=60)
        tp_range = 0.5
        stop_trail = 0.001

        if self.data0.datetime.datetime(0).minute % 60:
            if not self.position:
                return
            else:
                if self.data0.datetime.datetime(2).day != self.data0.datetime.datetime(1).day:
                    self.order_close = self.close()
                    self.log(f'Close position by the end of day', doprint=True)
                    self.order_buy_close_tp.cancel()
                    self.order_buy_close_sl.cancel()
                    self.order_buy_close_sl_trail.cancel()

        elif self.data0.datetime.datetime(0).minute % 60 == 0:
            if self.sma_fast[-1] > self.sma_slow[-1]:
                self.count_golden_cross += 1
            else:
                self.count_golden_cross = 0

            if not self.position:
                if (round(self.sma_fast[-1], 7) > round(self.sma_slow[-1], 7)) and (0 < self.count_golden_cross <= 5):
                    if (self.data1.low[-1] < self.sma_slow[-1]) and (
                            self.data1.close[-1] > self.data1.open[-1] > self.sma_fast[-1]):
                        if ((self.data1.open[-1] - self.data1.low[-1]) > 1 / 2 * (
                                self.data1.high[-1] - self.data1.low[-1])) and (
                                (self.data1.close[-1] - self.data1.open[-1]) > 1 / 2 * (
                                self.data1.high[-1] - self.data1.open[-1])):
                            self.long_tp_target = math.ceil(
                                (self.data1.high[-1] + tp_range * self.atr[-1]) * self.factor) / self.factor
                            self.long_sl_target = math.floor(self.data1.low[-1] * self.factor) / self.factor

                            self.order_buy = self.buy()

                            self.order_buy_close_tp = self.sell(exectype=bt.Order.Limit,
                                                                price=self.long_tp_target,
                                                                valid=valid1
                                                                )

                            self.order_buy_close_sl = self.sell(exectype=bt.Order.Stop,
                                                                price=self.long_sl_target,
                                                                valid=valid1,
                                                                oco=self.order_buy_close_tp
                                                                )

                            self.order_buy_close_tp.addinfo(name="Inti TP order")
                            self.order_buy_close_sl.addinfo(name="Inti SL order")

                            self.log(f'TP @ {self.order_buy_close_tp.price}', doprint=True)
                            self.log(f'SL @ {self.order_buy_close_sl.price}', doprint=True)

            else:
                position_size = self.getposition().size
                if position_size > 0:
                    if self.sma_fast[-1] < self.sma_slow[-1] and self.sma_fast[-2] > self.sma_slow[-2]:
                        self.order_close = self.close()
                        self.log(f'Close position by cross MA signal', doprint=True)
                        self.order_buy_close_tp.cancel()
                        self.order_buy_close_sl.cancel()
                        self.order_buy_close_sl_trail.cancel()
                    else:
                        self.order_buy_close_tp = self.sell(exectype=bt.Order.Limit,
                                                            price=self.long_tp_target,
                                                            valid=valid1,
                                                            size=abs(position_size))

                        self.order_buy_close_sl_trail = self.sell(exectype=bt.Order.StopTrail,
                                                                  trailpercent=stop_trail,
                                                                  oco=self.order_buy_close_tp,
                                                                  valid=valid1,
                                                                  size=abs(position_size))

                        self.order_buy_close_sl = self.sell(exectype=bt.Order.Stop,
                                                            price=self.long_sl_target,
                                                            oco=self.order_buy_close_tp,
                                                            valid=valid1,
                                                            size=abs(position_size))

                        self.order_buy_close_tp.addinfo(name="TP order")
                        self.order_buy_close_sl.addinfo(name="SL order")
                        self.order_buy_close_sl_trail.addinfo(name="SL Trail order")
                else:
                    raise ValueError('Position should be positive.')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('交易紀錄： 毛利：%.4f, 淨利：%.4f, 手續費：%.4f, 市值：%.2f, 現金：%.2f' %
                 (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()),
                 doprint=True)
        self.log('===========================================================================', doprint=True)


class MACandlestick_ShortStrategy(bt.Strategy):

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data1, period=6)
        self.sma_slow = bt.indicators.SMA(self.data1, period=12)
        self.atr = bt.indicators.ATR(self.data1)
        self.count_death_cross = 0
        self.order_sell = list()
        self.order_sell_close_sl = list()
        self.order_sell_close_tp = list()
        self.order_sell_close_sl_trail = list()
        self.order_close = list()
        self.short_tp_target = None
        self.short_sl_target = None
        self.factor = 100000 * self.get_factor(self.data1._name.split('_')[0])

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        # 訂單已被處理
        if order.status in [order.Submitted, order.Accepted]:
            return

        # self.log("訂單名稱: {}, 訂單編號: {}, 訂單狀態: {}".format(order.info['name'], order.ref, order.status), doprint=True)

        # 訂單完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("買單執行: 訂單名稱: {}, 訂單編號: {}, 執行價格: {}, 部位大小: {}".format(order.info['name'],
                                                                               order.ref, order.executed.price,
                                                                               order.executed.size), doprint=True)

            elif order.issell():
                self.log("賣單執行: 訂單名稱: {}, 訂單編號: {}, 執行價格: {}, 部位大小: {}".format(order.info['name'],
                                                                               order.ref,
                                                                               order.executed.price,
                                                                               order.executed.size), doprint=True)
        elif order.status in [order.Canceled]:
            if order.exectype in [order.Limit, order.Stop]:
                self.log("Order Canceled: 訂單名稱: {}, 訂單編號: {}, 限價價格: {}".format(order.info['name'], order.ref,
                                                                               order.price), doprint=True)
            elif order.exectype in [order.StopTrail]:
                self.log(
                    "Order Canceled: 訂單名稱: {}, 訂單編號: {}, 滾動停損百分比: {}".format(order.info['name'], order.ref,
                                                                             order.trailpercent), doprint=True)
        elif order.status in [order.Expired]:
            self.log("Order Expired: 訂單名稱: {}, 訂單編號: {}".format(order.info['name'], order.ref), doprint=True)

    def next(self):
        valid1 = datetime.timedelta(minutes=60)
        tp_range = 2
        stop_trail = 0.002

        if self.data0.datetime.datetime(0).minute % 60:
            if not self.position:
                return
            else:
                if self.data0.datetime.datetime(2).day != self.data0.datetime.datetime(1).day:
                    self.order_close = self.close()
                    self.log(f'Close position by the end of day', doprint=True)
                    self.order_sell_close_tp.cancel()
                    self.order_sell_close_sl.cancel()
                    self.order_sell_close_sl_trail.cancel()

        elif self.data0.datetime.datetime(0).minute % 60 == 0:
            if self.sma_fast[-1] < self.sma_slow[-1]:
                self.count_death_cross += 1
            else:
                self.count_death_cross = 0

            if not self.position:
                if (round(self.sma_fast[-1], 7) < round(self.sma_slow[-1], 7)) and (0 < self.count_death_cross <= 5):
                    if (self.data1.high[-1] > self.sma_slow[-1]) and (
                            self.data1.close[-1] < self.data1.open[-1] < self.sma_fast[-1]):
                        if ((self.data1.high[-1] - self.data1.open[-1]) > 1 / 2 * (
                                self.data1.high[-1] - self.data1.low[-1])) and (
                                (self.data1.open[-1] - self.data1.close[-1]) > 1 / 2 * (
                                self.data1.open[-1] - self.data1.low[-1])):
                            self.short_tp_target = math.ceil(
                                (self.data1.low[-1] - tp_range * self.atr[-1]) * self.factor) / self.factor
                            self.short_sl_target = math.floor(self.data1.high[-1] * self.factor) / self.factor

                            self.order_sell = self.sell()

                            self.order_sell_close_tp = self.buy(exectype=bt.Order.Limit,
                                                                price=self.short_tp_target,
                                                                valid=valid1)

                            self.order_sell_close_sl = self.buy(exectype=bt.Order.Stop,
                                                                price=self.short_sl_target,
                                                                valid=valid1,
                                                                oco=self.order_sell_close_tp)

                            self.order_sell_close_tp.addinfo(name="init TP order")
                            self.order_sell_close_sl.addinfo(name="init SL order")

                            self.log(f'TP @ {self.order_sell_close_tp.price}', doprint=True)
                            self.log(f'SL @ {self.order_sell_close_sl.price}', doprint=True)
            else:
                position_size = self.getposition().size

                if position_size < 0:
                    if self.sma_fast[-1] > self.sma_slow[-1] and self.sma_fast[-2] < self.sma_slow[-2]:
                        self.order_close = self.close()
                        self.log(f'Close position by cross MA signal', doprint=True)
                        self.order_sell_close_tp.cancel()
                        self.order_sell_close_sl.cancel()
                        self.order_sell_close_sl_trail.cancel()
                    else:
                        self.order_sell_close_tp = self.buy(exectype=bt.Order.Limit,
                                                            price=self.short_tp_target,
                                                            valid=valid1,
                                                            size=abs(position_size))

                        self.order_sell_close_sl_trail = self.buy(exectype=bt.Order.StopTrail,
                                                                  trailpercent=stop_trail,
                                                                  oco=self.order_sell_close_tp,
                                                                  valid=valid1,
                                                                  size=abs(position_size))

                        self.order_sell_close_sl = self.buy(exectype=bt.Order.Stop,
                                                            price=self.short_sl_target,
                                                            oco=self.order_sell_close_tp,
                                                            valid=valid1,
                                                            size=abs(position_size))

                        self.order_sell_close_tp.addinfo(name="TP order")
                        self.order_sell_close_sl_trail.addinfo(name="SL Trail order")
                        self.order_sell_close_sl.addinfo(name="SL order")
                else:
                    raise ValueError('Position should be negative.')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('交易紀錄： 毛利：%.4f, 淨利：%.4f, 手續費：%.4f, 市值：%.2f, 現金：%.2f' %
                 (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()),
                 doprint=True)
        self.log('===========================================================================', doprint=True)
