import math
import os
import sys
import time
import datetime
import pandas as pd
import quantstats
import backtrader as bt

from backtrader.utils.db_conn import MyPostgres
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo


# from backtrader.binancetest.db_conn import MySqlite


class Intra15MinutesReverseStrategy(bt.Strategy):

    def __init__(self):
        self.atr = bt.indicators.ATR()
        self.order_buy = list()
        self.order_sell = list()
        self.order_buy_close = list()
        self.order_sell_close = list()

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.exectype in [order.StopTrail, order.StopTrailLimit]:
                print('執行停損單')
            if order.isbuy():
                self.log("買單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(order.ref, order.executed.price,
                                                                              order.executed.comm, order.executed.size),
                         doprint=True)
            elif order.issell():
                self.log("賣單執行: 訂單編號: {}, 執行價格: {}, 手續費: {}, 部位大小: {}".format(order.ref, order.executed.price,
                                                                              order.executed.comm, order.executed.size),
                         doprint=True)
            self.log("ATR: {}, Threshold: {}".format(self.atr[0], math.sqrt(self.data.low[0])), doprint=True)
            self.log("ATR-1: {}, Threshold-1: {}".format(self.atr[-1], math.sqrt(self.data.low[-1])), doprint=True)
        elif order.status in [order.Canceled]:
            self.log("Order Canceled: 訂單編號: {}, 限價價格: {}".format(order.ref, order.plimit), doprint=True)

    def next(self):
        # str = 'Open: {}, High: {}, Low: {}, Close: {}, ATR: {:5f}, index: {} '.format(
        #    self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0], self.atr[0], len(self))
        # self.log(str, doprint=True)

        valid1 = datetime.timedelta(minutes=15)

        if not self.position:
            # EHT 的門檻用 math.sqrt(math.sqrt(self.data.low[0]))
            # BTC 的門檻用 math.sqrt(self.data.low[0])
            # AUD 沒有明顯的持續漲勢或跌勢，不用設定門檻
            # if self.atr[0] >= math.sqrt(self.data.low[0]):
            # 用當根K bar的資訊計算，下限價單會在下一根K bar生效
            # 賺取向下偏離過多的reversion
            trigger_price = self.data.low[0] - 2 * self.atr[0]
            buy_limit_price = self.data.low[0] - 5 * self.atr[0]
            self.order_buy = self.buy(exectype=bt.Order.StopLimit, price=trigger_price, plimit=buy_limit_price,
                                      valid=valid1)
            self.order_buy_close = self.sell(exectype=bt.Order.StopLimit, price=buy_limit_price, plimit=trigger_price,
                                             valid=valid1)

            # 賺取向上偏離過多的reversion
            trigger_price = self.data.high[0] + 2 * self.atr[0]
            sell_limit_price = self.data.high[0] + 5 * self.atr[0]
            self.order_sell = self.sell(exectype=bt.Order.StopLimit, price=trigger_price, plimit=sell_limit_price,
                                        valid=valid1)
            self.order_sell_close = self.buy(exectype=bt.Order.StopLimit, price=sell_limit_price, plimit=trigger_price,
                                             valid=valid1)
        else:
            position_size = self.getposition().size
            if position_size != 0:
                self.close()

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('交易紀錄： 毛利：%.4f, 淨利：%.4f, 手續費：%.4f, 市值：%.2f, 現金：%.2f' %
                 (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()),
                 doprint=True)
        self.log('===========================================================================', doprint=True)


def get_data_df(table_name):
    myDB = MyPostgres('172.27.110.247', '5433', 'FX_Market')
    get_data = f"SELECT date + time, open, high, low, close FROM {table_name} " \
               + f"WHERE ccys = 'AUDUSD' AND freq = '5min' ORDER BY date, time"
    rows = myDB.get_data(get_data)
    myDB.disconnect()

    cols = ['date', 'open', 'high', 'low', 'close']
    temp_df = pd.DataFrame(rows, columns=cols)
    temp_df['volume'] = 0

    # timeArray = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(x))) for x in temp_df['date'].tolist()]
    # temp_df['date'] = pd.DataFrame(timeArray)
    # temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_df.set_index('date', inplace=True)
    data = bt.feeds.PandasDirectData(dataname=temp_df, dtformat="%Y-%m-%d %H:%M:%S", fromdate=start_date,
                                     todate=end_date, open=1, high=2, low=3, close=4, volume=5, openinterest=-1)
    return data


if __name__ == '__main__':
    target = 'AUDUSD'
    cerebro = bt.Cerebro()
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2021, 1, 1)
    # freq = '_5m'
    freq = '_15m'
    # freq = '_1h'
    cerebro.adddata(get_data_df('fx_hourly_data'), name=target + freq)
    cerebro.addstrategy(Intra15MinutesReverseStrategy)

    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10000)
    cerebro.broker.set_slippage_fixed(fixed=0.0002)
    # cerebro.broker.setcommission(commission=0.001)  # 0.1%

    # 策略分析模塊
    # cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    # cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率時序數據
    # cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
    # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')  # 夏普比率
    # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤
    # 觀察器模塊
    # cerebro.addobserver(bt.observers.Value)
    # cerebro.addobserver(bt.observers.DrawDown)
    # cerebro.addobserver(bt.observers.Trades)

    results = cerebro.run(stdstats=True)
    strat = results[0]
    # portfolio_stats = strat.analyzers.getbyname('pyfolio')
    # returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    # returns.index = returns.index.tz_convert(None)
    # file_name = target + freq + '_reversion.html'
    # quantstats.reports.html(returns, output=os.path.join(__file__, '../', file_name), title=target)

    b = Bokeh(style='bar', scheme=Tradimo())
    cerebro.plot(b)
    sys.exit(0)
