import math
import os
import sys
import datetime
import pandas as pd
import quantstats
import configparser
import backtrader as bt
import numpy as np

from backtrader.utils.db_conn import MyPostgres
from scipy import optimize


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


def get_data_df(table_name, target, freq_data, start, end):
    myDB = MyPostgres('172.27.110.247', '5433', 'FX_Market')
    freq_data = freq_data.replace('_', '').replace('m', 'min')
    get_data = f"SELECT date + time, open, high, low, close FROM {table_name} " \
               + f"WHERE ccys = '{target}' AND freq = '{freq_data}' AND data_source='Histdata' " \
               + f"ORDER BY date, time"
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


def read_config():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    return config


def execute(target, freq_data, start, end):
    cerebro = bt.Cerebro()
    config = read_config()

    if target in config['slippage_dict']:
        slippage = config['slippage_dict'][target]
    else:
        slippage = 0.0002

    data = get_data_df('fx_hourly_data', target, freq_data, start, end)

    cerebro.adddata(data, name=target + freq_data)
    cerebro.addstrategy(Intra15MinutesReverseStrategy)

    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10000)
    cerebro.broker.set_slippage_fixed(fixed=slippage)
    # cerebro.broker.setcommission(commission=0.001)  # 0.1%

    # 策略分析模塊
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率時序數據
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')  # 夏普比率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤
    # 觀察器模塊
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Trades)

    results = cerebro.run(stdstats=True)
    strat = results[0]
    portfolio_stats = strat.analyzers.getbyname('pyfolio')
    exe_ret, exe_pos, transactions, gross_lev = portfolio_stats.get_pf_items()
    exe_ret.index = exe_ret.index.tz_convert(None)

    # b = Bokeh(style='bar', scheme=Tradimo(), file_name=target + freq + '.html')
    # b.params.filename = './output/' + target + freq + '.html'
    # cerebro.plot(b)

    return exe_ret, exe_pos


def maximize_sharpe_ratio(mean, cov):
    def target(x):
        denomr = np.sqrt(np.matmul(np.matmul(x, cov), x.T))
        numer = np.matmul(np.array(mean), x.T)
        func = -(numer / denomr)
        return func

    def constraint(x):
        a = np.ones(x.shape)
        b = 1
        const = np.matmul(a, x.T) - b
        return const

    # define bounds and other parameters
    xinit = np.repeat(1. / len(mean), len(mean))
    cons = ({'type': 'eq', 'fun': constraint})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    # invoke minimize solver
    opt = optimize.minimize(target, x0=xinit, method='SLSQP',
                            bounds=bnds, constraints=cons, tol=10 ** -3)

    return opt


def multiple_strategy(targets, freq_data, start, end):
    title = ''
    for t in targets:
        if len(title) == 0:
            title += t
        else:
            title += ' + ' + t

    config = read_config()

    positions = pd.DataFrame(columns=targets)
    position = pd.DataFrame()
    for t in targets:
        _, pos = execute(t, freq_data, start, end)
        positions[t] = pos.sum(axis=1)
        if len(position) == 0:
            position['cash'] = round(positions[t] * config['portfolio_weight'][t], 2)
        else:
            position['cash'] += round(positions[t] * config['portfolio_weight'][t], 2)

    returns = position.pct_change()
    returns['cash'][0] = 0
    returns.index = [x.to_datetime64() for x in returns.index]

    file_name = '3CCY' + freq_data + '_reversion.html'
    quantstats.reports.html(returns['cash'], output=os.path.join(__file__, '../output/', file_name), title=title)

    date_diff = pd.DataFrame(positions.index).diff()
    date_diff['Datetime'] /= np.timedelta64(1, 'D')

    pos_mean = positions.mean()
    pos_cov = positions.cov()
    pos_weight = maximize_sharpe_ratio(pos_mean, pos_cov)


if __name__ == '__main__':
    targets = ['USDZAR', 'GBPUSD', 'USDMXN']
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2021, 8, 1)

    freq = '_15m'

    multiple_strategy(targets, freq, start_date, end_date)
    sys.exit(0)
