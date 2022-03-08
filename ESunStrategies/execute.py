import math
import os
import sys
import datetime
import pandas as pd
import quantstats
import configparser
import backtrader as bt
import numpy as np

from Intra15mins_reversion_strategy import Intra15MinutesReverseStrategy

from backtrader.utils.db_conn import MyPostgres
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
from scipy import optimize


class MySizer(bt.Sizer):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')

    def _getsizing(self, comminfo, cash, data, isbuy):
        unit = self.config['size']['unit_basic']
        cash_max_limit_portion = self.config['size']['size_portion']
        size = math.floor(cash * cash_max_limit_portion / data.close[-1] / unit) * unit
        return size


def get_data_df(table_name, target, freq_data, start, end):
    myDB = MyPostgres()
    freq_data = freq_data.replace('_', '').replace('m', 'min')
    get_data = '''SELECT date + time, open, high, low, close FROM ''' + '\"' + table_name + "\" "  \
               + f"WHERE freq = '{freq_data}' AND broker='Histdata' " \
               + f"ORDER BY date, time"
    rows = myDB.get_data(get_data)
    myDB.disconnect()

    cols = ['date', 'open', 'high', 'low', 'close']
    temp_df = pd.DataFrame(rows, columns=cols)
    temp_df['volume'] = 0
    temp_df.set_index('date', inplace=True)

    if target.startswith('USD'):
        temp_df[['open', 'high', 'low', 'close']] = 1 / temp_df[['open', 'low', 'high', 'close']]

    data = bt.feeds.PandasDirectData(dataname=temp_df, dtformat="%Y-%m-%d %H:%M:%S", fromdate=start_date,
                                     todate=end_date, open=1, high=2, low=3, close=4, volume=5, openinterest=-1)
    return data


def get_settings4tag(target, tag):
    """
    target is XXXYYY
    tag is slippage or size here
    """
    config = read_config()
    if target in config[tag]:
        return float(config[tag][target])
    else:
        return float(config[tag][tag + '_basic'])


def read_config():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    return config


def execute(target, freq_data, start, end):
    cerebro = bt.Cerebro()
    slippage = get_settings4tag(target, 'slippage')
    data = get_data_df(target + '_OHLC', target, freq_data, start, end)

    cerebro.adddata(data, name=target + freq_data)
    cerebro.addstrategy(Intra15MinutesReverseStrategy)

    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(MySizer)
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

    b = Bokeh(style='bar', scheme=Tradimo(), file_name=target + freq + '.html')
    b.params.filename = './output/' + target + freq + '_figure.html'
    cerebro.plot(b)

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


def single_strategy(target, freq_data, start, end):
    title = target
    returns, positions = execute(target, freq_data, start, end)

    file_name = target + freq_data + '_reversion_performance_report.html'
    quantstats.reports.html(returns, output=os.path.join(os.getcwd(), 'output\\', file_name), title=title)


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
    start_date = datetime.datetime(2020, 4, 1)
    end_date = datetime.datetime(2021, 9, 1)

    freq = '_15m'

    # ccy_targets = ['USDZAR', 'GBPUSD', 'USDMXN']
    # multiple_strategy(ccy_targets, freq, start_date, end_date)

    # ccy_targets = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
    #                'USDJPY', 'USDMXN', 'USDSEK', 'USDSGD', 'USDZAR']
    # for ccy_target in ccy_targets:
    #     single_strategy(ccy_target, freq, start_date, end_date)

    ccy_target = 'GBPUSD'
    single_strategy(ccy_target, freq, start_date, end_date)

    sys.exit(0)
