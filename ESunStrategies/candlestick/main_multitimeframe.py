import datetime
import math
import os
import sys

import numpy as np
import pandas as pd
import quantstats

from MA_candlestick_strategy import MACandlestick_LongStrategy, MACandlestick_ShortStrategy
from backtrader.utils.db_conn import MyPostgres

sys.path.insert(0, os.path.abspath(__file__ + "/../../../EsunBacktrader/"))
import backtrader as bt


class MySizer(bt.Sizer):
    def _getsizing(self, comminfo, cash, data, isbuy):
        unit = unit_basic
        size = math.floor(cash * size_portion / data.close[-1] / unit) * unit
        return size


def get_size(target):
    if target in size_dict:
        return size_dict[target]
    else:
        return size_basic


def get_slippage(target):
    if target in slippage_dict:
        return slippage_dict[target]
    else:
        return slippage_basic


def get_data_df(table_name, target, freq_data, start, end, source):
    myDB = MyPostgres()
    freq_data = freq_data.replace('_', '').replace('m', 'min').replace('h', 'H')
    get_data = '''SELECT date + time, open, high, low, close FROM ''' + '\"' + table_name + "\" " \
               + f"WHERE freq = '{freq_data}' AND broker='{source}' ORDER BY date, time"

    rows = myDB.get_data(get_data)
    myDB.disconnect()

    cols = ['date', 'open', 'high', 'low', 'close']
    temp_df = pd.DataFrame(rows, columns=cols)
    temp_df['volume'] = 0
    temp_df.set_index('date', inplace=True)

    if target.startswith('USD'):
        temp_df[['open', 'high', 'low', 'close']] = 1 / temp_df[['open', 'low', 'high', 'close']]

    data = bt.feeds.PandasDirectData(dataname=temp_df, dtformat="%Y-%m-%d %H:%M:%S", fromdate=start,
                                     todate=end, open=1, high=2, low=3, close=4, volume=5, openinterest=-1)
    return data


def execute(target, freq1, freq2, start, end):
    log_filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M') + f'{target}_log.txt'
    log_file = os.path.join(__file__, '../log/', log_filename)
    # with open(log_file, 'w') as f:
    #     with redirect_stdout(f):

    cerebro = bt.Cerebro()
    slippage = get_slippage(target)

    data1 = get_data_df('fx_hourly_data', target, freq1, start, end)
    cerebro.adddata(data1, name=target + freq1)
    data2 = get_data_df('fx_hourly_data', target, freq2, start, end)
    cerebro.adddata(data2, name=target + freq2)

    if target.startswith('USD'):
        cerebro.addstrategy(MACandlestick_LongStrategy)
    else:
        cerebro.addstrategy(MACandlestick_ShortStrategy)

    cerebro.broker.setcash(1000000)
    cerebro.addsizer(MySizer)
    cerebro.broker.set_slippage_perc(perc=slippage)
    cerebro.broker.set_checksubmit(False)

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

    # b = Bokeh(style='bar', scheme=Tradimo(), file_name=target + freq1 + freq2 + '.html')
    # b.params.filename = './output/' + target + freq1 + freq2 + '.html'
    # cerebro.plot(b)

    payoff = list()
    payoff_positive = list()
    payoff_negative = list()
    for i in range(len(transactions)):
        if not i % 2:
            continue

        p = transactions['value'][i] + transactions['value'][i - 1]
        payoff.append(p)

        if p > 0:
            payoff_positive.append(p)
        else:
            payoff_negative.append(p)

    return exe_ret, exe_pos


def single_strategy(target, freq1, freq2, start, end):
    title = target
    returns, positions = execute(target, freq1, freq2, start, end)

    file_name = target + freq1 + freq2 + '_reversion.html'
    quantstats.reports.html(returns, output=os.path.join(__file__, '../output/', file_name), title=title)


def multiple_strategy(targets, freq1, freq2, start, end):
    title = ''
    for t in targets:
        if len(title) == 0:
            title += t
        else:
            title += ' + ' + t

    positions = pd.DataFrame(columns=targets)
    position = pd.DataFrame()
    for t in targets:
        print(t)
        ret, pos = execute(t, freq1, freq2, start, end)
        file_name = t + freq1 + freq2 + '_reversion.html'
        quantstats.reports.html(ret, output=os.path.join(__file__, '../../output/', file_name), title=title)

        positions[t] = pos.sum(axis=1)
        if len(position) == 0:
            position['cash'] = round(positions[t] * portfolio_weight[t], 2)
        else:
            position['cash'] += round(positions[t] * portfolio_weight[t], 2)

    returns = position.pct_change()
    returns['cash'][0] = 0
    returns.index = [x.to_datetime64() for x in returns.index]

    file_name = targets[0] + '_' + targets[1] + freq1 + freq2 + '_reversion.html'
    quantstats.reports.html(returns['cash'], output=os.path.join(__file__, '../../output/', file_name), title=title)

    date_diff = pd.DataFrame(positions.index).diff()
    date_diff['Datetime'] /= np.timedelta64(1, 'D')


# parameters ------------------------------------------
# slippage
slippage_dict = {'USDMXN': 0.001, 'USDZAR': 0.001, 'USDSEK': 0.001, 'USDSGD': 0.0005}
slippage_basic = 0.0002

# size
size_dict = {'EURUSD': 8000, 'GBPUSD': 6000, 'USDCHF': 8000, 'USDJPY': 800000,
             'USDMXN': 150000, 'USDSEK': 60000, 'USDZAR': 100000}
size_basic = 10000

unit_basic = 1000  # minimum size of transaction
size_portion = 0.9  # cash portion to trade

portfolio_weight = {'USDJPY': 0.5, 'USDCHF': 0.5}
# -----------------------------------------------------

if __name__ == '__main__':
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2021, 12, 31)

    freq_data1 = '_1m'
    freq_data2 = '_1h'

    ccy_targets = ['USDJPY', 'USDCHF']
    multiple_strategy(ccy_targets, freq_data1, freq_data2, start_date, end_date)

    #ccy_targets = ['USDZAR', 'USDMXN']
    #ccy_targets = ['USDCAD', 'USDSEK', 'USDSGD']
    #ccy_targets = ['AUDUSD', 'NZDUSD']
    #ccy_targets = ['USDJPY', 'USDCHF']

    # for ccy_target in ccy_targets:
    #     print(ccy_target)
    #     single_strategy(ccy_target, freq_data1, freq_data2, start_date, end_date)

    # ccy_target = 'USDZAR'
    # single_strategy(ccy_target, freq_data1, freq_data2, start_date, end_date)

    sys.exit(0)
