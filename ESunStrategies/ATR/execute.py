import math
import os
import sys
import datetime
import pandas as pd
import quantstats
import configparser
import backtrader as bt
import numpy as np
from scipy import stats

# from Intra15mins_reversion_strategy import Intra15MinutesReverseStrategy
from multipleTimeframes import Intra15MinutesReverseStrategy
from stochatic_oscillator_strategy import StochasticOscillatorStrategy

from backtrader.utils.db_conn import MyPostgres
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo


class MySizer(bt.Sizer):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')

    def _getsizing(self, comminfo, cash, data, isbuy):
        unit = self.config['size']['unit_basic']

        # Find latest data index
        j = -1
        while math.isnan(data.close[j]):
            j -= 1

        cash_max_limit_portion = self.config['size']['size_portion']
        size = math.floor(cash * cash_max_limit_portion / data.close[j] / unit) * unit
        return size


def get_data_df(table_name, target, freq_data, start, end, source):
    myDB = MyPostgres()
    freq_data = freq_data.replace('_', '').replace('m', 'min').replace('h', 'H')
    get_data = '''SELECT date + time, open, high, low, close FROM ''' + '\"' + table_name + "\" "  \
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


def get_data_live(target, freq_data):
    if 'm' in freq_data:
        time_unit = 'Minutes'
        datacomp = int(freq_data[0:freq_data.index('m')])
    elif 'h' in freq_data:
        time_unit = 'Hours'
        datacomp = int(freq_data[0:freq_data.index('h')])
    else:
        time_unit = ''
        datacomp = 0

    timeframe = bt.TimeFrame.TFrame(time_unit)

    data_tf = bt.TimeFrame.Ticks
    data_comp = 5
    datakwargs = dict(
        timeframe=data_tf, compression=data_comp,
    )
    rekwargs = dict(timeframe=timeframe, compression=datacomp)

    data_factory = bt.feeds.HSBCData
    ccy = f"{target[:3]}/{target[4:]}"
    data = data_factory(dataname=ccy, **datakwargs)

    return data, rekwargs


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


def execute_history(target, freq_data, partial_name, start, end, strategy, sizer=MySizer, source='Histdata'):
    cerebro = bt.Cerebro()
    slippage = get_settings4tag(target, 'slippage')

    if len(freq_data) == 1:
        data = get_data_df(target + '_OHLC', target, freq_data[0], start, end, source)
        cerebro.adddata(data, name=target + freq_data[0])
    elif len(freq_data) == 2:
        data1 = get_data_df(target + '_OHLC', target, freq_data[0], start, end, source)
        cerebro.adddata(data1, name=target + freq_data[0])
        data2 = get_data_df(target + '_OHLC', target, freq_data[1], start, end, source)
        cerebro.adddata(data2, name=target + freq_data[1])

    cerebro.addstrategy(strategy)
    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(sizer)
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

    # b = Bokeh(style='bar', scheme=Tradimo(), file_name=target + partial_name + '.html')
    # b.params.filename = './output/' + target + partial_name + '_figure.html'
    # cerebro.plot(b)

    trans = analyze_transaction(strat.transaction)

    return exe_ret, exe_pos, trans


def execute_live(target, freq, strategy, sizer=MySizer):
    cerebro = bt.Cerebro()
    slippage = get_settings4tag(target, 'slippage')

    data, args = get_data_live(target, freq)
    cerebro.resampledata(data, **args)
    cerebro.addstrategy(strategy)

    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(sizer)
    cerebro.broker.set_slippage_fixed(fixed=slippage)

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

    return exe_ret, exe_pos, strat.transaction


def analyze_transaction(trans):
    # classify data
    payoff_list = [[], [], [], [], []]
    if 'atr' in list(trans):
        payoff = pd.DataFrame(
            columns=['value', 'in_price', 'out_price', 'amount', 'atr', 'is_long', 'is_win', 'is_strategy'])
    else:
        payoff = pd.DataFrame(
            columns=['value', 'in_price', 'out_price', 'amount', 'is_long', 'is_win', 'is_strategy'])

    for i in range(len(trans)):
        if not i % 2:
            continue

        p = trans['value'][i] + trans['value'][i - 1]
        payoff_list[0].append(p)  # all

        if 'atr' in list(trans):
            pa = pd.DataFrame(
                [[p, trans['price'][i - 1], trans['price'][i], abs(trans['amount'][i]), trans['atr'][i - 1],
                    trans['amount'][i - 1] > 0, p > 0, trans.index[i].minute % 15 != 1]],
                columns=payoff.columns, index=[trans.index[i]])
        else:
            pa = pd.DataFrame(
                [[p, trans['price'][i - 1], trans['price'][i], abs(trans['amount'][i]),
                  trans['amount'][i - 1] > 0, p > 0, trans.index[i].minute % 15 != 1]],
                columns=payoff.columns, index=[trans.index[i]])

        payoff = payoff.append(pa)

        if p > 0:
            payoff_list[1].append(p)  # positive
        else:
            payoff_list[2].append(p)  # negative

        if trans.index[i].minute % 15 == 1:
            payoff_list[4].append(p)  # clean
        else:
            payoff_list[3].append(p)  # strategy

    # statistics
    col = ['All', 'Positive', 'Negative', 'Strategy', 'Clean']
    summary = pd.DataFrame(columns=col,
                           index=['count', 'sum', 'mean', 'median', 'std', 'std_r', 'pvalue'])
    for i in range(len(payoff_list)):
        cnt = len(payoff_list[i])
        summary[col[i]]['count'] = cnt
        summary[col[i]]['sum'] = sum(payoff_list[i])
        summary[col[i]]['mean'] = np.mean(payoff_list[i])

        # skip for only one sample
        if cnt < 2:
            continue

        summary[col[i]]['median'] = np.median(payoff_list[i])
        summary[col[i]]['std'] = np.std(payoff_list[i])
        summary[col[i]]['std_r'] = stats.iqr(payoff_list[i], scale='normal')
        if col[i] == 'Negative':
            summary[col[i]]['pvalue'] = stats.ttest_1samp(payoff_list[i], 0, alternative='less').pvalue
        else:
            summary[col[i]]['pvalue'] = stats.ttest_1samp(payoff_list[i], 0, alternative='greater').pvalue

    print(summary)
    return payoff


def get_title(targets):
    title = ''
    for t in targets:
        if len(title) == 0:
            title += t
        else:
            title += ' + ' + t
    return title


def single_strategy(target, freq_data, partial_name, start, end, src='Histdata'):
    title = target
    returns, positions, transactions = execute_history(target, freq_data, partial_name, start, end,
                                                       Intra15MinutesReverseStrategy, source=src)
    file_name = f'{target}{partial_name}_atr_performance_report.html'
    quantstats.reports.html(returns, output=os.path.join(os.getcwd(), '../output\\', file_name), title=title)


def single_strategy_live(target, freq):
    title = target
    returns, positions, transactions = execute_live(target, freq, Intra15MinutesReverseStrategy)
    file_name = f'{target}{freq}_live_atr_performance_report.html'
    quantstats.reports.html(returns, output=os.path.join(os.getcwd(), '../output\\', file_name), title=title)


def multiple_strategy(targets, freq_data, partial_name, start, end, src='Histdata'):
    title = get_title(targets)
    config = read_config()
    positions = pd.DataFrame(columns=targets)
    position = pd.DataFrame()
    for t in targets:
        print(t)
        _, pos, trans = execute_history(t, freq_data, partial_name, start, end,
                                        Intra15MinutesReverseStrategy, source=src)
        positions[t] = pos.sum(axis=1)
        if len(position) == 0:
            position['cash'] = round(positions[t] * config['portfolio_weight'][t], 2)
        else:
            position['cash'] += round(positions[t] * config['portfolio_weight'][t], 2)

    returns = position.pct_change()
    returns['cash'][0] = 0
    returns.index = [x.to_datetime64() for x in returns.index]

    if int(config['control']['isWeek']):
        num_week = end.isocalendar()[1]
        file_name = f'{str(len(targets))}CCY{partial_name}_w{num_week}_reversion_performance.html'
    else:
        file_name = f'{str(len(targets))}CCY{partial_name}_reversion_performance.html'

    quantstats.reports.html(returns['cash'], output=os.path.join(__file__, '../output/', file_name), title=title)

    date_diff = pd.DataFrame(positions.index).diff()
    date_diff['Datetime'] /= np.timedelta64(1, 'D')


if __name__ == '__main__':
    config = read_config()
    if int(config['control']['isWeek']):
        today = datetime.datetime.today().date()
        start_date = today - datetime.timedelta(days=today.weekday() + 1 + 7)
        end_date = start_date + datetime.timedelta(days=6)
        source = 'HSBC'
    else:
        start_date = datetime.datetime(2020, 2, 1)
        end_date = datetime.datetime(2020, 5, 1)
        source = 'Histdata'

    freq_list = ['_1m', '_15m']

    sub_name_str = ''
    for i in range(0, len(freq_list)):
        sub_name_str = sub_name_str + freq_list[i]

    # ccy_targets = ['AUDUSD', 'GBPUSD', 'NZDUSD', 'USDZAR']
    # multiple_strategy(ccy_targets, freq_list, sub_name_str, start_date, end_date, source)

    # ccy_targets = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
    #                'USDJPY', 'USDMXN', 'USDSEK', 'USDSGD', 'USDZAR']
    # for ccy_target in ccy_targets:
    #     single_strategy(ccy_target, freq, start_date, end_date)

    # ccy_targets = ['AUDUSD', 'GBPUSD', 'NZDUSD', 'USDZAR']
    # multiple_all_strategy(ccy_targets, freq_list, sub_name_str, start_date, end_date)

    ccy_target = 'GBPUSD'
    single_strategy(ccy_target, freq_list, sub_name_str, start_date, end_date)
    # single_strategy_live(ccy_target, '_15m')

    sys.exit(0)
