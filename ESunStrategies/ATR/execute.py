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

from ESunStrategies.ATR.teststrategy import TestStrategy
from multipleTimeframes import Intra15MinutesReverseStrategy
from stochatic_oscillator_strategy import StochasticOscillatorStrategy
from backtrader.utils.timeit import *
from backtrader.utils.db_conn import MyPostgres


class MySizer(bt.Sizer):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')

    def _getsizing(self, comminfo, cash, data, isbuy):
        unit = int(self.config['size']['unit_basic'])

        # Find latest data index
        j = -1
        while math.isnan(data.close[j]):
            j -= 1

        cash_max_limit_portion = float(self.config['size']['size_portion'])
        size = math.floor(cash * cash_max_limit_portion / data.close[j] / unit) * unit
        return size


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


def get_data_live(target):
    data_factory = bt.feeds.HSBCData
    ccy = f"{target[:3]}/{target[3:]}"
    data = data_factory(dataname=ccy)
    return data


def get_arg_live(freq):
    """
    Gets arguments of data feed for live data
    :param freq: Frequency of bar
    """
    if 's' in freq:
        time_unit = 'Seconds'
        datacomp = int(freq[1:freq.index('s')])
    elif 'm' in freq:
        time_unit = 'Minutes'
        datacomp = int(freq[1:freq.index('m')])
    elif 'h' in freq:
        time_unit = 'Hours'
        datacomp = int(freq[1:freq.index('h')])
    else:
        time_unit = ''
        datacomp = 0

    timeframe = bt.TimeFrame.TFrame(time_unit)
    return dict(timeframe=timeframe, compression=datacomp)


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


def execute_history(target, freq_data, start, end, strategy, sizer=MySizer, source='Histdata'):
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

    return exec_cerebro(cerebro, slippage, sizer)


def execute_live(cerebro, targets, freqs, strategy):
    # data
    for t in targets:
        data_name = [d._name for d in cerebro.datas]
        data = get_data_live(t)

        if t not in data_name:
            cerebro.adddata(data, name=t)

        for freq in freqs:
            tfreq = t + freq
            if tfreq not in data_name:
                args = get_arg_live(freq)
                cerebro.resampledata(data, **args, name=tfreq)

    # strategy
    for t in targets:
        cerebro.addstrategy(strategy, name=t)


def execute_multiple_live(targets, freq, strategy, sizer=MySizer, broker=None):
    cerebro = bt.Cerebro()

    # slippage
    slippage = 0
    # slippage = max([get_settings4tag(t, 'slippage') for t in targets])

    # data
    for t in targets:
        data_name = [d._name for d in cerebro.datas]
        data = get_data_live(t)

        if t not in data_name:
            cerebro.adddata(data, name=t)

        if t + freq not in data_name:
            args = get_arg_live(freq)
            cerebro.resampledata(data, **args, name=t + freq)

    # strategy
    for t in targets:
        cerebro.addstrategy(strategy, name=t)

    # broker
    if broker is not None:
        cerebro.broker = broker

    return exec_cerebro(cerebro, slippage, sizer)


def exec_cerebro(cerebro, slippage, sizer):
    capital = 1000000.0
    cerebro.broker.setcash(capital)
    if sizer is None:
        cerebro.addsizer(bt.sizers.SizerFix, stake=capital)
    else:
        cerebro.addsizer(sizer)
    cerebro.broker.set_slippage_perc(perc=slippage, slip_out=True)

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
    return results


def analyze_result(results):
    for strategy in results:
        analyze_single_result(strategy)

    strat_set = set([type(s).__name__ for s in results])
    for strate_name in strat_set:
        analyze_multi_result(strate_name, results)


def analyze_single_result(strategy):
    """
    Analyzes single result of strategy
    :param strategy: Strategy
    :return:
    """
    title = strat_str_dict[type(strategy).__name__]
    portfolio_stats = strategy.analyzers.getbyname('pyfolio')
    returns, _, _, _ = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)

    if len(strategy.transaction) == 0:
        return

    _ = analyze_transaction(strategy.transaction)

    file_name = f'{title}_{strategy.name}_live.html'
    quantstats.reports.html(
        returns,
        output=os.path.join(os.getcwd(), '../output\\', file_name),
        title=title
    )


def analyze_multi_result(strat_name, results):
    """
    Analyzes multiple results given the strategy name
    :param strat_name: Name of strategy
    :param results: Strategy results
    :return:
    """
    cnt = 0
    position = pd.DataFrame()
    ccys = []
    for s in results:
        if type(s).__name__ != strat_name:
            continue
        cnt += 1
        ccys.append(s.name)

        portfolio_stats = s.analyzers.getbyname('pyfolio')
        _, pos, _, _ = portfolio_stats.get_pf_items()

        p = pos.sum(axis=1)
        if len(position) == 0:
            position['cash'] = round(p, 2)
        else:
            position['cash'] += round(p, 2)

    returns = position.pct_change()
    returns['cash'][0] = 0
    returns.index = [x.to_datetime64() for x in returns.index]

    file_name = f'{strat_name}_{cnt}CCY_live.html'
    title = get_title(ccys)

    quantstats.reports.html(
        returns['cash'],
        output=os.path.join(os.getcwd(), '../output\\', file_name),
        title=title
    )


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
    idx = ['count', 'sum', 'mean', 'median', 'std', 'std_r', 'pvalue']
    summary = pd.DataFrame(columns=col, index=idx)
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


def analyze_result(results, title, partial_name):
    for strat in results:
        portfolio_stats = strat.analyzers.getbyname('pyfolio')
        returns, _, _, _ = portfolio_stats.get_pf_items()
        returns.index = returns.index.tz_convert(None)

        if len(strat.transaction) == 0:
            continue

        _ = analyze_transaction(strat.transaction)

        file_name = f'{title}{partial_name}_atr_performance_report.html'
        quantstats.reports.html(returns, output=os.path.join(os.getcwd(), '../output\\', file_name), title=title)


def single_strategy(target, freq_data, start, end, src='Histdata'):
    results = execute_history(
        target, freq_data, start, end,
        Intra15MinutesReverseStrategy,
        source=src)
    analyze_result(results, "", freq_data[1])


def multiple_strategy(targets, freq_data, start, end, src='Histdata'):
    title = get_title(targets)
    config = read_config()
    positions = pd.DataFrame(columns=targets)
    position = pd.DataFrame()
    for t in targets:
        print(t)
        results = execute_history(
            t, freq_data, start, end,
            Intra15MinutesReverseStrategy,
            source=src
        )
        # Gets useful items from results
        strat = results[0]
        trans = strat.transaction
        portfolio_stats = strat.analyzers.getbyname('pyfolio')
        _, pos, _, _ = portfolio_stats.get_pf_items()

        # Analyze and add performance
        _ = analyze_transaction(trans)
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
        file_name = f'{str(len(targets))}CCY{freq_data[0]}{freq_data[1]}_w{num_week}_reversion_performance.html'
    else:
        file_name = f'{str(len(targets))}CCY{freq_data[0]}{freq_data[1]}_reversion_performance.html'

    quantstats.reports.html(returns['cash'], output=os.path.join(os.getcwd(), '../output\\', file_name), title=title)

    date_diff = pd.DataFrame(positions.index).diff()
    date_diff['Datetime'] /= np.timedelta64(1, 'D')


def single_strategy_live(target, freq):
    broker = bt.brokers.FXBroker()
    results = execute_multiple_live(
        [target], freq, Intra15MinutesReverseStrategy,
        sizer=None,
        broker=broker
    )
    analyze_result(results, "", freq)


def multiple_strategy_live(targets, freq):
    title = f'{len(targets)}CCY'
    broker = bt.brokers.FXBroker()
    results = execute_multiple_live(
        targets, freq, Intra15MinutesReverseStrategy,
        sizer=None,
        broker=broker
    )
    analyze_result(results, title, freq)


@measuretime
def wastetime():
    sum([i ** 2 for i in range(1000000)])


def fx_broker_strategy_live(target, freq):
    broker = bt.brokers.FXBroker()
    _, _, _ = execute_multiple_live(
        [target], freq, TestStrategy,
        sizer=None,
        broker=broker)


is_live = False

# strategies ------------------------------------------
strat_list = ["Intra15MinutesReverseStrategy"]

strat_str_dict = {
    "Intra15MinutesReverseStrategy": "ATR"
}

strat_dict = {
    "Intra15MinutesReverseStrategy": Intra15MinutesReverseStrategy
}

target_dict = {
    "Intra15MinutesReverseStrategy": ['AUDUSD', 'GBPUSD', 'NZDUSD', 'USDZAR']
}

freq_dict = {
    "Intra15MinutesReverseStrategy": ['_5s']
}
# -----------------------------------------------------


if __name__ == "__main__":
    if is_live:
        cerebro = bt.Cerebro()

        # slippage
        slippage = 0
        # slippage = max([get_slippage(t) for t in targets])

        broker = bt.brokers.FXBroker()
        cerebro.broker = broker

        # Add strategy
        for strat in strat_list:
            targets = target_dict[strat]
            freq = freq_dict[strat]
            strategy = strat_dict[strat]
            execute_live(cerebro, targets, freq, strategy)

        res = exec_cerebro(cerebro, slippage, None)
        analyze_result(res)
    else:
        print(datetime.datetime.now())
        config = read_config()
        if int(config['control']['isWeek']):
            today = datetime.datetime.today().date()
            start_date = today - datetime.timedelta(days=today.weekday() + 1 + 7)
            end_date = start_date + datetime.timedelta(days=6)
            source = 'HSBC'
        else:
            start_date = datetime.datetime(2020, 1, 1)
            end_date = datetime.datetime(2021, 5, 1)
            source = 'Histdata'

        freq_list = ['_1m', '_15m']

        ccy_target = 'GBPUSD'
        single_strategy(ccy_target, freq_list, start_date, end_date)
        # TODO: live data does not test yet
        # single_strategy_live(ccy_target, '_15m')

    sys.exit(0)

# if __name__ == '__main__':
#     freq_data = '_5s'
#
#     ccy_target = 'USDZAR'
#     fx_broker_strategy_live(ccy_target, freq_data)
