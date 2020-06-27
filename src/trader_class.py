import numpy as np
import pandas as pd
from src.helper import data


class Trader:
    def __init__(self, asset_data):
        self.shares = 10000
        self.asset_data = asset_data
        self.all_indices = list(self.asset_data.index)
        self.asset_data['PCT Change'] = self.asset_data['(Close, Bid)*'].pct_change()
        self.buy_cols = {'LONG_ONLY_': '(Open, Ask)', 'LONG_SHORT_': '(Open, Bid)*'}
        self.sell_cols = {'LONG_ONLY_': '(High, Bid)*', 'LONG_SHORT_': '(Low, Ask)'}
        self.buys = {}
        self.buy_logs = []
        self.buy_date_logs = []
        self.buy_dates = {}
        self.sells = {}
        self.sell_dates = {}
        self.sell_logs = []
        self.sell_date_logs = []
        self.win_loss = {}
        self.risk_reward = {}
        self.dd = {}
        self.trade_count = {}
        self.earnings = {}

    def reset_logs(self):
        self.buy_logs = []
        self.sell_logs = []
        self.buy_date_logs = []
        self.sell_date_logs = []

    def assess_conditions(self, condition_type, trade, ind):
        index_value = self.all_indices[ind]
        # cross-over strategy: buy signal
        if condition_type == 'cross-over' and trade == 'buy':
            if self.asset_data.loc[index_value, 'MVA(.Close,21)'] < self.asset_data.loc[index_value, 'MVA(.Close,50)']:
                return True
            else:
                return False
        # cross-over strategy: buy signal
        elif condition_type == 'cross-over' and trade == 'sell':
            if self.asset_data.loc[index_value, 'MVA(.Close,21)'] > self.asset_data.loc[index_value, 'MVA(.Close,50)']:
                return True
            else:
                return False
        # hour based movement strategy
        elif 'hour-move' in condition_type:
            hour = int(condition_type.split('-')[0])
            if self.asset_data.loc[index_value, 'Hour of Day'] == hour:
                return True
            else:
                return False
        # volume based strategy
        elif condition_type == 'volume-move':
            if self.asset_data.loc[index_value, 'Real Volume Change'] > 1:
                return True
            else:
                return False
        # difference in pips strategy
        elif 'single-diff-check' in condition_type:
            magnitude = int(condition_type.split('-')[0])
            if self.asset_data.loc[index_value, 'Abs Dif'] > magnitude:
                return True
            else:
                return False
        # difference in current and previous pips strategy
        elif 'double-diff-check' in condition_type:
            magnitude = int(condition_type.split('-')[0])
            if self.asset_data.loc[index_value, 'Abs Dif'] > magnitude and \
                    self.asset_data.loc[self.all_indices[ind-1], 'Abs Dif'] > magnitude:
                return True
            else:
                return False

    def check_sell_position(self, open_trade_type, close_trade_type, ind, suffix):
        if len(self.__getattribute__(open_trade_type+'_logs')) == 0:
            return False
        open_trade_price = self.__getattribute__(open_trade_type+'_logs')[-1]
        close_trade_price = self.asset_data.loc[self.all_indices[ind],
                                                self.__getattribute__(close_trade_type+'_cols')[suffix]].squeeze()
        earnings = round(close_trade_price*self.shares/open_trade_price, 1)
        if earnings >= 10:
            return True
        elif earnings <= -10:
            return True
        return False

    def perform_trades(self, condition_type, long_only=True):
        suffix = 'LONG_ONLY_' if long_only else 'LONG_SHORT_'
        name = suffix + str(condition_type)
        isOpen = False
        isClose = True
        open_trade_type = 'buy' if long_only else 'sell'
        close_trade_type = 'sell' if long_only else 'buy'
        print('Performing trades .. ')
        for ind, val in enumerate(self.all_indices):
            if ind < 2:
                continue
            open_at = [not isOpen, isClose, self.assess_conditions(condition_type, open_trade_type, ind)]
            close_at = [isOpen, not isClose, self.assess_conditions(condition_type, close_trade_type, ind)] # self.check_sell_position(open_trade_type, close_trade_type, ind, suffix)
            if all(open_at):
                self.__getattribute__(open_trade_type+'_logs').\
                    append(self.asset_data.loc[val, self.__getattribute__(open_trade_type+'_cols')[suffix]].squeeze())
                self.__getattribute__(open_trade_type+'_date_logs').append(val)
                isOpen = True
                isClose = False

            elif all(close_at):
                self.__getattribute__(close_trade_type+'_logs').\
                    append(self.asset_data.loc[val, self.__getattribute__(open_trade_type+'_cols')[suffix]].squeeze())
                self.__getattribute__(close_trade_type+'_date_logs').append(val)
                isOpen = False
                isClose = True

        if len(self.__getattribute__(open_trade_type+'_logs')) != len(self.__getattribute__(close_trade_type+'_logs')):
            self.__getattribute__(open_trade_type+'_logs').pop(-1)
            self.__getattribute__(open_trade_type + '_date_logs').pop(-1)
        self.buys[name] = np.array(self.buy_logs)
        self.buy_dates[name] = np.array(self.buy_date_logs)
        self.sells[name] = np.array(self.sell_logs)
        self.sell_dates[name] = np.array(self.sell_date_logs)
        self.reset_logs()

    def _win_loss(self, key):
        buy_data = self.buys[key] * self.shares
        sell_data = self.sells[key] * self.shares
        diff = sell_data - buy_data
        try:
            win_loss = diff[diff > 0].shape[0] / diff[diff < 0].shape[0]
        except ZeroDivisionError:
            win_loss = diff[diff > 0].shape[0] / diff.shape[0]
        return win_loss

    def _risk_reward(self, key):
        buy_data = self.buys[key] * self.shares
        sell_data = self.sells[key] * self.shares
        diff = sell_data - buy_data
        return diff.mean() / diff.std()

    def _earnings(self, key):
        buy_data = self.buys[key] * self.shares
        sell_data = self.sells[key] * self.shares
        return sum(sell_data - buy_data)

    def _dd(self, key):
        if 'SHORT' in key:
            trade_type = 'LONG_SHORT_'
            open_suffix = 'sell'
            close_suffix = 'buy'
            check_method = 'min'
        else:
            trade_type = 'LONG_ONLY_'
            open_suffix = 'buy'
            close_suffix = 'sell'
            check_method = 'max'
        draw_downs = []
        for ind, open_date in enumerate(self.__getattribute__(open_suffix + '_dates')[key]):
            close_date = self.__getattribute__(close_suffix + '_dates')[key][ind]
            dates = self.all_indices[self.all_indices.index(open_date):self.all_indices.index(close_date)]
            temp = (self.__getattribute__(open_suffix + 's')[key][0]
                    - self.asset_data.loc[dates,
                                          self.__getattribute__(close_suffix + '_cols')[trade_type]]
                    .__getattribute__(check_method)())
            draw_downs.append(temp)
        return sum(draw_downs)

    def summarize_statistics(self, key):
        win_loss = self._win_loss(key)
        risk_reward = self._risk_reward(key)
        dd = self._dd(key)
        earnings = self._earnings(key)
        print(key, "= WL:", round(win_loss, 2), ', TRADES:', len(self.buys[key]),
              ', RR:', round(risk_reward, 2), ', EARNINGS: $', round(earnings, 2))

        self.win_loss[key] = round(win_loss, 2)
        self.risk_reward[key] = round(risk_reward, 2)
        self.trade_count[key] = len(self.buys[key])
        self.dd[key] = dd
        self.earnings[key] = earnings

    def assemble_and_export(self):
        writer = pd.ExcelWriter('..\\results\\TradingData.xlsx', datetime_format='YYYY-MM-DD HH:MM:SS')
        strategies = self.buys.keys()
        for key in strategies:
            if 'SHORT' in key:
                open_suffix = 'sell'
                close_suffix = 'buy'
            else:
                open_suffix = 'buy'
                close_suffix = 'sell'
            opens = self.__getattribute__(open_suffix + 's')[key]
            open_dates = self.__getattribute__(open_suffix + '_dates')[key]
            closes = self.__getattribute__(close_suffix + 's')[key]
            close_dates = self.__getattribute__(close_suffix + '_dates')[key]
            trade_diff = closes - opens
            df = pd.DataFrame([opens, open_dates, closes, close_dates, trade_diff]).T
            df.columns = ['Open Trade', 'Open Trade Date', 'Close Trade', 'Close Trade Date', 'ClosePrice - OpenPrice']
            df.to_excel(writer, sheet_name=key)
        writer.save()
        return

    def export_statistics(self):
        writer = pd.ExcelWriter('..\\results\\Results.xlsx')
        win_loss = pd.DataFrame.from_dict(self.win_loss, orient='index').squeeze()
        risk_reward = pd.DataFrame.from_dict(self.risk_reward, orient='index').squeeze()
        dd = pd.DataFrame.from_dict(self.dd, orient='index').squeeze()
        trade_count = pd.DataFrame.from_dict(self.trade_count, orient='index').squeeze()
        earnings = pd.DataFrame.from_dict(self.earnings, orient='index').squeeze()
        results = pd.concat([win_loss, risk_reward, dd, earnings, trade_count], axis=1)
        results.columns = ['WinLoss', 'RiskReward', 'Draw Down', 'Earnings ($)', 'Total Trades']
        results.to_excel(writer, sheet_name='results')
        writer.save()
        return


def run_all_cases():
    hour_moves = [str(ind).zfill(2) + '-hour-move' for ind in [2, 3, 4, 18, 19, 20, 15, 16, 17]]
    strategies = ['cross-over', 'volume-move', '5-single-diff-check',
                  '5-double-diff-check', '10-single-diff-check', '10-double-diff-check']
    strategies.extend(hour_moves)
    # strategies = strategies[-2:]
    test = Trader(asset_data=data)
    for strategy in strategies:
        print(f'Working on {strategy}')
        long_key = 'LONG_ONLY_' + strategy
        short_key = 'LONG_SHORT_' + strategy
        test.perform_trades(condition_type=strategy, long_only=True)
        test.summarize_statistics(long_key)
        test.perform_trades(condition_type=strategy, long_only=False)
        test.summarize_statistics(short_key)
        print(' ------------------------------------ ')
    test.assemble_and_export()
    test.export_statistics()
    return test


if __name__ == '__main__':
    instance = run_all_cases()
