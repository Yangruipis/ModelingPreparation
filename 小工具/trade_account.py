# -*- coding:utf-8 -*-
from WindPy import *
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

INSTRUMENT_OPTION = 0
INSTRUMENT_FUTURE = 1

DIRECTION_BUY = 0
DIRECTION_SELL = 1

OPENTYPE_OPEN = 0
OPENTYPE_CLOSE = 1 

future_parameter = {
    'interest_rate' : 0.0000231,
    'deposit_rate' : 0.2,
    'point_value' : 300
}

option_parameter = {
    'interest' : 2.5
}

class PositionQueue:

    def __init__(self):
        self.buy_queue_dict = defaultdict(list)
        self.sell_queue_dict = defaultdict(list)
        self.positions = defaultdict(int)

    def add(self, instrument, direction, price):
        self.positions[(instrument, direction)] += 1

        if direction == DIRECTION_BUY:
            self.buy_queue_dict[instrument].append(price)
        else:
            self.sell_queue_dict[instrument].append(price)
    
    def pop(self, instrument, direction):
        self.positions[(instrument, direction)] -= 1
        if direction == DIRECTION_SELL:
            self.buy_queue_dict[instrument].pop(0)
        else:
            self.sell_queue_dict[instrument].pop(0)

    def have_position(self, instrument, direction):
        if direction == DIRECTION_SELL:
            if instrument in self.buy_queue_dict:
                return True
            else:
                return False
        else:
            if instrument in self.sell_queue_dict:
                return True
            else:
                return False

    def display(self):
        pass


class TradeAccount:

    def __init__(self, init_capital):
        self.capital = init_capital
        self.init_capital = init_capital
        self.deposit = 0
        self.cash = init_capital
        self.position = PositionQueue()
    
    def order_future(self, price, instrument, direction, open_type, amount=1):
        if open_type == OPENTYPE_OPEN:
            deposit = amount * price * future_parameter['point_value'] * future_parameter['deposit_rate']
            interest = amount * price * future_parameter['point_value'] * future_parameter['interest_rate']
            self.deposit += deposit
            self.cash -= deposit + interest
            self.capital -= interest
            for i in range(amount):
                self.position.add(instrument, direction, price)

        elif open_type == OPENTYPE_CLOSE:
            if direction == DIRECTION_BUY:
                interest = price * future_parameter['point_value'] * future_parameter['interest_rate']
                for i in range(amount):
                    to_be_offset_list = self.position.sell_queue_dict[instrument]
                    if to_be_offset_list == []:
                        print '期货空头合约仓位不足， 剩余 %s 个请求无法平仓' % (amount-i)
                        break
                    else:
                        deposit = to_be_offset_list[0] * future_parameter['point_value'] * future_parameter['deposit_rate']
                        earn = (to_be_offset_list[0] - price) *  future_parameter['point_value']
                        self.deposit -= deposit
                        self.cash += deposit + earn - interest
                        self.capital += earn - interest
                        self.position.pop(instrument, direction)

            elif direction == DIRECTION_SELL:
                interest = price * future_parameter['point_value'] * future_parameter['interest_rate']
                for i in range(amount):
                    to_be_offset_list = self.position.buy_queue_dict[instrument]
                    if to_be_offset_list == []:
                        print '期货多头合约仓位不足， 剩余 %s 个请求无法平仓' % (amount - i)
                        break
                    else:
                        deposit = to_be_offset_list[0] * future_parameter['point_value'] * future_parameter[
                            'deposit_rate']
                        earn = (price - to_be_offset_list[0]) * future_parameter['point_value']
                        self.deposit -= deposit
                        self.cash += deposit + earn - interest
                        self.capital += earn - interest
                        self.position.pop(instrument, direction)
        else:
            raise Exception('No Such Open Type')

    def order_option(self, price, instrument, direction, open_type, amount):
        # 期权卖开无手续费
        if open_type == OPENTYPE_OPEN:
            if direction == DIRECTION_BUY:
                interest = option_parameter['interest'] * amount
                cost = price * 10000 * amount
                self.cash += - cost - interest
                self.capital += - interest
                for i in range(amount):
                    self.position.buy_queue_dict[instrument].append(price)

            elif direction == DIRECTION_SELL:
                interest = 0
                get = price * 10000 * amount
                self.cash += get - interest
                self.capital += - interest
                for i in range(amount):
                    self.position.sell_queue_dict[instrument].append(price)
        
        elif open_type == OPENTYPE_CLOSE:
            interest = option_parameter['interest']
            if direction == DIRECTION_BUY:
                for i in range(amount):
                    to_be_offset_list = self.position.sell_queue_dict[instrument]
                    if to_be_offset_list == []:
                        print '期权空头合约仓位不足， 剩余 %s 个请求无法平仓' % (amount - i)
                        break
                    else:
                        earn = (to_be_offset_list[0] - price) * 10000
                        self.cash += - price * 10000 - interest
                        self.capital += earn - interest
                        self.position.pop(instrument, direction)

            elif direction == DIRECTION_SELL:
                for i in range(amount):
                    to_be_offset_list = self.position.buy_queue_dict[instrument]
                    if to_be_offset_list == []:
                        print '期权多头合约仓位不足， 剩余 %s 个请求无法平仓' % (amount - i)
                        break
                    else:
                        earn = (price - to_be_offset_list[0]) * 10000
                        self.cash += price * 10000 - interest
                        self.capital += earn - interest
                        self.position.pop(instrument, direction)
    
    def end_trade(self):
        with open('account_record', 'wb') as f:
            pickle.dump(self, f)


class ValueCalculate():

    def __init__(self, capital_list, init_capital):
        self._init_capital = init_capital
        self.capital_list = capital_list
        self.return_list = []
        self.profit_list = []
        self.get_return_list()

    def get_return_list(self):
        for i, capital in enumerate(self.capital_list):
            if i == 0:
                self.return_list.append((capital - self._init_capital) / self._init_capital)
                self.profit_list.append(capital - self._init_capital)
            else:
                self.return_list.append((capital - self.capital_list[i-1]) / self.capital_list[i-1])
                self.profit_list.append(capital - self.capital_list[i-1])


    def get_total_return(self):
        return (self.capital_list[-1] - self._init_capital) / self._init_capital

    def get_annual_return(self):
        return self.get_total_return() / len(self.capital_list) * 250.0

    def get_average_return(self):
        return np.mean(self.return_list)

    def get_total_trade_times(self):
        return "%s / %s" % ((self.get_win_times() + self.get_lose_times()), len(self.capital_list))

    def get_return_volatility(self):
        rit_bar = self.get_average_return()
        sum_temp = 0
        for i in self.return_list:
            sum_temp += np.square(i - rit_bar)
        volatility = np.sqrt((250.0 / (len(self.capital_list) - 1)) * sum_temp)
        return volatility

    def get_win_times(self):
        win_list = [i for i in self.return_list if i > 0]
        return len(win_list)

    def get_lose_times(self):
        lose_list = [i for i in self.return_list if i < 0]
        return len(lose_list)

    def get_win_ratio(self):
        return self.get_win_times() * 1.0 / (self.get_win_times() + self.get_lose_times())

    def get_win_lose_ratio(self):
        win_sum  = np.sum([i for i in self.profit_list if i > 0])
        lose_sum = np.sum([i for i in self.profit_list if i < 0])
        return - win_sum * 1.0 / lose_sum

    def get_max_win(self):
        return max([i for i in self.profit_list if i > 0]) / self._init_capital

    def get_max_lose(self):
        return -min([i for i in self.profit_list if i < 0]) / self._init_capital

    def get_continue_win_times(self):
        time_count_list = []
        temp = 0
        for i, returns in enumerate(self.return_list):
            if returns > 0:
                temp += 1
            else:
                time_count_list.append(temp)
                temp = 0
        return max(time_count_list)

    def get_continue_lose_times(self):
        time_count_list = []
        temp = 0
        for i, returns in enumerate(self.return_list):
            if returns < 0:
                temp += 1
            else:
                time_count_list.append(temp)
                temp = 0
        return max(time_count_list)

    def get_max_drawdown(self):
        drawdown_list = []
        for i, capital in enumerate(self.capital_list):
            new_capital_list = self.capital_list[:i]
            if len(new_capital_list) > 0:
                max_capital_past = max(new_capital_list)
                drawdown = (1 - capital / max_capital_past)
                drawdown_list.append(drawdown)
        return max(drawdown_list)

    def get_sharp_ratio(self):
        volatility = self.get_return_volatility()
        sharp = ((self.capital_list[-1] - self._init_capital) / self._init_capital - 0.03) / volatility
        return sharp

    def display(self):
        print "总收益率: ", self.get_total_return()
        print "年化收益率: ", self.get_annual_return()
        print "日均收益率: ", self.get_average_return()
        print "总交易次数: ", self.get_total_trade_times()
        print "收益波动率: ", self.get_return_volatility()
        print "获胜次数: ", self.get_win_times()
        print "失败次数: ", self.get_lose_times()
        print "胜率: ", self.get_win_ratio()
        print "盈亏比: ", self.get_win_lose_ratio()
        print "单次最大盈利: ", self.get_max_win()
        print "单次最大亏损: ", self.get_max_lose()
        print "最大连胜次数: ", self.get_continue_win_times()
        print "最大连负次数: ", self.get_continue_lose_times()
        print "最大回撤: ", self.get_max_drawdown()
        print "夏普比率: ", self.get_sharp_ratio()

if __name__ == "__main__":
    pass
