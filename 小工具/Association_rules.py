# -*- coding:utf-8 -*-

"""
通过树状结构，更高效的挖掘频繁项集

git:
    http://github.com/enaeseth/python-fp-growth/
"""

from fp_growth import find_frequent_itemsets
from apriori import *

test_case = [
['a','b'],
['b','c','d'],
['a','b','d','e'],
['a','d','e'],
['a','b','c'],
['a','b','c','d'],
['a'],
['a','b','c'],
['a','b','d'],
['b','c','e'],
]

# ================  Approach 1: how to get a faster frequent items  ====================
for item, support in find_frequent_itemsets(test_case, 2, True):
   print item, support


# ================ Approach 2: gen a min support rate =================
result_list = []
for i in my_apriori(test_case):
    temp = '-'.join([k for k in i[0]]) + ',' + '-'.join([k for k in i[1]]) + ','
    result_list.append((temp, i[2]))

result_list.sort(key = lambda x: x[1], reverse= False)
