# -*- coding:utf-8 -*-

"""
数据清洗，包括
    - 缺失值补全
        - 均值、中位数补全        Done
        - 插值补全              Done
    - 异常值处理： winsor处理
    - 归一化                    Done
    - 标准化                    Done
    - 二值化                    No Need
    - 分类变量编码
        - 有序                  No Need
        - 无序                  Done
    - 正则化检验（针对纯文本）
    - 去重
    - 去无效值                   No Need
    - 关联性验证                 No Need

"""

import sklearn.preprocessing as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fill_na(df, excep_columns=[], how='mean'):
    """
    补全缺失值
    :param how:
        = 'mean'
        = 'median'
        = 'most_frequent'
    """
    select_columns = [i for i in df.columns if i not in excep_columns]
    df_temp = df.loc[:, select_columns]

    imp = sp.Imputer(missing_values='NaN', strategy=how, axis=0)
    imp.fit(df_temp)
    result = imp.transform(df_temp)
    for i in range(result.shape[1]):
        df[select_columns[i]] = result[:, i]

    return df

def interpolate_na(df, excep_columns=[], how='lagrange'):
    """

    :param df:
    :param how:
        lagrange    拉格朗日插值
        spline      样条插值
    :return:
    """
    select_columns = [i for i in df.columns if i not in excep_columns]

    if how == 'lagrange':
        from scipy.interpolate import lagrange
        def ployinterp_column(s, n, k=5):
            set1 = set(range(len(s)))
            set2 = set(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))
            x = list(set1 & set2)
            y = s[x]  # 取数
            x = np.array(x)[pd.notnull(y)]
            y = y[pd.notnull(y)]  # 剔除空值
            lagrange_result =lagrange(x, y)
            return lagrange_result(n)  # 插值并返回插值结果
        for column in select_columns:
            ds = df.loc[:,column].values
            if isinstance(ds[0], int) or isinstance(ds[0], float):
                for j in range(len(ds)):
                    if pd.isnull(ds[j]):
                        ds[j] = ployinterp_column(ds,j)
                df[column] = ds
        return df
    elif how == 'spline':
        from scipy.interpolate import spline
        for column in select_columns:
            ds = df.loc[:,column].values
            if isinstance(ds[0], int) or isinstance(ds[0], float):
                target_index= np.arange(len(ds))
                index = target_index[pd.notnull(ds)]
                ds_notnull = ds[pd.notnull(ds)]
                new_ds = spline(index, ds_notnull, target_index)
                df[column] = new_ds
        return df

def standardize(df, excep_columns=[]):
    """
    标准化，假设服从正态分布
    """
    select_columns = [i for i in df.columns if i not in excep_columns]
    df_temp = df.loc[:, select_columns]
    scaler = sp.StandardScaler().fit(df_temp)
    result = scaler.transform(df_temp)
    for i in range(result.shape[1]):
        df[select_columns[i]] = result[:, i]
    return df

def normalize(df, excep_columns=[]):
    """
    极值归一化，根据最大最小值使其在[0,1]之间
    """
    select_columns = [i for i in df.columns if i not in excep_columns]
    df_temp = df.loc[:, select_columns]
    min_max_scaler = sp.MinMaxScaler()
    min_max_scaler.fit_transform(df_temp)
    result = min_max_scaler.transform(df_temp)
    for i in range(result.shape[1]):
        df[select_columns[i]] = result[:, i]
    return df


def label_encode(df, encode_column=[]):
    """
    将分类标签进行编码，注意：只针对无序标签
    :param df:              数据框
    :param encode_column:   列名列表
    :return:                数据框
    """
    le = sp.LabelEncoder()
    for column in encode_column:
        # 非数值型转化为数值型
        ds = df.loc[:, column].values
        le.fit(ds)
        df[column] = le.transform(ds)  # array([2, 2, 1])
    return df

def drop_duplicate(df, columns=[]):
    return df.drop_duplicates(subset=columns)


def replace_outlier(df):
    # 有问题，未调试，用winsor
    result =  sp.robust_scale(df, with_scaling=False, with_centering=False)
    return pd.DataFrame(result)

def winsorize(df, low_q=1, up_q=99):
    temp_df = df.copy()
    for column in temp_df.columns:
        ds = temp_df[column].values
        if isinstance(ds[0], int) or isinstance(ds[0], float):
            lower_bound = np.percentile(ds, low_q)
            upper_bound = np.percentile(ds, up_q)
            ds = map(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x, ds)
            temp_df[column] = ds
    return temp_df


if __name__ == '__main__':
    df = pd.read_csv("/home/ray/Documents/suibe/2017/建模/Modeling_Preparation/dataset/auto.csv")
    df_columns = df.columns

    # 分类变量编码
    df = label_encode(df, ['make', 'foreign'])

    # 由于该数据非时序数据，因此无法线性插值，我们用样本均值填补
    # 样条插值补全缺失值
    # df = interpolate_na(df, ['rep78'], how='spline')
    # 均值补全缺失值
    df = fill_na(df)

    # 标准化
    df = standardize(df, ['make','foreign']) # 这两列是分类变量，不需要标准化

    # 归一化
    # df = normalize(df, ['make','foreign']) # 这两列是分类变量，不需要归一化

    # 去重
    # df = drop_duplicate(df, ['foreign', 'rep78'])

    temp = df['price'].values
    temp[0] = 5
    df['price'] =temp

    # 异常值
    df2 = winsorize(df,1,99)

    ax = plt.subplot(111)
    ax.scatter(df.index, df.price.values, color='r', label='1')
    ax.plot(df2.index, df2.price, color='b', label='2')
    ax.legend(['1','2'])
    plt.show()

