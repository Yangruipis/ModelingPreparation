# Modeling_Preparation
数学建模准备工作——每日一算法，包括各种各样可复用的小函数

> /dataset

- **abalone.txt** 鲍鱼数据集 数据
- **abalone.names** 鲍鱼数据集 变量名
- **auto.csv** 大众数据集原始版
- **auto_1.csv** 大众数据集 中文替换
- **auto.mat**	大众数据集 matlab格式
- **SZIndex.csv** 上证指数数据集
- **SZIndex.desc** 上证指数数据集说明
- **international-airline-passengers.csv** International airline passengers: monthly totals in thousands. Jan 49 – Dec 60

> /优化模型

- **genetic_algorithm.py** 遗传算法
- **PSO.py** 粒子群算法
- **simulated_annealing.py** 模拟退火算法
- **sa_tsp_example.py** 模拟退火算法解决TSP问题

> /小工具

- **due_date_calculate.py** wind接口调用 + 计算期权/期货到期日
- **lasso_regression.m** lasso回归
- **ridge_regression.m** 岭回归主程序
- **ridgeRegression_func1.m** 岭回归函数1
- **trade_account.py**
    - 交易仓位类
    - 交易模拟账户类，支持期权和期货模拟交易
    - 净值指标计算类，输入净值序列，输出夏普、年化收益等
- **二分法期权计算器.cs** 根据BS公式与二分法，计算特定期权指标（权费/行权价/无风险收益/到期日/隐含波动率/标的价格）
- **Association_rules.py** 关联规则的Apriori算法，包括使用fp-growth方法寻找频繁项集
- **data_clean.py** 数据预处理

> /评价模型

- **PPE.m** 投影寻踪法主程序，确定权重
- **get_Q.m** 投影寻踪法获取目标函数值
- **constraint.m** 投影寻踪法约束条件，用以输入优化工具箱
- **pso_optimal.m** 粒子群算法求解投影寻踪法结果
- **optimal_tools.png** 优化工具箱使用方法：参数输入
- **EntropyWeight.m** 商权法确定去那种
- **SOM.py** 神经网络聚类方法
- **cluster.py** K均值和层次聚类法，包括PCA降维

> /预测模型

- **decision_tree.py** 决策树手写
- **ML_classify_mode.py** 机器学习分类模型汇总 调用sklearn包
- **neural_network.py** 神经网络BP算法，用于连续值预测
- **neural_network.m** 神经网络BP算法，调用matlab神经网络工具箱
- **SVR.py** 支持向量回归，用于连续值预测
- **HMM.py** 隐马尔科夫模型
- **evaluate.py** 预测效果评估
- **GM1_1.py** 灰色预测
- **LSTM_predict.py** 长短记忆神经网络预测模型
- **PLSR.m** 典型相关分析，偏最小二乘，研究变量间影响，尤其是多对多，并进行预测

> Some Notes
- 分层优化
    - 例1：分级排班优化建模，主要包含两层优化:一是利用飞机使用最小化模型得到每一天覆盖所有航班的最小航班串;二是利用飞机维修机会最大化模型得到覆盖所有航班串的一周飞机路线,并进行仿真
    - 例2：全国5A景点旅游路线规划，两层:一是省内区域（或者是聚类得到的景点簇）内旅游路线进行优化（TSP），二是省际优化方法
- 典型相关
    - 《基于格兰杰因果检验和典型相关的农民收入影响因素研究》
	- 《典型相关分析综述》
	- 《基于核典型相关分析和支持向量机的语音情感识别模型》


> TODO LIST:

- 格兰杰因果检验，时间序列算法 （stata搞定）
- 双种群遗传算法 (了解)
- 多车辆路径问题 (http://blog.csdn.net/wangqiuyun/article/details/7664995)
- 最小生成树 (http://blog.csdn.net/heisediwei/article/details/50326847)
- MTSP 多旅行商tsp问题 (文献很多)
- 系统动力学     (了解)
- 隶属度函数     (了解)
- 连续区间有序加权平均算子 COWA     (尚未了解)
- 插值与拟合 matlab 实现   (拟合工具箱和interp1函数)
- lingo     (放弃)
- 灰色关联度 筛选变量    (灰色关联度在变量选择之中的误区)
- latex 编译 内容填充
- 空间作图 复习
- R ggplot2 各种作图 复习 





















































