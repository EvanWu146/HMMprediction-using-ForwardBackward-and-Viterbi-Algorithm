import scipy.io as scio
import numpy as np
from FB_algorithm import Forward_Backward

def FuturePrediction(transition, emission, price_change, previousPred):
    T = len(previousPred)
    _next28Day = np.zeros((28,3))  # 将来28天的隐藏变量预测矩阵
    obsState = np.zeros((28,5))  # 将来28天的观测变量预测矩阵
    result = np.zeros((1, 28))  # 28天的观测结果

    # 初始化将来28天内第一天的市场状态
    _next28Day[0] = np.dot(previousPred[T - 1], transition) \
                         * emission[:, price_change[T]]
    obsState[0] = np.dot(_next28Day[0].T, emission)

    # 预测将来的28天
    for i in range(1, 28):
        temp = np.dot(_next28Day[i - 1],transition)
        _next28Day[i] = temp * emission[:, price_change[i + T]]
        obsState[i] = np.dot(_next28Day[i], emission)

    for i in range(28):
        obsState[i] = obsState[i] / np.sum(obsState[i])
        for j in range(1, 5):
            obsState[i][j] += obsState[i][j - 1]

    # 随机采样
    for i in range(100):
        for j in range(28):
            rand = np.random.random()
            for k in range(5):
                if rand < obsState[j][k]:
                    result[0][j] += k
                    break

    result = (result/100)[0]
    delta = price_change[100:128] - result  # 预测值与真实值的差别
    return result, delta


hmm_params = scio.loadmat('hmm_params.mat')
prob, pred = Forward_Backward(hmm_params, num_HiddenState=3, len_HiddenState=100)
prob = prob.transpose(1, 0)
result, delta = FuturePrediction(
                        hmm_params['transition'],
                        hmm_params['emission'],
                        hmm_params['price_change'][0] - 1,
                        prob)
print('预测均值: ', result)
print('观测结果: ', hmm_params['price_change'][0][100:128]-1)
print('预测/观测差值: ', delta)
print('平均误差: ', np.average(delta))
print('预测方差: ', np.var(delta))

default_diff = 0.99
total = len(delta)
correct = 0
for i in delta:
    if np.abs(i) < default_diff:
        correct += 1
print('准确率:', correct/total)

