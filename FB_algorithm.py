import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

hmm_params = scio.loadmat('hmm_params.mat')
"""
params:
1.transition：状态转移矩阵
2.prior：状态的先验
3.emission：发射概率矩阵
4.price_change：观测
"""

def Forward_Backward(HmmParams, num_HiddenState, len_HiddenState):
    """ 3个状态，时间序列长度为100 """
    def Forward(params, num, len):
        alpha = np.zeros((num, len))  # 前向传播矩阵

        for i in range(num):
            x = params['price_change'][0][i] - 1
            alpha[i][0] = params['emission'][i][x] * params['prior'][i][0]

        for k in range(1, len):
            for m in range(num):
                sum = 0
                for n in range(num):
                    """ 
                    alpha(k-1)[z(k-1)] * P(z(k)|z(k-1)) * P(xk|zk)
                    x表示xk；
                    最内层循环n表示z(k-1)，遍历前隐藏状态的所有可能状态； 
                    m表示当前状态zk，同样遍历所有可能状态。
                    """

                    x = params['price_change'][0][k] - 1
                    sum += alpha[n][k-1] \
                           * params['transition'][n][m] \
                           * params['emission'][m][x]

                alpha[m][k] = sum

        return alpha


    def Backward(params, num, len):
        beta = np.zeros((num, len))  # 后向传播矩阵

        for i in range(num):
            beta[i][len - 1] = 1

        for k in np.arange(len - 2, -1, -1):
            for m in range(num):
                sum = 0
                for n in range(num):
                    """ 
                    beta(k+1)[z(k+1)] * P(z(k+1)|z(k1)) * P(x(k+1)|z(k+1))
                    x表示x(k+1)；
                    最内层循环n表示z(k+1)，遍历前隐藏状态的所有可能状态； 
                    m表示当前状态zk，同样遍历所有可能状态。
                    """
                    x = params['price_change'][0][k+1] - 1
                    sum += beta[n][k + 1] \
                           * params['transition'][m][n] \
                           * params['emission'][n][x]
                beta[m][k] = sum

        return beta

    alpha = Forward(HmmParams, num_HiddenState, len_HiddenState)
    beta = Backward(hmm_params, num_HiddenState, len_HiddenState)
    product = alpha * beta
    prob = np.zeros_like(product)
    pred_order = np.zeros((1, len_HiddenState), dtype=int)
    for i in range(100):
        sum = np.sum(product[:, i])
        max_id = 0
        for j in range(3):
            prob[j, i] = product[j, i] / sum
            if prob[j, i] > prob[max_id, i]:
                max_id = j
        pred_order[0, i] = max_id + 1

    return np.around(prob, decimals=3), pred_order


if __name__ == '__main__':
    len_of_HiddenState = 100
    prob, pred = Forward_Backward(hmm_params, num_HiddenState=3, len_HiddenState=len_of_HiddenState)
    x = np.arange(0, len_of_HiddenState, 1)
    print(pred)

    plt.style.use('ggplot')
    plt.title('State Sequence')
    plt.xlabel('Day')
    plt.ylabel('Probability')
    plt.plot(x, prob[2], color='red', label='Stable')
    plt.plot(x, prob[1], color='blue', label='Bear')
    plt.plot(x, prob[0], color='green', label='Bull')
    plt.legend()
    plt.show()




