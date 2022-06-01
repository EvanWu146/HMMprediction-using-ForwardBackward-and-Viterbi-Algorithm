import numpy as np
import scipy.io as scio

def Viterbi(prior, emission, transition, obs):
    N = len(transition)  # 隐藏状态种类数
    T = len(obs)  # 观测序列长度
    sigma = np.zeros((N, T))
    psi = np.zeros((N, T))
    bestPath = np.zeros((T))

    # 初始化
    for i in range(N):
        x = obs[0]
        sigma[i, 0] = prior[i] * emission[i, x]
        psi[0, i] = 0

    # 递推
    for t in range(1, T):
        for i in range(N):
            tempVal = 0
            for j in range(N):
                xt = obs[t]
                tempVal = sigma[j, t-1] * transition[j, i] * emission[i, xt]
                if tempVal > sigma[i, t]:
                    sigma[i, t] = tempVal

            tempList = [sigma[j, t-1] * transition[j, i] for j in range(0, N)]
            psi[i, t] = tempList.index(max(tempList))

    # 终止
    finalProb = max([sigma[i, T-1] for i in range(N)])
    tempList = [sigma[i, T-1] for i in range(N)]
    Terminal = tempList.index(max(tempList))

    # 最优路径回溯
    bestPath[T-1] = Terminal
    for t in np.arange(T-2, -1, -1):
        bestPath[t] = psi[int(bestPath[t+1]), t+1]

    return bestPath, finalProb


params = scio.loadmat('hmm_params.mat')
path, prob = Viterbi(prior=params['prior'],
                     emission=params['emission'],
                     transition=params['transition'],
                     obs=params['price_change'][0][0:100] - 1)


print(path)
state_name = ['牛市', '熊市', '稳定市场']
for i in range(100):
    print('Day', i, ' :', state_name[int(path[i])])
