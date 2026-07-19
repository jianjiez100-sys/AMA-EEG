import numpy as np

def LDS(sequence):
    """
    全向量化加速版 LDS (Linear Dynamical System) 平滑算法
    时间复杂度从 O(D * T) 降级为 O(T) 的矩阵运算，极大加速高维特征提取
    """
    # sequence shape: (timeSample, n_dims)
    N, D = sequence.shape
    if N == 0:
        return sequence

    X = sequence.T  # 转置为 (D, N)，方便按时间步切片
    u0 = np.mean(X, axis=1)  # (D,) 1024维度的均值

    # 状态空间模型常数
    V0, A, T_cov, C, sigma = 0.01, 1, 0.0001, 1, 1

    # 预分配一维数组记录状态变化 (完全不依赖数据维度)
    K = np.zeros(N)
    V = np.zeros(N)
    P = np.zeros(N)

    # 均值 u 与数据 X 有关，形状为 (D, N)
    u = np.zeros((D, N))

    # === 初始化 (t = 0) ===
    K[0] = V0 * C / (C * V0 * C + sigma)
    # 利用 NumPy 广播机制，一次性计算 1024 维
    u[:, 0] = u0 + K[0] * (X[:, 0] - C * u0)
    V[0] = (1 - K[0] * C) * V0

    # === 前向滤波 (Forward Pass) ===
    for i in range(1, N):
        P[i - 1] = A * V[i - 1] * A + T_cov
        K[i] = P[i - 1] * C / (C * P[i - 1] * C + sigma)

        u[:, i] = A * u[:, i - 1] + K[i] * (X[:, i] - C * A * u[:, i - 1])
        V[i] = (1 - K[i] * C) * P[i - 1]

    # === 后向平滑 (Backward Pass / RTS Smoother) ===
    uAll = np.zeros((D, N))
    J = np.zeros(N)

    uAll[:, N - 1] = u[:, N - 1]

    # 倒序遍历
    for i in range(N - 2, -1, -1):
        J[i] = V[i] * A / P[i]
        uAll[:, i] = u[:, i] + J[i] * (uAll[:, i + 1] - A * u[:, i])

    # 转置回去返回，形状恢复为 (timeSample, n_dims)
    return uAll.T


def LDS_acc(self, sequence):
    """
    全向量化加速版 LDS_acc (Linear Dynamical System)
    修复了原版缺少后向平滑的问题，并彻底抛弃了拖慢速度的 np.concatenate
    """
    # sequence shape: (timeSample, n_dims)
    N, D = sequence.shape
    if N == 0:
        return sequence

    X = sequence.T  # (D, N)
    u0 = np.mean(X, axis=1)  # (D,)

    V0, A, T_cov, C, sigma = 0.01, 1, 0.0001, 1, 1

    # 预分配内存，杜绝 concatenate
    K = np.zeros(N)
    V = np.zeros(N)
    P = np.zeros(N)
    u = np.zeros((D, N))

    # === 初始化 (t = 0) ===
    K[0] = V0 * C / (C * V0 * C + sigma)
    u[:, 0] = u0 + K[0] * (X[:, 0] - C * u0)
    V[0] = (1 - K[0] * C) * V0

    # === 前向滤波 (Forward Pass) ===
    for i in range(1, N):
        P[i - 1] = A * V[i - 1] * A + T_cov
        K[i] = P[i - 1] * C / (C * P[i - 1] * C + sigma)

        u[:, i] = A * u[:, i - 1] + K[i] * (X[:, i] - C * A * u[:, i - 1])
        V[i] = (1 - K[i] * C) * P[i - 1]

    # === 后向平滑 (Backward Pass / RTS Smoother) ===
    uAll = np.zeros((D, N))
    J = np.zeros(N)

    uAll[:, N - 1] = u[:, N - 1]

    for i in range(N - 2, -1, -1):
        J[i] = V[i] * A / P[i]
        uAll[:, i] = u[:, i] + J[i] * (uAll[:, i + 1] - A * u[:, i])

    return uAll.T


def running_norm(data,data_mean,data_var,decay_rate):
    # data  (subs,n_points,dim,...)
    # output data_norm:(subs,n_points,dim,...)

    data_norm = np.zeros_like(data)
    for sub in range(data.shape[0]):
        running_sum = np.zeros(data.shape[-1])
        running_square = np.zeros(data.shape[-1])
        decay_factor = 1
        for counter in range(data.shape[1]):
            data_one = data[sub, counter]
            running_sum = running_sum + data_one
            running_mean = running_sum / (counter+1)
            running_square = running_square + data_one**2
            running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

            curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
            curr_var = decay_factor*data_var + (1-decay_factor)*running_var
            decay_factor = decay_factor*decay_rate

            data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-50)
            data_norm[sub, counter, :] = data_one
    return data_norm

def running_norm_onesubsession(data,data_mean,data_var,decay_rate):
    # data  (n_points,dim,...)
    # output data_norm:(n_points,dim,...)
    # one session represent one sub

    data_norm = np.zeros_like(data)

    running_sum = np.zeros(data.shape[-1])
    running_square = np.zeros(data.shape[-1])
    decay_factor = 1

    for counter in range(data.shape[0]):
        data_one = data[counter]
        running_sum = running_sum + data_one
        running_mean = running_sum / (counter+1)
        running_square = running_square + data_one**2
        running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

        curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
        curr_var = decay_factor*data_var + (1-decay_factor)*np.maximum(running_var,0)
        decay_factor = decay_factor*decay_rate
        
        data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-50)
        data_norm[counter, :] = data_one
    return data_norm


