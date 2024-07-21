""" 
lab_utils_common.py
    functions common to all optional labs, Course 1, Week 2 
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 的样式
plt.style.use('./deeplearning.mplstyle')

# 定义一些颜色
dlblue = '#0096ff'
dlorange = '#FF9300'
dldarkred = '#C00000'
dlmagenta = '#FF40FF'
dlpurple = '#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

# 创建一个字典来存储颜色
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')


##########################################################
# 回归函数
##########################################################

# 计算代价的函数
def compute_cost_matrix(X, y, w, b, verbose=False):
    """
    计算线性回归的代价
    参数:
      X (ndarray (m,n)): 数据，m个样本，每个样本有n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数  
      b (scalar)       : 模型参数
      verbose : (Boolean) 如果为True，打印中间的f_wb值
    返回:
      cost: (scalar) 代价
    """
    m = X.shape[0]  # m是样本数

    # 计算所有样本的f_wb
    f_wb = X @ w + b  
    # 计算代价
    total_cost = (1/(2*m)) * np.sum((f_wb-y)**2)

    if verbose: print("f_wb:")
    if verbose: print(f_wb)

    return total_cost

# 计算梯度的函数
def compute_gradient_matrix(X, y, w, b):
    """
    计算线性回归的梯度
    参数:
      X (ndarray (m,n)): 数据，m个样本，每个样本有n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数  
      b (scalar)       : 模型参数
    返回:
      dj_dw (ndarray (n,1)): 相对于参数w的代价梯度
      dj_db (scalar):        相对于参数b的代价梯度
    """
    m,n = X.shape  # m是样本数，n是特征数
    f_wb = X @ w + b  # 计算所有样本的f_wb
    e = f_wb - y  # 计算误差
    dj_dw = (1/m) * (X.T @ e)  # 计算相对于w的梯度
    dj_db = (1/m) * np.sum(e)  # 计算相对于b的梯度

    return dj_db, dj_dw

# 多变量compute_cost的循环版本
def compute_cost(X, y, w, b):
    """
    计算代价
    参数:
      X (ndarray (m,n)): 数据，m个样本，每个样本有n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数  
      b (scalar)       : 模型参数
    返回:
      cost (scalar)    : 代价
    """
    m = X.shape[0]  # 样本数
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # 计算第i个样本的预测值
        cost = cost + (f_wb_i - y[i])**2  # 累加平方误差
    cost = cost / (2 * m)  # 计算平均代价
    return cost 

# 计算梯度的函数
def compute_gradient(X, y, w, b):
    """
    计算线性回归的梯度
    参数:
      X (ndarray (m,n)): 数据，m个样本，每个样本有n个特征
      y (ndarray (m,)) : 目标值
      w (ndarray (n,)) : 模型参数  
      b (scalar)       : 模型参数
    返回:
      dj_dw (ndarray Shape (n,)): 相对于参数w的代价梯度
      dj_db (scalar):             相对于参数b的代价梯度
    """
    m, n = X.shape  # m是样本数，n是特征数
    dj_dw = np.zeros((n,))  # 初始化w的梯度
    dj_db = 0.  # 初始化b的梯度

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]  # 计算第i个样本的误差
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]  # 计算w的梯度
        dj_db = dj_db + err  # 计算b的梯度
    dj_dw = dj_dw / m  # 计算w的平均梯度
    dj_db = dj_db / m  # 计算b的平均梯度

    return dj_db, dj_dw
