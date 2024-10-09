import numpy as np
import pandas as pd

class ID3:
    def __init__(self):
        self.tree = None  # 初始化决策树

    def entropy(self, y):
        # 计算熵
        value_counts = np.unique(y, return_counts=True)[1]  # 统计每个类别的数量
        probabilities = value_counts / len(y)  # 计算每个类别的概率
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # 计算熵，加上小常数避免log(0)

    def information_gain(self, X, y, feature):
        # 计算信息增益
        total_entropy = self.entropy(y)  # 计算当前数据集的总熵
        feature_values, feature_counts = np.unique(X[:, feature], return_counts=True)  # 获取特征的所有取值及其计数
        
        weighted_entropy = 0  # 加权熵初始化
        for value in feature_values:
            subset_y = y[X[:, feature] == value]  # 获取特征取值为当前值的子集的标签
            weighted_entropy += (feature_counts[np.where(feature_values == value)[0][0]] / len(y)) * self.entropy(subset_y)  # 累加加权熵

        return total_entropy - weighted_entropy  # 返回信息增益

    def best_feature(self, X, y):
        # 寻找最佳特征
        gains = [self.information_gain(X, y, feature) for feature in range(X.shape[1])]  # 计算每个特征的信息增益
        return np.argmax(gains)  # 返回信息增益最大的特征索引

    def build_tree(self, X, y):
        # 构建决策树
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]  # 如果所有样本都属于同一类别，则返回该类别

        if X.shape[1] == 0:
            return np.unique(y, return_counts=True)[0][np.argmax(np.unique(y, return_counts=True)[1])]  # 如果没有特征可分裂，返回多数类别

        best_feat = self.best_feature(X, y)  # 找到最佳特征
        tree = {best_feat: {}}  # 创建树节点
        feature_values = np.unique(X[:, best_feat])  # 获取最佳特征的所有取值

        for value in feature_values:
            subset_X = X[X[:, best_feat] == value]  # 获取当前特征取值为value的子集
            subset_y = y[X[:, best_feat] == value]  # 获取当前特征取值为value的子集标签

            # 移除已使用的特征，递归构建子树
            subtree = self.build_tree(np.delete(subset_X, best_feat, axis=1), subset_y)
            tree[best_feat][value] = subtree  # 将子树添加到当前节点

        return tree  # 返回构建好的决策树

    def fit(self, X, y):
        # 训练模型
        self.tree = self.build_tree(X, y)  # 构建决策树

    def predict_one(self, tree, x):
        # 对单个样本进行预测
        if not isinstance(tree, dict):
            return tree  # 如果当前节点不是字典，返回叶节点类别

        feature = next(iter(tree))  # 获取当前节点的特征
        feature_value = x[feature]  # 获取当前样本的特征值

        if feature_value in tree[feature]:
            return self.predict_one(tree[feature][feature_value], x)  # 递归到下一个节点
        else:
            return None  # 如果特征值不在树中，则返回None

    def predict(self, X):
        # 对多个样本进行预测
        return [self.predict_one(self.tree, x) for x in X]  # 对每个样本进行预测

# 使用示例
if __name__ == "__main__":
    # 示例数据集
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
        'Windy': [False, True, False, False, False, True, True, False, True, True],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
    }

    df = pd.DataFrame(data)  # 将数据转换为DataFrame
    X = df.iloc[:, :-1].values  # 获取特征
    y = df.iloc[:, -1].values    # 获取标签

    # 将特征转换为数值型（可以使用标签编码或独热编码）
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for i in range(X.shape[1]):
        X[:, i] = le.fit_transform(X[:, i])  # 对每个特征进行标签编码

    # 训练ID3模型
    model = ID3()
    model.fit(X, y)  # 训练模型

    # 打印决策树
    print("决策树结构:", model.tree)  # 输出决策树结构

    # 预测
    test_data = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])  # 示例测试数据
    predictions = model.predict(test_data)  # 对测试数据进行预测
    print("预测结果:", predictions)  # 输出预测结果
