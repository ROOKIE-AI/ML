import numpy as np
import pandas as pd

class C45:
    def __init__(self):
        self.tree = None  # 用于存储决策树

    def entropy(self, y):
        """ 计算熵 """
        value_counts = np.unique(y, return_counts=True)[1]  # 获取每个类别的计数
        probabilities = value_counts / len(y)  # 计算每个类别的概率
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # 计算熵

    def information_gain_ratio(self, X, y, feature):
        """ 计算信息增益比 """
        total_entropy = self.entropy(y)  # 计算数据集的总熵
        feature_values, feature_counts = np.unique(X[:, feature], return_counts=True)  # 获取特征取值及其计数

        weighted_entropy = 0  # 初始化加权熵
        split_info = 0  # 初始化分裂信息，用于计算增益比
        for value in feature_values:
            subset_y = y[X[:, feature] == value]  # 获取当前特征取值下的子集标签
            prob = len(subset_y) / len(y)  # 当前特征取值的概率
            weighted_entropy += prob * self.entropy(subset_y)  # 加权熵累加
            split_info -= prob * np.log2(prob + 1e-10)  # 计算分裂信息

        info_gain = total_entropy - weighted_entropy  # 信息增益
        if split_info == 0:  # 如果分裂信息为0，直接返回信息增益（避免除以0）
            return info_gain
        return info_gain / split_info  # 返回信息增益比

    def best_feature(self, X, y):
        """ 寻找最佳特征 """
        gain_ratios = [self.information_gain_ratio(X, y, feature) for feature in range(X.shape[1])]  # 计算每个特征的信息增益比
        return np.argmax(gain_ratios)  # 返回增益比最大的特征索引

    def build_tree(self, X, y, feature_names):
        """ 构建决策树 """
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]  # 如果所有样本属于同一类，返回该类

        if X.shape[1] == 0:
            return np.unique(y, return_counts=True)[0][np.argmax(np.unique(y, return_counts=True)[1])]  # 如果没有特征，返回多数类别

        best_feat = self.best_feature(X, y)  # 找到最佳特征
        best_feat_name = feature_names[best_feat]  # 获取最佳特征的名称
        tree = {best_feat_name: {}}  # 创建树节点

        feature_values = np.unique(X[:, best_feat])  # 获取最佳特征的所有取值
        for value in feature_values:
            subset_X = X[X[:, best_feat] == value]  # 获取当前特征取值下的数据子集
            subset_y = y[X[:, best_feat] == value]  # 获取子集标签
            subtree = self.build_tree(np.delete(subset_X, best_feat, axis=1), subset_y, np.delete(feature_names, best_feat))  # 递归构建子树
            tree[best_feat_name][value] = subtree  # 将子树添加到当前节点

        return tree  # 返回构建好的决策树

    def fit(self, X, y, feature_names):
        """ 训练模型，构建决策树 """
        self.tree = self.build_tree(X, y, feature_names)  # 递归构建决策树

    def predict_one(self, tree, x, feature_names):
        """ 对单个样本进行预测 """
        if not isinstance(tree, dict):
            return tree  # 如果当前节点不是字典，说明到达叶节点，返回预测类别

        feature = next(iter(tree))  # 获取当前节点的特征
        feature_index = feature_names.index(feature)  # 获取该特征在数据中的索引
        feature_value = x[feature_index]  # 获取当前样本在该特征上的取值

        if feature_value in tree[feature]:
            return self.predict_one(tree[feature][feature_value], x, feature_names)  # 递归到下一个节点
        else:
            return None  # 如果取值不在树中，返回None

    def predict(self, X, feature_names):
        """ 对多个样本进行预测 """
        return [self.predict_one(self.tree, x, feature_names) for x in X]  # 对每个样本进行预测


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
    y = df.iloc[:, -1].values  # 获取标签
    feature_names = df.columns[:-1].tolist()  # 获取特征名称

    # 将离散特征转换为数值型（可以使用标签编码）
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for i in range(X.shape[1]):
        X[:, i] = le.fit_transform(X[:, i])  # 对每个特征进行标签编码

    # 训练C4.5模型
    model = C45()
    model.fit(X, y, feature_names)  # 训练模型

    # 打印决策树
    print("决策树结构:", model.tree)  # 输出决策树结构

    # 预测
    test_data = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])  # 示例测试数据
    predictions = model.predict(test_data, feature_names)  # 对测试数据进行预测
    print("预测结果:", predictions)  # 输出预测结果
