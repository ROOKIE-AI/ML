from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np

class MLModelPipeline:
    def __init__(self, model, model_name):
        """
        初始化机器学习模型流水线。
        
        参数:
        - model: 机器学习模型 (如 RandomForestRegressor)
        - model_name: 模型名称 (如 'RandomForest')
        """
        self.model = model
        self.model_name = model_name
        self.trained_model = None

    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        划分数据集为训练集、验证集和测试集。
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - test_size: 测试集比例
        - val_size: 验证集比例 (从训练集中再划分)
        - random_state: 随机数种子
        """
        # 先划分为训练集+验证集 和 测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # 再从训练集中划分出验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def cross_validate_model(self, X_train, y_train, cv_splits=5):
        """
        对模型进行 K 折交叉验证，并返回交叉验证的 MSE 和 R²得分。
        
        参数:
        - X_train: 训练集特征矩阵
        - y_train: 训练集目标变量
        - cv_splits: 交叉验证的折数 (默认为5折)
        
        返回:
        - mse_scores: 每次交叉验证的均方误差 (MSE)
        - r2_scores: 每次交叉验证的R²得分
        """
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        # 交叉验证MSE
        mse_scores = cross_val_score(self.model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        # 交叉验证R²
        r2_scores = cross_val_score(self.model, X_train, y_train, cv=kf, scoring='r2')
        
        return -mse_scores, r2_scores

    def train(self, X_train, y_train):
        """
        训练模型。
        
        参数:
        - X_train: 训练集特征矩阵
        - y_train: 训练集目标变量
        """
        self.trained_model = self.model.fit(X_train, y_train)
        
    def evaluate(self, X_val, y_val, error_threshold=0.1):
        """
        评估模型在验证集上的性能，并基于误差阈值计算准确率。
        
        参数:
        - X_val: 验证集特征矩阵
        - y_val: 验证集目标变量
        - error_threshold: 误差阈值，用于计算准确率，默认值为0.1 (即误差小于10%)
        
        返回:
        - val_mse: 验证集上的均方误差 (MSE)
        - val_r2: 验证集上的 R²得分
        - accuracy: 基于误差阈值的预测准确率
        """
        y_val_pred = self.trained_model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # 基于误差阈值计算准确率
        errors = np.abs((y_val_pred - y_val) / y_val)
        accuracy = np.mean(errors < error_threshold)  # 误差小于阈值的比例
        
        return val_mse, val_r2, accuracy

    def test(self, X_test, y_test, error_threshold=0.1):
        """
        在测试集上评估模型性能，并基于误差阈值计算准确率。
        
        参数:
        - X_test: 测试集特征矩阵
        - y_test: 测试集目标变量
        - error_threshold: 误差阈值，用于计算准确率，默认值为0.1 (即误差小于10%)
        
        返回:
        - test_mse: 测试集上的均方误差 (MSE)
        - test_r2: 测试集上的 R²得分
        - accuracy: 基于误差阈值的预测准确率
        """
        y_test_pred = self.trained_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 基于误差阈值计算准确率
        errors = np.abs((y_test_pred - y_test) / y_test)
        accuracy = np.mean(errors < error_threshold)  # 误差小于阈值的比例
        
        return test_mse, test_r2, accuracy

