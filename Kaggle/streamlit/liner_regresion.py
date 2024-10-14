import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt

# 设置页面标题
st.title("多标签分类模型在线实验（支持逻辑回归决策面可视化与在线预测）")

# 上传数据集文件
uploaded_file = st.file_uploader("上传数据集（CSV格式）", type=["csv"])

if uploaded_file is not None:
    # 读取上传的 CSV 文件
    df = pd.read_csv(uploaded_file)
    st.write("数据集预览：")
    st.dataframe(df)

    # 自动检测字符类型的列，并对其进行编码
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_columns:
        st.write(f"检测到以下字符类型列：{categorical_columns}")
        st.write("这些列将会自动进行 One-Hot 编码。")
        # 对字符类型列进行 One-Hot 编码
        df = pd.get_dummies(df, columns=categorical_columns)
        st.write("编码后的数据集：")
        st.dataframe(df)

    # 用户选择特征和目标列
    st.write("请选择特征列和多个目标列（用于多标签分类）")
    all_columns = df.columns.tolist()
    feature_columns = st.multiselect("特征列", all_columns)
    target_columns = st.multiselect("目标列（多列）", all_columns)

    if feature_columns and target_columns:
        # 准备训练数据
        X = df[feature_columns]
        y = df[target_columns]

        # 选择模型
        st.write("选择要使用的模型：")
        model_option = st.selectbox(
            "模型选择",
            ("逻辑回归", "决策树分类")
        )

        # 初始化模型
        if model_option == "逻辑回归":
            base_model = LogisticRegression()
            model = OneVsRestClassifier(base_model)
        elif model_option == "决策树分类":
            base_model = DecisionTreeClassifier()
            model = MultiOutputClassifier(base_model)

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 模型评估
        y_pred = model.predict(X_test)
        st.write("分类报告：")
        st.text(classification_report(y_test, y_pred))

        # 可视化模型参数
        st.write("模型参数可视化：")

        if model_option == "逻辑回归":
            st.write("逻辑回归模型的每个类别的系数：")
            for i, class_coefs in enumerate(model.estimators_):
                st.write(f"类别 {i + 1} 系数：{class_coefs.coef_}")

        elif model_option == "决策树分类":
            # 决策树结构可视化
            st.write("决策树结构可视化：")
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_tree(base_model, feature_names=feature_columns, filled=True, ax=ax)
            st.pyplot(fig)

        # 在线输入数据预测部分
        st.write("### 在线输入特征进行预测")
        input_data = {}

        # 根据选择的特征列动态生成输入框
        for col in feature_columns:
            input_data[col] = st.number_input(f"输入 {col} 的值", value=0.0)

        # 将用户输入的数据转换为模型输入格式
        input_df = pd.DataFrame([input_data])

        # 进行预测
        if st.button("进行预测"):
            prediction = model.predict(input_df)
            st.write(f"预测结果：{prediction}")

            if model_option == "逻辑回归":
                st.write("逻辑回归的决策面（仅适用于二维特征）：")
                if X_train.shape[1] == 2:
                    # 绘制决策面
                    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
                    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                         np.arange(y_min, y_max, 0.1))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig, ax = plt.subplots()
                    ax.contourf(xx, yy, Z, alpha=0.8)
                    ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train.iloc[:, 0], edgecolors='k', marker='o', s=20)
                    ax.set_title("决策面可视化")
                    st.pyplot(fig)
                else:
                    st.write("决策面可视化仅适用于二维特征情况。")
