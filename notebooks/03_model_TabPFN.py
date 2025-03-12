import pandas as pd
from scripts.evaluate_model import ModelEvaluator
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor


data = pd.read_csv('../data/processed/dataset_reduced.csv')

# 对 Cation, Anion 进行One-hot编码
data_encoded = pd.get_dummies(data, columns=['Cation', 'Anion'])

# 划分训练集和测试集
data_encoded['target_class'] = pd.qcut(data_encoded['Cs'], q=10, labels=False)
X = data_encoded.drop(['Cs', 'target_class'], axis=1)
y = data_encoded['Cs']
stratify_column = data_encoded['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)

# 初始化TabPFN回归模型
model = TabPFNRegressor()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

evaluator = ModelEvaluator(model_name='TabPFN', y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred)
df_metrics = evaluator.metrics_to_dataframe()
df_metrics.to_csv('../results/reports/metrics_TabPFN.csv', index=False)
print(df_metrics)

figpath = '../results/figures/avp_TabPFN.png'
evaluator.plot_actual_vs_predicted(figpath=figpath)

# Z, Period 与 CR 的相关系数过高，特征冗余
# 删除 Z, Period
X_train_r = X_train.drop(columns=['Z', 'Period'])
X_test_r = X_test.drop(columns=['Z', 'Period'])

# 初始化TabPFN回归模型
model_r = TabPFNRegressor()

# 重新训练模型
model_r.fit(X_train_r, y_train)

# 重新预测
y_train_r_pred = model_r.predict(X_train_r)
y_test_r_pred = model_r.predict(X_test_r)

# 重新评估
evaluator_r = ModelEvaluator(model_name='TabPFN', y_train=y_train, y_train_pred=y_train_r_pred, y_test=y_test, y_test_pred=y_test_r_pred)

df_metrics_r = evaluator_r.metrics_to_dataframe()
df_metrics_r.to_csv('../results/reports/metrics_TabPFN_r.csv', index=False)
print(df_metrics_r)

figpath = '../results/figures/avp_TabPFN_r.png'
evaluator_r.plot_actual_vs_predicted(figpath=figpath)