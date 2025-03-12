import pandas as pd
from scripts.evaluate_model import ModelEvaluator
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scripts.train_model import optuna_xgb
import joblib


data = pd.read_csv('../data/processed/dataset_reduced.csv')

# 对 Cation, Anion 进行One-hot编码
data_encoded = pd.get_dummies(data, columns=['Cation', 'Anion'])

# 划分训练集和测试集
data_encoded['target_class'] = pd.qcut(data_encoded['Cs'], q=10, labels=False)
X = data_encoded.drop(['Cs', 'target_class'], axis=1)
y = data_encoded['Cs']
stratify_column = data_encoded['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)

# optuna 自动调参
# optuna_xgb(X_train, y_train, n_trials=100, model_save_path="../models/best_xgb.pkl")

# 加载最佳模型
best_model = joblib.load("../models/best_xgb.pkl")

# 预测
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 评估
evaluator = ModelEvaluator(model_name='XGBoost', y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred)
df_metrics = evaluator.metrics_to_dataframe()
df_metrics.to_csv('../results/reports/metrics_XGBoost.csv', index=False)
print(df_metrics)

figpath = '../results/figures/avp_XGBoost.png'
evaluator.plot_actual_vs_predicted(figpath=figpath)

# Z, Period 与 CR 的相关系数过高，特征冗余
# 删除 Z, Period
X_train_r = X_train.drop(columns=['Z', 'Period'])
X_test_r = X_test.drop(columns=['Z', 'Period'])

# 重新训练模型
# optuna_xgb(X_train_r, y_train, n_trials=100, model_save_path="../models/best_xgb_r.pkl")

# 加载最佳模型
best_model_r = joblib.load("../models/best_xgb_r.pkl")

# 重新预测
y_train_r_pred = best_model_r.predict(X_train_r)
y_test_r_pred = best_model_r.predict(X_test_r)

# 重新评估
evaluator_r = ModelEvaluator(model_name='XGBoost', y_train=y_train, y_train_pred=y_train_r_pred, y_test=y_test, y_test_pred=y_test_r_pred)

df_metrics_r = evaluator_r.metrics_to_dataframe()
df_metrics_r.to_csv('../results/reports/metrics_XGBoost_r.csv', index=False)
print(df_metrics_r)

figpath = '../results/figures/avp_XGBoost_r.png'
evaluator_r.plot_actual_vs_predicted(figpath=figpath)
