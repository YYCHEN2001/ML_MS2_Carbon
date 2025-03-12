import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import joblib


def optuna_xgb(X_train, y_train, n_trials=100, model_save_path="best_model.pkl"):
    """
    使用 Optuna 调参，并训练 XGBRegressor 最佳模型后保存到文件。

    参数:
        X_train: 训练特征数据
        y_train: 训练目标数据
        n_trials: 调参试验次数，默认为 100
        model_save_path: 保存模型的路径，默认为 "best_model.pkl"

    返回:
        study: Optuna 的研究对象
        best_model: 训练好的最佳模型
    """

    def objective(trial):
        # 定义参数搜索空间
        param = {
            'n_estimators': 300,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        }

        model = XGBRegressor(**param)
        # 进行 5 折交叉验证，使用负的 MAPE 作为评估指标
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error')
        return -scores.mean()

    # 创建 Optuna 的 study 对象，目标是使评分最小化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # 输出最佳 trial 的结果
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params:')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # 根据最佳参数构建模型，并在全训练集上拟合
    best_params = trial.params.copy()
    best_params['n_estimators'] = 300  # 固定 n_estimators 参数
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # 将训练好的最佳模型保存到指定文件
    joblib.dump(best_model, model_save_path)
    print("Best model saved to:", model_save_path)

    return study, best_model

# 示例调用（在实际使用时，确保 X_train, y_train 已经定义）
# study, best_model = tune_and_save_model(X_train, y_train, n_trials=100, model_save_path="best_model.pkl")
