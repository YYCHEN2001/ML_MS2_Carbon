import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

class ModelEvaluator:
    """
    评估回归模型：
    1. 计算回归指标 (R2, MAE, MAPE, RMSE)
    2. 生成包含训练集和测试集结果的 DataFrame
    3. 绘制实际值 vs 预测值图 (含 y=x 参考线)
    """

    def __init__(self, model_name, y_train, y_train_pred, y_test, y_test_pred, rounding=None, font_family="Arial"):
        """
        初始化模型评估器

        参数:
            model_name (str): 模型名称
            y_train (array-like): 训练集真实值
            y_train_pred (array-like): 训练集预测值
            y_test (array-like): 测试集真实值
            y_test_pred (array-like): 测试集预测值
            rounding (dict, optional): 指定各指标的舍入位数，默认为 {'R2': 3, 'MAE': 2, 'MAPE': 2, 'RMSE': 2}
            font_family (str, optional): 设置绘图的全局字体，默认为 'Arial'
        """
        self.model_name = model_name
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        self.rounding = rounding if rounding is not None else {'R2': 3, 'MAE': 2, 'MAPE': 2, 'RMSE': 2}
        self.font_family = font_family

    def _calculate_metrics(self, y_true, y_pred):
        """
        计算并返回模型的各项指标

        参数:
            y_true: 真实值
            y_pred: 预测值

        返回:
            dict: 包含 'R2', 'MAE', 'MAPE' 和 'RMSE' 指标
        """
        r2 = round(r2_score(y_true, y_pred), self.rounding['R2'])
        mae = round(mean_absolute_error(y_true, y_pred), self.rounding['MAE'])
        mape = round(mean_absolute_percentage_error(y_true, y_pred) * 100, self.rounding['MAPE'])
        rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), self.rounding['RMSE'])
        return {'R2': r2, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}

    def metrics_to_dataframe(self):
        """
        生成包含训练集和测试集结果的 DataFrame

        返回:
            pandas.DataFrame: 包含模型名称和训练、测试指标的单行 DataFrame
        """
        metrics_train = self._calculate_metrics(self.y_train, self.y_train_pred)
        metrics_test = self._calculate_metrics(self.y_test, self.y_test_pred)

        # 组合训练集和测试集指标
        metrics = {'model': self.model_name}
        metrics.update({f"{k}_train": v for k, v in metrics_train.items()})
        metrics.update({f"{k}_test": v for k, v in metrics_test.items()})

        return pd.DataFrame([metrics])

    def plot_actual_vs_predicted(self, figtitle=None, figpath=None, save_plot=True):
        """
        绘制实际值 vs 预测值的散点图：
        1. 设置全局字体为 Arial（不改变全局字号）
        2. 图片大小固定为 (12,12)
        3. x 和 y 轴固定范围为 [0, 2000]
        4. 刻度严格每 500 一个
        5. 回归线 y = x 贯穿整个图表
        6. 刻度线加长加粗，边框加粗
        7. 仅对该图设置字号，不影响全局字号
        """
        if figtitle is None:
            figtitle = self.model_name

        # **设置全局字体为 Arial**
        plt.rcParams["font.family"] = "Times New Roman"

        # **创建画布，固定图片大小**
        plt.figure(figsize=(8, 8))

        # 绘制散点图
        plt.scatter(self.y_train, self.y_train_pred, color='blue', label='Train set', s=30, alpha=0.6)
        plt.scatter(self.y_test, self.y_test_pred, color='red', label='Test set', s=30, alpha=0.6)

        # **固定坐标轴范围**
        plt.xlim(-100, 2000)
        plt.ylim(-100, 2000)

        # **确保回归线贯穿整个图**
        plt.plot([-100, 2000], [-100, 2000], 'k--', lw=2, label='Regression Line')

        # **设置刻度，每隔 500**
        major_ticks = np.arange(0, 2001, 500)
        plt.xticks(major_ticks, fontweight='semibold', fontsize=20)  # 坐标轴刻度字号 16
        plt.yticks(major_ticks, fontweight='semibold', fontsize=20)

        # **刻度线加粗加长**
        plt.tick_params(axis='both', which='major', length=10, width=2)

        # **设置标题和轴标签**
        plt.title(figtitle, fontsize=28, fontweight='semibold', pad=10)
        plt.xlabel("Actual Values (F g\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})", fontsize=24, fontweight='semibold')
        plt.ylabel("Predicted Values (F g\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})", fontsize=24, fontweight='semibold')

        # **设置图例**
        plt.legend(frameon=False, loc='upper left', prop={'weight': 'semibold', 'size': 20})

        # **边框加粗**
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')

        # **保存或显示**
        if save_plot and figpath:
            plt.savefig(figpath, bbox_inches='tight', transparent=True, dpi=300)

        plt.show()



