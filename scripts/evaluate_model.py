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

    def plot_actual_vs_predicted(self, figtitle=None, figpath=None):
        if figtitle is None:
            figtitle = self.model_name

        if figpath is None:
            figpath = f"../results/figures/avp_{self.model_name}.png"

        # **设置全局字体为 Arial**
        plt.rcParams["font.family"] = "Arial" # Adjusted for better compatibility

        # **创建画布，固定图片大小**
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # Increase figure size
        # fig.suptitle(figtitle, fontsize=24, fontname="Arial", y=0.98)

        # 绘制散点图
        ax.scatter(self.y_train, self.y_train_pred, color=(30/255, 144/255, 255/255), label='Train set', s=50, alpha=0.5)
        ax.scatter(self.y_test, self.y_test_pred, color=(255/255, 144/255, 200/255), label='Test set', s=50, alpha=0.5)

        # **固定坐标轴范围**
        ax.set_xlim(-100, 2000)  # Adjusted to ensure no negative values
        ax.set_ylim(-100, 2000)  # Adjusted to ensure no negative values

        # **生成回归线 y = x**
        x_vals = np.linspace(-100, 2000, 100)
        y_vals = x_vals  # y = x

        # **绘制回归线**
        ax.plot(x_vals, y_vals, 'k--', lw=2, label='Regression Line')

        # **刻度线加粗加长**
        ax.tick_params(axis='both', which='both', length=10, width=2, colors='black')

        # **不显示网格线**
        ax.grid(False)

        # **设置刻度，每隔 500**
        major_ticks = np.arange(0, 2001, 500)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)

        # **设置刻度线和刻度标签的大小**
        ax.tick_params(axis='both', which='major', length=5, width=2, labelsize=18)  # Increased size

        # **设置标题和轴标签**
        ax.set_title(figtitle, fontsize=24, pad=10)  # Increased font size
        ax.set_xlabel("Actual Values", fontsize=24, labelpad=10)  # Increased font size
        ax.set_ylabel("Predicted Values", fontsize=24)

        # **设置图例**
        ax.legend(frameon=False, loc='upper left', fontsize=16)  # Increased font size

        # **边框加粗**
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')

        # **保存或显示**
        plt.tight_layout()
        plt.savefig(figpath, transparent=False)
        plt.show()


