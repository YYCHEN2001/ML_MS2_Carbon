import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import shap

data = pd.read_csv('../data/processed/dataset_reduced.csv')

data = data.drop(columns=['Z', 'Period'])
data_encoded = pd.get_dummies(data, columns=['Cation', 'Anion'])

# 划分训练集和测试集
data_encoded['target_class'] = pd.qcut(data_encoded['Cs'], q=10, labels=False)
X = data_encoded.drop(['Cs', 'target_class'], axis=1)
y = data_encoded['Cs']
stratify_column = data_encoded['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)

# 加载最佳模型
best_model = joblib.load("../models/best_xgb_r.pkl")

# 重新预测
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include=['bool']).columns})

# 训练 SHAP 解释器
explainer = shap.Explainer(best_model, X_train)

# 计算 SHAP 值
shap_values = explainer(X_train)

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
# fig.set_size_inches(8, 8, forward=True)

shap.plots.bar(shap_values, show=False, max_display=20, ax=ax)

ax.set_title("SHAP Summary Plot", fontsize=28, fontweight='bold', fontname="Times New Roman", pad=20)

for ax in fig.axes:
    ax.xaxis.label.set_size(20)  # 放大 x 轴标签字体
    ax.xaxis.label.set_fontweight('bold')  # 加粗 x 轴标签字体
    ax.xaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.yaxis.label.set_size(20)  # 放大 y 轴标签字体
    ax.yaxis.label.set_fontweight('bold')  # 加粗 y 轴标签字体
    ax.yaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)  # 放大刻度标签字体
        label.set_fontweight('bold')  # 加粗刻度标签字体
        label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

# **移除 `bbox_inches='tight'`，防止裁剪**
output_path = '../results/figures/shap_values_fixed.png'
plt.tight_layout()
plt.savefig(output_path, transparent=False)

# **检查最终图片尺寸**
from PIL import Image
img = Image.open(output_path)
print(f"✅ Final Image Size: {img.size[0]} x {img.size[1]} pixels")  # 应该是 3000 x 2400

plt.show()

