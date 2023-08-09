import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report


## 载入类别名称和ID
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)


## 载入测试集预测结果表格
df = pd.read_csv('测试集预测结果.csv')
sum(df['标注类别名称'] == df['top-1-预测名称']) / len(df)


## top-n准确率
sum(df['top-n预测正确']) / len(df)

## 各类别其它评估指标
print(classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes))
# macro avg 宏平均：直接将每一类的评估指标求和取平均（算数平均值）
# weighted avg 加权平均：按样本数量（support）加权计算评估指标的平均值

report = classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes, output_dict=True)
del report['accuracy']
df_report = pd.DataFrame(report).transpose()
print(df_report)
## 补充：各类别准确率（其实就是recall）

accuracy_list = []
for fruit in tqdm(classes):
    df_temp = df[df['标注类别名称']==fruit]
    accuracy = sum(df_temp['标注类别名称'] == df_temp['top-1-预测名称']) / len(df_temp)
    accuracy_list.append(accuracy)

# 计算 宏平均准确率 和 加权平均准确率
acc_macro = np.mean(accuracy_list)
acc_weighted = sum(accuracy_list * df_report.iloc[:-2]['support'] / len(df))

accuracy_list.append(acc_macro)
accuracy_list.append(acc_weighted)

df_report['accuracy'] = accuracy_list

print(df_report)

df_report.to_csv('各类别准确率评估指标.csv', index_label='类别')

