import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)

df = pd.read_csv('测试集预测结果.csv')

## 绘制某一类别的ROC曲线
specific_class = '荔枝'
# 二分类标注
y_test = (df['标注类别名称'] == specific_class)
# 二分类置信度
y_score = df['荔枝-预测置信度']
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, y_score)

def getoneimg():
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, linewidth=5, label=specific_class)
    plt.plot([0, 1], [0, 1], ls="--", c='.3', linewidth=3, label='随机模型')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.rcParams['font.size'] = 22
    plt.title('{} ROC曲线  AUC:{:.3f}'.format(specific_class, auc(fpr, tpr)))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.grid(True)

    plt.savefig('{}-ROC曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
    plt.show()

    # yticks = ax.yaxis.get_major_ticks()
    # yticks[0].label1.set_visible(False)
print(auc(fpr, tpr))



## 绘制所有类别的ROC曲线
from matplotlib import colors as mcolors
import random
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']
def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg


auc_list = []
def getallimg():
    plt.figure(figsize=(14, 10))
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.plot([0, 1], [0, 1], ls="--", c='.3', linewidth=3, label='随机模型')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.rcParams['font.size'] = 22
    plt.grid(True)

    for each_class in classes:
        y_test = list((df['标注类别名称'] == each_class))
        y_score = list(df['{}-预测置信度'.format(each_class)])
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, **get_line_arg(), label=each_class)
        plt.legend()
        auc_list.append(auc(fpr, tpr))

    plt.legend(loc='best', fontsize=12)
    plt.savefig('各类别ROC曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
    plt.show()

df_report = pd.read_csv('各类别准确率评估指标.csv')
# 计算 AUC值 的 宏平均 和 加权平均
macro_avg_auc = np.mean(auc_list)
weighted_avg_auc = sum(auc_list * df_report.iloc[:-2]['support'] / len(df))
auc_list.append(macro_avg_auc)
auc_list.append(weighted_avg_auc)
df_report['AUC'] = auc_list
df_report.to_csv('各类别准确率评估指标.csv', index=False)