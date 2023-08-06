import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='SimHei') # 中文字体
import pandas as pd
import numpy as np

## 载入类别名称和ID
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)

## 载入测试集预测结果表格
df = pd.read_csv('测试集预测结果.csv')

## 绘制某一类别的PR曲线
specific_class = '荔枝'
# 二分类标注
y_test = (df['标注类别名称'] == specific_class)
# 二分类预测置信度
y_score = df['荔枝-预测置信度']
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
AP = average_precision_score(y_test, y_score, average='weighted')

def getoneimg():
    plt.figure(figsize=(12, 8))
    # 绘制 PR 曲线
    plt.plot(recall, precision, linewidth=5, label=specific_class)

    # 随机二分类模型
    # 阈值小，所有样本都被预测为正类，recall为1，precision为正样本百分比
    # 阈值大，所有样本都被预测为负类，recall为0，precision波动较大
    plt.plot([0, 0], [0, 1], ls="--", c='.3', linewidth=3, label='随机模型')
    plt.plot([0, 1], [0.5, sum(y_test == 1) / len(df)], ls="--", c='.3', linewidth=3)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.rcParams['font.size'] = 22
    plt.title('{} PR曲线  AP:{:.3f}'.format(specific_class, AP))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig('{}-PR曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
    plt.show()

getoneimg()


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

get_line_arg()

ap_list = []
def getallimg():
    plt.figure(figsize=(14, 10))
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    # plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.rcParams['font.size'] = 22
    plt.grid(True)

    for each_class in classes:
        y_test = list((df['标注类别名称'] == each_class))
        y_score = list(df['{}-预测置信度'.format(each_class)])
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        AP = average_precision_score(y_test, y_score, average='weighted')
        plt.plot(recall, precision, **get_line_arg(), label=each_class)
        plt.legend()
        ap_list.append(AP)

    plt.legend(loc='best', fontsize=12)
    plt.savefig('各类别PR曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
    plt.show()


## 将AP增加至`各类别准确率评估指标`表格中
df_report = pd.read_csv('各类别准确率评估指标.csv')
# 计算 AUC值 的 宏平均 和 加权平均
macro_avg_auc = np.mean(ap_list)
weighted_avg_auc = sum(ap_list * df_report.iloc[:-2]['support'] / len(df))
df_report['AP'] = ap_list
df_report.to_csv('各类别准确率评估指标.csv', index=False)

