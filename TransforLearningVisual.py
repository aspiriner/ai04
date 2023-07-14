import pandas as pd

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rc("font",family='SimHei') # 中文字体
import matplotlib
print(matplotlib.matplotlib_fname())

plt.plot([1,2,3], [100,500,300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()