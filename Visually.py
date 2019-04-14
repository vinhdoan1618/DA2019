import os
import io
from selenium import webdriver
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data= pd.read_excel("Data/Data_Processed.xlsx",error_bad_lines=False,encoding='utf-8')
vocab=pd.read_excel("Data/Vocabulary.xlsx",error_bad_lines=False,encoding='utf-8',index_col=False)
#Xoa hang voi rate = 0
indexZero=data[data['Rate'] == 0].index
data.drop(indexZero,inplace=True)

pd.set_option('display.expand_frame_repr', False)
print(data)

#Bieu do cot
data['Rate'].value_counts().plot(kind='bar')
plt.xlabel('Rate')
plt.ylabel('Số lượng')
plt.title('Biểu đồ cột biểu diển số sao')
plt.show()
data['Emotion'].value_counts().plot(kind='bar')
plt.xlabel('Cảm xúc')
plt.ylabel('Số lượng')
plt.title('Biểu đồ cột biểu diển số lượng cảm xúc')
plt.show()
#Bieu do tron
# rate_group=data.groupby('Rate').size()
# rate_group.plot(kind='pie', autopct='%.2f%%')
# plt.show()
#pie chart
rate_group=data.groupby('Emotion').size()
colors = ['yellow', 'brown',]
rate_group.plot(kind='pie', figsize=[8,6], autopct='%.2f%%', colors=colors)
plt.title('Biểu đồ tròn thể hiện tỉ lệ phần trăm cảm x ')
plt.show()

# hist = sns.FacetGrid(data=data, col='Rate')
# hist.map(plt.hist, 'Review_length', bins=50)
# plt.show()
sns.boxplot(x='Rate', y='Review_length', data=data)
plt.show()

data['Device'].value_counts().plot(kind='bar')
plt.xlabel('Hãng điện thoại')

plt.ylabel('Số lượng')

plt.title('Biểu đồ cột biểu diển số lượng hãng điện thoại')
plt.show()

g = sns.FacetGrid(data=data, col='Emotion')
g.map(plt.hist, 'Review_length', bins=50)
plt.show()