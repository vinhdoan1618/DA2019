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

#Bieu do cot
data['Rate'].value_counts().plot(kind='bar')
plt.show()
data['Promotion'].value_counts().plot(kind='bar')
plt.show()
#Bieu do tron
# rate_group=data.groupby('Rate').size()
# rate_group.plot(kind='pie', autopct='%.2f%%')
# plt.show()
#pie chart
rate_group=data.groupby('Rate').size()
colors = ['gold', 'yellowgreen', 'lightcoral','red','purple']
explode = (0.1, 0.1, 0.1,0.1,0.1)  # explode 1st slice
rate_group.plot(kind='pie', figsize=[8,6], autopct='%.2f%%', colors=colors, explode=explode)
plt.show()

# hist = sns.FacetGrid(data=data, col='Rate')
# hist.map(plt.hist, 'Review_length', bins=50)
# plt.show()
sns.boxplot(x='Rate', y='Review_length', data=data)
plt.show()

data['Device'].value_counts().plot(kind='bar')
plt.show()
