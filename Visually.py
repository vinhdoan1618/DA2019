import os
import io
from selenium import webdriver
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data= pd.read_csv("Data_Processed",error_bad_lines=False,encoding='utf-8')
vocab=pd.read_csv("Vocabulary",error_bad_lines=False,encoding='utf-8',index_col=False)
#Xoa hang voi rate = 0
indexZero=data[data['Rate'] == 0].index
data.drop(indexZero,inplace=True)
#Bieu do cot
data['Rate'].value_counts().plot(kind='bar')

#Bieu do tron
#rate_group=data.groupby('Rate').size()
#rate_group.plot(kind='pie', autopct='%.2f%%')

#pie chart
#rate_group=data.groupby('Rate').size()
#colors = ['gold', 'yellowgreen', 'lightcoral','red','purple']
#explode = (0.1, 0.1, 0.1,0.1,0.1)  # explode 1st slice
#rate_group.plot(kind='pie', figsize=[8,6], autopct='%.2f%%', colors=colors, explode=explode)


#hist = sns.FacetGrid(data=data, col='Rate')
#hist.map(plt.hist, 'Review_length', bins=50)

#sns.boxplot(x='Rate', y='Review_length', data=data)

#data['Device'].value_counts().plot(kind='bar')

plt.show()
