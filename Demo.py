from Sentiment_2Class import sentiment2class
from Preprocess_DataDemo import preprocess
import pandas as pd
from openpyxl import Workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib



data=pd.read_excel('Data/Data_Demo.xlsx')
dataforword=pd.read_excel('Data/Data_Demo.xlsx')

text=[preprocess(data['Review'])]
dataforwordprocessed=[preprocess(dataforword['Review'])]
datasave=pd.DataFrame({'Review':dataforwordprocessed[0]})
datasave.to_excel("Data/Data_DemoPreprocessed.xlsx",encoding='utf-8')


tf = TfidfVectorizer(min_df=5,max_df= 0.8,max_features=3000,sublinear_tf=True)
tf.fit(data['Review'].values.astype('U'))


ketqua=[]
for i in range(len(text[0])):
   if int(sentiment2class(text)[i])== 1:
      ketqua.append('Positive')
   else:
      ketqua.append('Negative')

dataoutput=pd.DataFrame({'Review':data['Review'],'Result':ketqua})
dataoutput.to_excel("Data/Data_DemoPredicted.xlsx",encoding='utf-8')

import matplotlib.pyplot as plt

dataoutput['Result'].value_counts().plot(kind='bar')
plt.xlabel('Cảm xúc')
plt.ylabel('Số lượng')
plt.title('Biểu đồ cột biểu diển kết quả phân tích ')
plt.show()

rate_group=dataoutput.groupby('Result').size()
colors = ['yellow', 'brown',]
rate_group.plot(kind='pie', figsize=[8,6], autopct='%.2f%%', colors=colors)
plt.title('Biểu đồ tròn thể hiện tỉ lệ kết quả phân tích ')
plt.show()

