from Sentiment import sentiment
from function import tachtu
import pandas as pd
from openpyxl import Workbook


data=pd.read_excel('New_Data.xlsx', sheetname='Sheet1')

text=[tachtu(data['Review'])]
a=sentiment(text)
ketqua=[]
for i in range(len(text[0])):
   if int(sentiment(text)[i])== 1:
      ketqua.append('Positive')
   else:
      ketqua.append('Negative')
dataoutput=pd.DataFrame({'Review':data['Review'],'Result':ketqua})
dataoutput.to_excel("Data_Predict.xlsx",encoding='utf-8')
