from Sentiment_2Class import sentiment2class
#from Sentiment_3Class import sentiment3class
from Function import tachtu
import pandas as pd
from openpyxl import Workbook


data=pd.read_excel('Data/New_Data.xlsx')

text=[tachtu(data['Review'])]
#a=sentiment3class(text)
ketqua=[]
for i in range(len(text[0])):
   if int(sentiment2class(text)[i])== 1:
      ketqua.append('Positive')
   else:
      ketqua.append('Negative')

dataoutput=pd.DataFrame({'Review':data['Review'],'Result':ketqua})
dataoutput.to_excel("Data/Data_Predict.xlsx",encoding='utf-8')
