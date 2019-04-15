from NB import NB
#from Sentiment_3Class import sentiment3class
from Function import preprocess
import pandas as pd
from openpyxl import Workbook
from sklearn.feature_extraction.text import TfidfVectorizer


#Doc data moi chua dc gan nhan
data=pd.read_excel('Data/NBdatanew.xlsx')

#tien xu ly o file  Function.py
text=[preprocess(data['Review'])]
#predict roi luu vo ket qua
ketqua=[]
for i in range(len(text[0])):
    ketqua.append(str(NB(text)[i]))
#cho ra ket qua la data da dc gan nhan
dataoutput=pd.DataFrame({'Review':data['Review'],'Result':ketqua})
dataoutput.to_excel("Data/Data_NB_Predict.xlsx",encoding='utf-8')


