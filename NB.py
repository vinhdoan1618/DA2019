from pyvi import ViTokenizer
import pandas as pd
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from Function import preprocess
from pyvi import ViTokenizer
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
#Preprocessing data
stop_word = ['bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 'chuyện', 'có', 'có_thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến_nỗi', 'đều', 'điều', 'do', 'đó', 'được', 'dưới', 'gì', 'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một_cách', 'này', 'nên', 'nếu', 'ngay', 'nhiều', 'như', 'nhưng', 'những', 'nơi', 'nữa', 'phải', 'qua', 'ra', 'rằng', 'rằng', 'rất', 'rất', 'rồi', 'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên', 'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'vì', 'việc', 'với', 'vừa', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
data= pd.read_excel("Data/dataZ.xlsx",error_bad_lines=False,encoding='utf-8') #doc data chua tien xu ly data = dataZ.xlsx

comments = []
for text in data['Review'] :
        text=str(text)
        text_lower = text.lower()
        text_token = ViTokenizer.tokenize(text_lower)
        comments.append(text_token)
sentences = []
for comment in comments:
    sent = []
    for word in comment.split(" ") :
            if (word not in stop_word) :
                if ("_" in word) or (word.isalpha() == True):
                    sent.append(word)
    sentences.append(" ".join(sent))

data0=pd.DataFrame({'Review':sentences,'Emotion':data['Emotion']})
#VECTOR HOA
tf = TfidfVectorizer(min_df=5,max_df= 0.8,max_features=3000,sublinear_tf=True)
tf.fit(data0['Review'].values.astype('U'))
X = tf.transform(data['Review'].values.astype('U'))

y=data0['Emotion']

#Chia du lieu thanh 2:  training va testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=7,shuffle=True)

#Cho data train vo mo hinh
model =MultinomialNB()
model.fit(X_train,y_train)
#Cho mo hinh du doan data testing
y_pre = model.predict(X_test)

#Accuracy
def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/y_true.shape[0]
print('accuracy = ', acc(y_test,y_pre ))

#Ham goi model de demo
def NB(text):
    for i in text:
        test=tf.transform(i)
    return model.predict(test)
print(classification_report(y_test,y_pre))





