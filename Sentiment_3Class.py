import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



data =pd.read_excel('/Users/admin/DA2019/Data/3class.xlsx', sheetname='Sheet1')

tf = TfidfVectorizer(min_df=5,max_df= 0.8,max_features=3000,sublinear_tf=True)
tf.fit(data['Review'].values.astype('U'))
X = tf.transform(data['Review'].values.astype('U'))
y=data['Promotion']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10,shuffle=True)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pre = model.predict(X_test)

print(classification_report(y_test,y_pre))

#Get vocab from tf idf
vocabulary=pd.DataFrame(tf.vocabulary_.items(),columns=['Vocabulary','Count'])
vocabulary.to_csv('/Users/admin/DA2019/Data/Vocabulary', encoding='utf-8',index=False)
#Get stop word from tf idf
stop_word=pd.DataFrame(tf.stop_words_)
stop_word.to_csv('/Users/admin/DA2019/Data/Stop_Word', encoding='utf-8')

#Accuracy
def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/y_true.shape[0]
    print(classification_report(y_test, y_pre))
    print('accuracy = ', acc(y_test,y_pre ))

def sentiment3class(text):
    for i in text:
        test=tf.transform(i)
    return model.predict(test)
