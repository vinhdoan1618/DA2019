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

data= pd.read_excel("Data/Data_Processed.xlsx",error_bad_lines=False,encoding='utf-8')
tf = TfidfVectorizer(min_df=5,max_df= 0.8,max_features=3000,sublinear_tf=True)
tf.fit(data['Review'].values.astype('U'))
X = tf.transform(data['Review'].values.astype('U'))
y_score=(data['Rate'].values).reshape(-1,1)
binaray = Binarizer(threshold=3)
y = binaray.fit_transform(y_score)
y = np.array(y).flatten()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=7,shuffle=True)


model =LogisticRegression()
model.fit(X_train,y_train)
y_pre = model.predict(X_test)

# print(classification_report(y_test,y_pre))

#Get vocab from tf idf
vocabulary=pd.DataFrame(tf.vocabulary_.items(),columns=['Vocabulary','Count'])
vocabulary.to_excel('Data/Vocabulary.xlsx', encoding='utf-8',index=False)
#Get stop word from tf idf
stop_word=pd.DataFrame(tf.stop_words_)
stop_word.to_excel('Data/Stop_Word.xlsx', encoding='utf-8')



#Accuracy
def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/y_true.shape[0]
print('accuracy = ', acc(y_test,y_pre ))


def sentiment2class(text):
    for i in text:
        test=tf.transform(i)
    return model.predict(test)
print(classification_report(y_test,y_pre))
