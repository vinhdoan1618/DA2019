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



from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pre)
print('Confusion matrix:')
print(cnf_matrix)

normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
print('\nConfusion matrix (with normalizatrion:)')
print(normalized_confusion_matrix)

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
class_names = ['Negative', 'Positve']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Unnormalization confusion matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



probs=model.predict_proba(X_test)
probs = probs[:, 1]

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs,pos_label = 1)

# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

feature_names = tf.get_feature_names()

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
import mglearn
mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names  ,n_top_features=25)
plt.show()

vector= grid.best_estimator_.named_steps['tfidfvectorizer']

sorted_by_idf = np.argsort(vector.idf_)
print("Feature with lowest idf : \n{}".format(feature_names[sorted_by_idf[:100]]))