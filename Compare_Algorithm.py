import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import Data
data= pd.read_excel("Data/Data_Processed.xlsx",error_bad_lines=False,encoding='utf-8')

tf = TfidfVectorizer(min_df=5,max_df= 0.8,max_features=3000,sublinear_tf=True)
tf.fit(data['Review'].values.astype('U'))
X = tf.transform(data['Review'].values.astype('U'))

y_score=(data['Rate'].values).reshape(-1,1)
binaray = Binarizer(threshold=3)
y = binaray.fit_transform(y_score)
y = np.array(y).flatten()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=7,shuffle=True)
fig = plt.figure(figsize=(15,20))



#Compare algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


models=[]
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Multinomial Naive Bayes ', MultinomialNB()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('K-Nearest Neighbor', KNeighborsClassifier()))
models.append(('Neural network', MLPClassifier()))
models.append(('SVM', SVC()))
# results=[]
# names=[]
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10, random_state=7)
#     cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
#     print(msg)
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
names=[]
for name, model in models:
    names.append(name)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print("Độ chính xác {} là {}%".format(name, round(acc*100,2)))
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='{} (area = {}%)'.format(name,  round(logit_roc_auc*100,2) ))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Visually/{}_ROC'.format(name))
    plt.show()

# results=[]
# names=[]
# for name, model in models:
#     probs = model.predict_proba(X_test)
#     probs = probs[:, 1]
#
#     from sklearn.metrics import roc_auc_score
#     from sklearn.metrics import roc_curve, auc
#     import matplotlib.pyplot as plt
#
#     auc = roc_auc_score(y_test, probs)
#     print('AUC: %.3f' % auc)
#     fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label=1)
#     # print('Thresholds:')
#     # print(thresholds)
#     # print('False Positive Rate:')
#     # print(fpr)
#     # print('True Positive Rate:')
#     # print(tpr)
#
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.plot(fpr, tpr, marker='.')
#     plt.show()