import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import Data
# ------ Load iris -----------
data= pd.read_excel("Data/Data_Processed.xlsx",error_bad_lines=False,encoding='utf-8')
#data =pd.read_excel('3class.xlsx', sheetname='Sheet1')

tf = TfidfVectorizer(min_df=5,max_df= 0.8,max_features=3000,sublinear_tf=True)
tf.fit(data['Review'].values.astype('U'))
X = tf.transform(data['Review'].values.astype('U'))

y_score=(data['Rate'].values).reshape(-1,1)
binaray = Binarizer(threshold=3)
y = binaray.fit_transform(y_score)
y = np.array(y).flatten()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
fig = plt.figure(figsize=(15,20))



#Compare algorithm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

models=[]
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Multinomial Naive Bayes ', MultinomialNB()))
models.append(('Multi-layer Perceptron', MLPClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('K-Nearest Neighbor', KNeighborsClassifier()))


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
# names=[]
# for name, model in models:
#     names.append(name)
#     model.fit(X_train,y_train)
#     y_pred=model.predict(X_test)
#     acc=accuracy_score(y_test,y_pred)
#     print("Độ chính xác {} là {}%".format(name, round(acc*100,2)))

results=[]
names=[]
for name, model in models:
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    auc = roc_auc_score(y_test, probs)
    print('AUC: %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label=1)
    # print('Thresholds:')
    # print(thresholds)
    # print('False Positive Rate:')
    # print(fpr)
    # print('True Positive Rate:')
    # print(tpr)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.show()