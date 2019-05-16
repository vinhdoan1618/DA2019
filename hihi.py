import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
import numpy as np
from sklearn.linear_model import LogisticRegression
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

data= pd.read_excel("Data/Data_TrainProcessed.xlsx",error_bad_lines=False,encoding='utf-8')
y_score=(data['Rate'].values).reshape(-1,1)
binaray = Binarizer(threshold=3)
y = binaray.fit_transform(y_score)
y_train = np.array(y).flatten()

pipe = make_pipeline(TfidfVectorizer(min_df=5,max_df= 0.8,max_features=3000,sublinear_tf=True),
 LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(data['Review'].values.astype('U'), y_train)
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset
X_train = vectorizer.transform(data['Review'].values.astype('U'))
# find maximum value for each of the features over the dataset
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorizer.get_feature_names())
mglearn.tools.visualize_coefficients(
 grid.best_estimator_.named_steps["logisticregression"].coef_,
 feature_names, n_top_features=40)
plt.show()