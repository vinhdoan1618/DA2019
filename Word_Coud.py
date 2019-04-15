import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
import wordcloud


df = pd.read_excel('Data/Vocabulary.xlsx', sheetname='Sheet1')
tuvung=df['Vocabulary'].to_list()
count=df['Count'].to_list()
dictionary = dict(zip(tuvung, count))
plt.figure(figsize=(20,10))
word_cloud = wordcloud.WordCloud(max_words=150,background_color ="white",width=2000,height=1000,mode="RGB").fit_words(dictionary)
plt.axis("off")
plt.imshow(word_cloud)
plt.show()

df = pd.read_excel('Data/Data_Trainfinal.xlsx', sheetname='Sheet1')
tuvung=df['Review'].to_string()
plt.figure(figsize=(20,10))
word_cloud = wordcloud.WordCloud(max_words=150,background_color ="white",width=2000,height=1000,mode="RGB").generate(tuvung)
plt.axis("off")
plt.imshow(word_cloud)
plt.show()

