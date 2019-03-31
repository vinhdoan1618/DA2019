import os
import io
from selenium import webdriver
import pandas as pd
from pyvi import ViTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import time
import timeunit
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# open list name
# with open("/Users/admin/DA2019/NameDevice/Name_Device.txt") as f:
#     names = f.read()
# list_name = names.split("\n")
# crawler data
driver = webdriver.Chrome()
texts = []
i = 0
m=0
comment=[]
score=[]
device_name=[]
# for name in range(len(list_name)):
#     path_link = path + list_name[name]
driver.get("https://www.thegioididong.com/dtdd/samsung-galaxy-a50")
actions = webdriver.ActionChains(driver)
wait = WebDriverWait(driver, 300)
#load all review .find_element_by_xpath("""//*[@id="totalrateres"]""").text: wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#danh-gia-nhan-xet > div.fs-dttrating > p > span")))
# while driver.find_element_by_xpath("""//*[@id="danh-gia-nhan-xet"]/div[2]/p/span""").find_element_by_xpath(
#             """//*[@id="totalrateres"]""").text:
#         driver.find_element_by_xpath("""//*[@id="danh-gia-nhan-xet"]/div[2]/p/span""").click()
#         time.sleep(0.75)
while driver.find_element_by_css_selector(""""#boxRatingCmt > div.list > div.pgrc > div > span""").is_enabled():
    texts = driver.find_elements_by_css_selector("""div.rc > p > i""")
    driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]/div[2]/div/a[5]""").click()

#Crawl review and save to txt: comment\comment_...txt
    for cmt in texts:
        print(cmt.text)
        comment.append(cmt.text)


    # Crawl rate star and save to txt: score\score_...txt


    #Save data in dataframe
# scraped_data=pd.DataFrame({'Review':comment,'Rate':score,'Device':device_name})
# scraped_data['Review_length']=scraped_data['Review'].apply(lambda x:len(x) - x.count(' '))
#
#
#     #export data to csv file
#scraped_data.to_csv('/Users/admin/DA2019/Data/Data_Train.csv', encoding='utf-8')





