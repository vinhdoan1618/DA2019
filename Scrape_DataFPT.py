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
with open("/Users/admin/DA2019/NameDevice/Name_Device.txt") as f:
    names = f.read()
list_name = names.split("\n")
# crawler data
driver = webdriver.Chrome()
path = "https://fptshop.com.vn/dien-thoai/"
texts = []
i = 0
m=0
comment=[]
score=[]
device_name=[]
for name in range(len(list_name)):
    path_link = path + list_name[name]
    driver.get(path_link)
    actions = webdriver.ActionChains(driver)
    wait = WebDriverWait(driver, 300)
    #load all review .find_element_by_xpath("""//*[@id="totalrateres"]""").text: wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#danh-gia-nhan-xet > div.fs-dttrating > p > span")))
    while driver.find_element_by_xpath("""//*[@id="danh-gia-nhan-xet"]/div[2]/p/span""").find_element_by_xpath(
            """//*[@id="totalrateres"]""").text:
        driver.find_element_by_xpath("""//*[@id="danh-gia-nhan-xet"]/div[2]/p/span""").click()
        time.sleep(0.75)

    name=driver.find_element_by_class_name("fs-dttname")
    texts = driver.find_elements_by_class_name("fs-dttrtxt")
    list_rate = driver.find_element_by_xpath("""//*[@id="listRate"]""")
    items = list_rate.find_elements_by_class_name('fs-dttrateitem')

    for cmt in texts:
        comment.append(cmt.text)
        device_name.append(name.text)

    for item in items[1:]:
            rate = item.find_elements_by_css_selector('div.fs-dttrate span.fs-dttr10')

            score.append(len(rate))


    scraped_data=pd.DataFrame({'Review':comment,'Rate':score,'Device':device_name})
    scraped_data['Review_length']=scraped_data['Review'].apply(lambda x:len(x) - x.count(' '))


    #export data to csv file
    scraped_data.to_excel('/Users/admin/DA2019/Data/Data_Train.', encoding='utf-8')
