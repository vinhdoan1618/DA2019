from selenium import webdriver
import pandas as pd
import time
import timeunit
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# open list name
with open("/Users/admin/DA2019/Data/tgdd_path") as f:
    names = f.read()
list_name = names.split("\n")
# crawler data
driver = webdriver.Chrome()
path = "https://www.thegioididong.com/dtdd/"
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

        wait = WebDriverWait(driver, 10)
        maxpage=driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]/div[2]/div/a[last()-1]""")
        maxpage=int(maxpage.text)
        for page in range(2,maxpage):
                time.sleep(3.5)
                texts = driver.find_elements_by_css_selector("""div.list > ul.ratingLst > li.par > div.rc > p > i""")
                name=driver.find_element_by_xpath("""/html/body/section/div[1]/h1""")
                for cmt in texts:
                        comment.append(cmt.text)
                        device_name.append(name.text)

                list_rate = driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]""")
                items = list_rate.find_elements_by_css_selector("""li.par > div.rc""")
                for item in items:
                        rate = item.find_elements_by_class_name("""iconcom-txtstar""")
                        score.append(len(rate))
                # Save data in dataframe

                scraped_data = pd.DataFrame({'Review': comment, 'Rate': score, 'Device': device_name})
                scraped_data['Review_length'] = scraped_data['Review'].apply(lambda x: len(x) - x.count(' '))

                # export data to csv file
                scraped_data.to_excel('/Users/admin/DA2019/Data/Data_Train_Tgdd.xlsx', encoding='utf-8')

                driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]/div[2]/div/a[text()="%s"]"""%page).click()






