from selenium import webdriver
import pandas as pd
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import ui
from selenium.webdriver.common.by import By


# crawler data
driver = webdriver.Chrome()
path = "https://www.thegioididong.com/dtdd/samsung-galaxy-a10"
texts = []
i = 0
m=0
comment=[]
score=[]
device_name=[]

driver.get(path)

maxpage=driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]/div[2]/div/a[last()-1]""")
maxpage=int(maxpage.text)
for page in range(2,maxpage+1):
                time.sleep(0.3)
                texts = driver.find_elements_by_css_selector("""div.list > ul.ratingLst > li.par > div.rc > p > i""")
                #name=driver.find_element_by_xpath("""/html/body/section/div[1]/h1""")
                for cmt in texts:
                        comment.append(cmt.text)
                        print(cmt.text)
                        #device_name.append(name.text)

                # list_rate = driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]""")
                # items = list_rate.find_elements_by_css_selector("""li.par > div.rc""")
                # for item in items:
                #         rate = item.find_elements_by_class_name("""iconcom-txtstar""")
                #         score.append(len(rate))
                # get length

                scraped_data = pd.DataFrame({'Review': comment})
                #scraped_data['Review_length'] = scraped_data['Review'].apply(lambda x: len(x) - x.count(' '))

                # export data to excel file
                scraped_data.to_excel('Data/Data_Demo.xlsx', encoding='utf-8')

                from selenium.webdriver.common.action_chains import ActionChains

                # element = driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]/div[2]/div/a[text()="%s"]"""%page)
                #
                # actions = ActionChains(driver)
                # actions.move_to_element(element).perform()
                driver.find_element_by_xpath("""//*[@id="comment"]/div[1]/div[3]/div[1]/form/input""").click()
                time.sleep(0.75)

                ui.WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, """//*[@id="boxRatingCmt"]/div[3]/div[2]/div/a[text()="%s"]"""%page))).click()

                #driver.find_element_by_xpath("""//*[@id="boxRatingCmt"]/div[3]/div[2]/div/a[text()="%s"]"""%page).click()





