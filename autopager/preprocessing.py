#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import sys
import requests
import numpy as np
import parsel
from urllib.parse import urlparse

sys.path.insert(0, '..')
from autopager.htmlutils import get_every_button_and_a
from autopager.model import page_to_features


# In[3]:


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import ssl
WINDOW_SIZE = "1920,1080"
chrome_options = Options()
chrome_options.binary_location = "/usr/bin/google-chrome"
chrome_options.add_argument(f"--window-size={WINDOW_SIZE}")
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
ssl._create_default_https_context = ssl._create_unverified_context


# In[3]:


import time


# In[4]:


DEFAULT_PROJECT_FOLDER = os.path.abspath('..')


# In[5]:


DEFAULT_PREDICT_FOLDER = os.path.abspath('..') + '/predict_folder'


# In[6]:


DEFAULT_MODEL_FOLDER = os.path.abspath('..') + '/models'


# In[7]:


IS_CONTAIN_BUTTON = True


# In[8]:


NB_TO_PY = True


# In[2]:


SCROLL_PAUSE_TIME = 0.5


# In[1]:


def _scrollToButtom(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


# In[4]:


def _get_html_from_selenium(url):
    # 然後將options加入Chrome方法裡面，至於driver請用executable_path宣告進入
    browser=webdriver.Chrome(options=chrome_options)
    browser.implicitly_wait(5)
    browser.set_page_load_timeout(30)
    # 在瀏覽器打上網址連入
    browser.get(url)
    _scrollToButtom(browser)
    time.sleep(SCROLL_PAUSE_TIME)
    html = browser.page_source
    browser.quit()
    return html


# In[10]:


def generate_page_component(url):
    html = _get_html_from_selenium(url)
    url_obj = urlparse(url)
    return {
        "html": html,
        "parseObj": url_obj,
    }


# In[11]:


def get_selectors_from_file(html):
    sel = parsel.Selector(html)
    links = get_every_button_and_a(sel)
    xseq = page_to_features(links)
    return xseq


# In[12]:


if __name__ == '__main__':
    if NB_TO_PY:
        get_ipython().system('jupyter nbconvert --to script preprocessing.ipynb')
    else:
        test_url = "https://kktix.com/events"
        page = generate_page_component(test_url)
        xseq = get_selectors_from_file(page["html"])
        print(xseq[:5])


# In[ ]:




