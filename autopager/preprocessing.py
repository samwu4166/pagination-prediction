#!/usr/bin/env python
# coding: utf-8

# In[28]:


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


# In[29]:


DEFAULT_PROJECT_FOLDER = os.path.abspath('..')


# In[30]:


DEFAULT_PREDICT_FOLDER = os.path.abspath('..') + '/predict_folder'


# In[31]:


DEFAULT_MODEL_FOLDER = os.path.abspath('..') + '/models'


# In[32]:


IS_CONTAIN_BUTTON = True


# In[39]:


NB_TO_PY = True


# In[42]:


def generate_page_component(url):
    html = requests.get(url).text
    url_obj = urlparse(url)
    return {
        "html": html,
        "parseObj": url_obj,
    }


# In[35]:


def get_selectors_from_file(file):
    sel = parsel.Selector(html)
    links = get_every_button_and_a(sel)
    xseq = page_to_features(links)
    return xseq


# In[41]:


if __name__ == '__main__':
    if NB_TO_PY:
        get_ipython().system('jupyter nbconvert --to script preprocessing.ipynb')
    else:
        test_url = "https://kktix.com/events"
        html = crawl_into_html(test_url)
        xseq = get_selectors_from_file(html)
        print(xseq[:5])


# In[ ]:




