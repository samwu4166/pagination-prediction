#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from __future__ import absolute_import
from urllib.parse import quote, unquote
from html.parser import HTMLParser
import matplotlib.pyplot as plt
import numpy as np
from parsel import Selector
from sklearn import preprocessing

# In[6]:


class TagParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.start_tags = []
        self.end_tags = []
        self.type = 'TAG'
    def handle_starttag(self, tag, attrs):
        tmp_dict = {}
        tmp_dict['tag'] = tag
        tmp_dict['attrs'] = attrs
        self.start_tags.append((tag, attrs))
    def get_tags(self):
        return self.start_tags, self.end_tags
    def pop_first_and_reset(self):
        tmp = self.start_tags[0]
        self._reset()
        return tmp
    def _reset(self):
        HTMLParser.reset(self)
        self.start_tags = []
        self.end_tags = []


# In[7]:


class MyHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.start_tags = []
        self.wellDownloaded = False
        self.type = 'HTML'
        
    def handle_starttag(self, tag, attrs):
        tmp_dict = {}
        for attr in attrs:
            if attr[0] not in tmp_dict and len(attr[0]) >= 2: #Filter wrong key(use id for minimum key)
                tmp_dict[attr[0]] = attr[1]
        temp_attrs = [(k,v) for k,v in tmp_dict.items()]
        self.start_tags.append((tag, (self.getpos()[0], self.getpos()[1]),temp_attrs))
        
    def handle_endtag(self, tag):
        if tag == 'html':
            self.wellDownloaded = True

    def get_tags(self):
        return self.start_tags, self.end_tags
    
    def get_scaled_page(self, only_train_data = True):
        def PageScaler(x, y):
            x = np.array(x)
            y = np.array(y)
            max_x = x.max()
            max_y = y.max()
            tranTheta = max_y / max_x
#             print(f"max_x: {max_x}")
#             print(f"max_y: {max_y}")
#             print(f"tran: {tranTheta}")
            x_scaler = preprocessing.MinMaxScaler((0, 1))
        #     y_scaler = preprocessing.MinMaxScaler((0, tranTheta))
            y_scaler = preprocessing.MinMaxScaler((0, 1))
            x_scaled = x_scaler.fit_transform(x.reshape(-1, 1))
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
            return x_scaled, y_scaled
        last_positions_y = [data[1][0] for data in self.start_tags]
        last_positions_x = [data[1][1] for data in self.start_tags]
        scaled_last_positions_x, scaled_last_positions_y = PageScaler(last_positions_x, last_positions_y)
        fixed_page = []
        for x, y, tag_info in zip(scaled_last_positions_x.tolist(), scaled_last_positions_y.tolist(), self.start_tags):
            if only_train_data is False:
                fixed_page.append((x[0],y[0]))
            else:
                if tag_info[0] == 'a' or tag_info[0] == 'button':
                    fixed_page.append((x[0],y[0]))
        return fixed_page

    def _reset(self):
        HTMLParser.reset(self)
        self.start_tags = []
        self.end_tags = []


# In[9]:


def draw_scaled_page(page):
    plt.scatter([node[0] for node in page], [node[1] for node in page])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.gca().invert_yaxis()
    
def compare_tag(tag_info, parsed_node):
    if tag_info[0] == parsed_node[0]:
        if len(tag_info[1]) == len(parsed_node[2]):
            if len(tag_info[1]) == 0:
                return True
            for attr_a, attr_b in zip(tag_info[1], parsed_node[2]):
                if attr_a[0] == attr_b[0]:
                    if attr_a[0] == 'href':
#                         if attr_a[1] == attr_b[1].replace(" ","%20") or attr_a[1] == attr_b[1].replace(" ",""):
                        if unquote(attr_a[1]) == unquote(attr_b[1]):
                            return True
                    else:
                        return True
    else:
        return False

def position_check(file, parser, tagParser):
    if parser.type != 'HTML' and tagParser.type != 'TAG':
        print("Must use MyHTMLParser and TagParser for position_check !")
        return
    selector = Selector(text=file)
    x_seq = selector.xpath(".//a|.//button").extract()
    parser._reset()
    parser.feed(file)
    just_a_button = [data for data in parser.start_tags if data[0] == 'a' or data[0] == 'button']
    if len(x_seq) == len(just_a_button):
        return True
    if len(x_seq) > len(just_a_button):
        print(f"Size of x_seq({len(x_seq)}) smaller than just_a_button({len(just_a_button)})!")
        return False
    i = 0
    i = 0
    while i < len(x_seq):
        tagParser.feed(x_seq[i])
        tag_info = tagParser.pop_first_and_reset()
        parsed_node = just_a_button[i]
        if not compare_tag(tag_info, parsed_node):
            return False
        i+=1
    return True


# In[ ]:




