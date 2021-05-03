import pandas as pd
from os import listdir
import sys
from sklearn import preprocessing
import numpy as np
from urllib.parse import quote, unquote
from parsel import Selector
sys.path.insert(0, '..')
from autopager.parserutils import (TagParser, MyHTMLParser, draw_scaled_page, position_check, compare_tag)

csv_dir = f"../autopager/data/multi_lingual_test/mlingual_data.xlsx"
__multi_pd = pd.read_excel(csv_dir, None, engine='openpyxl')
language_keys = list(__multi_pd.keys())

parser = MyHTMLParser()
ErrorList = {}
Testing_Label_list = ['PREV','PAGE','NEXT','FIRST','LAST']

def check_file_wellDownloaded(curr_lang, data, parser):
    for idx,row in data.iterrows():
        index = row['File Name']
        encoding = row['Encoding']
        try:
            f = open(html_dir+index+".html", "r", encoding=encoding)
        except:
            print(f"fExcept on file open for {index}")
            if curr_lang not in ErrorList:
                ErrorList[curr_lang] = []
                ErrorList[curr_lang].append(index)
            else:
                ErrorList[curr_lang].append(index)
            continue
        file = f.read()
        label_stat = {}
        for label in Testing_Label_list:
            page_selector = row[label]
            if page_selector == 'N/A':
                label_stat[label] = 'N/A'
                continue
            else:
                page_selector = page_selector.split(',')
            try:
                data_list = extract_data(file, page_selector)
            except Exception as e:
                label_stat[label] = False
                continue
            if len(data_list)==0:
                label_stat[label] = False
            else:
                label_stat[label] = True
        parser.feed(file)
        print(f"Index: {index}, Tag size: {len(parser.start_tags)}, Well Download: {parser.wellDownloaded}, Label Stat: {label_stat}")
        parser._reset()
        
def extract_data(file, selector_list):
    has_wa = False
    selector = Selector(text=file)
    data_list = []
    for css_selector in selector_list:
        extracted_data = selector.css(css_selector).extract()
        if len(extracted_data) == 0:
            has_wa = True
        data_list.append(extracted_data)
    if has_wa:
        return []
    else:
        return data_list        


print("Testing language: ", language_keys)
print("-----------------------------------------------------------")
for curr_language in language_keys:
#     target_language = 'en'
    html_dir = f"../autopager/data/multi_lingual_test/{curr_language}/"
    target_language_pd = pd.read_excel(csv_dir, sheet_name=curr_language, engine='openpyxl')
    data = target_language_pd[target_language_pd['Checked'] == 'T']
    data = data.fillna('N/A')
    data = data[data['Checked']=='T']
    data['File Name'] = data['File Name'].astype(int)
    data['File Name'] = data['File Name'].astype(str)
    print(f"====================Start Testing Language [__{curr_language}__]====================")
    check_file_wellDownloaded(curr_language, data, parser)
    
print("====================   Finish testing file    ====================")
print("====================Error List on Opening file====================")
for key, val in ErrorList.items():
    print("language: ", key)
    for idx in val:
        print(idx)