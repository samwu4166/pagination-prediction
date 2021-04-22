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

def check_file_wellDownloaded(curr_lang, data, parser):
    for idx,row in data.iterrows():
        index = row['File Name']
        page_selector = row['PREV']
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
        parser.feed(file)
        print(f"Index: {index}, Tag size: {len(parser.start_tags)}, Well Download: {parser.wellDownloaded}")
        parser._reset()

print("Testing language: ", language_keys)
print("-----------------------------------------------------------")
for curr_language in language_keys:
    target_language = 'en'
    html_dir = f"../autopager/data/multi_lingual_test/{target_language}/"
    target_language_pd = pd.read_excel(csv_dir, sheet_name=target_language, engine='openpyxl')
    data = target_language_pd[target_language_pd['Checked'] == 'T']
    data = data.fillna('N/A')
    data = data[data['Checked']=='T']
    data['File Name'] = data['File Name'].astype(int)
    data['File Name'] = data['File Name'].astype(str)
    print(f"====================Start Testing Language [__{curr_language}__]====================")
    check_file_wellDownloaded(curr_language, data, parser)
    
print("====================Finish testing file====================")
print("====================Error List         ====================")
for key, val in ErrorList.items():
    print("language: ", key)
    for idx in val:
        print(idx)