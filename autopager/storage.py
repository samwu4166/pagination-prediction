# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import io
import csv
import parsel
import pandas as pd

from autopager.htmlutils import get_xseq_yseq
from autopager.parserutils import (TagParser, MyHTMLParser, draw_scaled_page, position_check, compare_tag)

#Define parser
tagParser = TagParser()
parser = MyHTMLParser()

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_MULTI_DATA_PATH = os.path.join(DEFAULT_DATA_PATH, "multi_lingual_test/")
_multi_pd = pd.read_excel(DEFAULT_MULTI_DATA_PATH+"mlingual_data.xlsx", None, engine='openpyxl')
language_keys = list(_multi_pd.keys())
# ../autopager/data
DEFAULT_LABEL_MAP = {
    'PREV': 'PREV',
    'NEXT': 'NEXT',
    'PAGE': 'PAGE',

    'FIRST': 'PAGE',
    'LAST': 'PAGE',
}
TEST_FILE_MAP = {
    'NORMAL': 'test_data',
    'EVENT_SOURCE': 'test_yuching',
}

'''
Validate:
If we just use href for training, we need to guarantee that one url is only link to one class
'''


class Storage(object):

    def __init__(self, path=DEFAULT_DATA_PATH, label_map=DEFAULT_LABEL_MAP):
        self.path = path
        self.label_map = label_map
        self.__test_file = None
        print("Current test file: ",language_keys)
    
    @property
    def test_file(self):
        return self.__test_file
    
    @test_file.setter
    def test_file(self, value):
        if value not in TEST_FILE_MAP:
            print(f"{value} not in the list: {TEST_FILE_MAP.keys()}")
            return
        self.__test_file = value
    
    def get_all_test_languages(self):
        return language_keys
    
    def get_test_file_list(self):
        print("Test file list: ")
        print(TEST_FILE_MAP)
        
    def get_Xy(self, language=None, validate=True, contain_button=True, contain_position=False, file_type='T', scaled_page = 'normal', verbose = False):
        X, y, scaled_pages = [], [], []
        if verbose:
            print(f"Contain position: {contain_position}")
        for row in self.iter_records(contain_button, file_type, language):
            html = self._load_html(row)
            parser._reset()
            parser.feed(html)
            selectors = {key: row[key] for key in self.label_map.keys()}
            root = parsel.Selector(html)
            xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
            yseq = [self.label_map.get(_y, _y) for _y in yseq]
            if verbose:
                print(f"Finish: Get Page {row['File Name']} (Encoding: {row['Encoding']})records ... (len: {len(yseq)})")
            X.append(xseq)
            y.append(yseq)
            if contain_position is True:
                tag_positions = parser.get_scaled_page(scaler = scaled_page)
                scaled_pages.append(tag_positions[:len(xseq)])
        return X, y, scaled_pages
    
    def test_selector(self, target, validate=True, contain_button=True, file_type='T'):
        for row in self.iter_records(contain_button, file_type):
            filename, encoding = row['File Name'], row['Encoding']
            if filename == target:
                html = self._load_html(row)
                selectors = {key: row[key] for key in self.label_map.keys()}
                root = parsel.Selector(html)
                xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
                yseq = [self.label_map.get(_y, _y) for _y in yseq]
                print(xseq)
                print(yseq)
        return
    def get_test_Xy(self, validate=True, contain_button=True, contain_position=False, scaled_page = 'normal', exclude_en = None, verbose = False):
        X, y, scaled_pages = [], [], []
        if verbose:
            print(f"Contain position: {contain_position}")
        for row in self.iter_test_records(exclude_en):
            html = self._load_test_html(row)
            parser._reset()
            parser.feed(html)
            selectors = {key: row[key] for key in self.label_map.keys()}
            root = parsel.Selector(html)
            xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
            yseq = [self.label_map.get(_y, _y) for _y in yseq]
            X.append(xseq)
            y.append(yseq)
            if contain_position is True:
                tag_positions = parser.get_scaled_page(scaler = scaled_page)
                scaled_pages.append(tag_positions[:len(xseq)])
        return X, y, scaled_pages
    
    def get_test_Xy_by_language(self, language=None, validate=True, contain_button=True):
        if language is None:
            raise ValueError("language must be specified")
        X, y = [], []
        for row in self.iter_test_records_by_language(language):
            html = self._load_test_html_by_language(row, language)
            parser._reset()
            parser.feed(html)
            selectors = {key: row[key] for key in self.label_map.keys()}
            root = parsel.Selector(html)
            xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
            yseq = [self.label_map.get(_y, _y) for _y in yseq]
            X.append(xseq)
            y.append(yseq)
        return X, y
         
    def iter_records(self, contain_button, file_type, language):
#         info_path = os.path.join(self.path, 'data_2.csv')
        info_path = os.path.join(self.path, 'data_all.csv')
        with io.open(info_path, encoding='utf8') as f:
            for row in csv.DictReader(f):
                if row['failed']:
                    continue
                if row['Page Type'] == 'button':
                    if contain_button is False:
                        continue
                if language == None:
                    if row['Checked'] == file_type:
                        yield row
                else:
                    if row['Checked'] == file_type and row['Language'] == language:
                        yield row
    def iter_test_records(self, exclude_en):
        if self.__test_file is None:
            print("please assign test_file first")
            return
        info_path = os.path.join(self.path, TEST_FILE_MAP[self.__test_file]+'/test_data.csv')
        with io.open(info_path, encoding='utf8') as f:
            for row in csv.DictReader(f):
                if exclude_en == None:
                    if row['Checked'] == 'T':
                        yield row
                else:
                    if row['Checked'] == 'T' and row['Language'] != 'en':
                        yield row
#         test_csv = pd.read_csv(info_path)
#         test_csv = test_csv.fillna('N/A')
#         for idx, row in test_csv.iterrows():
#             yield row
    def iter_test_records_by_language(self, language):
        csv_path = DEFAULT_MULTI_DATA_PATH+"mlingual_data.xlsx"
        target_language_pd = pd.read_excel(csv_path, sheet_name=language, engine='openpyxl')
        data = target_language_pd[target_language_pd['Checked'] == 'T']
        data = data.fillna('N/A')
        data = data[data['Checked']=='T']
        data['File Name'] = data['File Name'].astype(int)
        data['File Name'] = data['File Name'].astype(str)
        for idx,row in data.iterrows():
            yield row
    def _load_test_html(self, row):
        if self.__test_file is None:
            print("please assign test_file first")
            return
        data_path = os.path.join(self.path, TEST_FILE_MAP[self.__test_file]+'/html')
        path = os.path.join(data_path, str(row['File Name']) + ".html")
        with io.open(path, encoding=row['Encoding']) as f:
            return f.read()
    def _load_test_html_by_language(self, row, language):
        data_path = os.path.join(DEFAULT_MULTI_DATA_PATH, language)
        path = os.path.join(data_path, str(row['File Name']) + ".html")
        with io.open(path, encoding=row['Encoding']) as f:
            return f.read()  
    def _load_html(self, row):
#         data_path = os.path.join(self.path, 'html_2')
        data_path = os.path.join(self.path, 'html_all')
        path = os.path.join(data_path, row['File Name'] + ".html")
        with io.open(path, encoding=row['Encoding']) as f:
            return f.read()
