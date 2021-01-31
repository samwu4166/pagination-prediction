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
    
    @property
    def test_file(self):
        return self.__test_file
    
    @test_file.setter
    def test_file(self, value):
        if value not in TEST_FILE_MAP:
            print(f"{value} not in the list: {TEST_FILE_MAP.keys()}")
            return
        self.__test_file = value
        
    def get_test_file_list(self):
        print("Test file list: ")
        print(TEST_FILE_MAP)
        
    def get_Xy(self, validate=True, contain_button=True, file_type='T'):
        X, y, scaled_pages = [], [], []
        for row in self.iter_records(contain_button, file_type):
            html = self._load_html(row)
            parser._reset()
            parser.feed(html)
            tag_positions = parser.get_scaled_page()
            selectors = {key: row[key] for key in self.label_map.keys()}
            root = parsel.Selector(html)
            xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
            yseq = [self.label_map.get(_y, _y) for _y in yseq]
            print(f"Finish: Get Page {row['File Name']} (Encoding: {row['Encoding']})records ... (len: {len(yseq)})")
            X.append(xseq)
            y.append(yseq)
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
    def get_test_Xy(self, validate=True, contain_button=True):
        X, y, scaled_pages = [], [], []
        for row in self.iter_test_records():
            html = self._load_test_html(row)
            parser._reset()
            parser.feed(html)
            tag_positions = parser.get_scaled_page()
            selectors = {key: row[key] for key in self.label_map.keys()}
#             print(selectors)
            root = parsel.Selector(html)
            xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
            yseq = [self.label_map.get(_y, _y) for _y in yseq]
            X.append(xseq)
            y.append(yseq)
            scaled_pages.append(tag_positions[:len(xseq)])
        return X, y, scaled_pages

    def iter_records(self, contain_button, file_type):
#         info_path = os.path.join(self.path, 'data_2.csv')
        info_path = os.path.join(self.path, 'data_all.csv')
        with io.open(info_path, encoding='utf8') as f:
            for row in csv.DictReader(f):
                if row['failed']:
                    continue
                if row['Page Type'] == 'button':
                    if contain_button is False:
                        continue
                if row['Checked'] == file_type:
                    yield row
    def iter_test_records(self):
        if self.__test_file is None:
            print("please assign test_file first")
            return
        info_path = os.path.join(self.path, TEST_FILE_MAP[self.__test_file]+'/test_data.csv')
        with io.open(info_path, encoding='utf8') as f:
            for row in csv.DictReader(f):
                if row['Checked'] == 'T':
                    yield row
#         test_csv = pd.read_csv(info_path)
#         test_csv = test_csv.fillna('N/A')
#         for idx, row in test_csv.iterrows():
#             yield row
    def _load_test_html(self, row):
        if self.__test_file is None:
            print("please assign test_file first")
            return
        data_path = os.path.join(self.path, TEST_FILE_MAP[self.__test_file]+'/html')
        path = os.path.join(data_path, str(row['File Name']) + ".html")
        with io.open(path, encoding=row['Encoding']) as f:
            return f.read()

    def _load_html(self, row):
#         data_path = os.path.join(self.path, 'html_2')
        data_path = os.path.join(self.path, 'html_all')
        path = os.path.join(data_path, row['File Name'] + ".html")
        with io.open(path, encoding=row['Encoding']) as f:
            return f.read()
