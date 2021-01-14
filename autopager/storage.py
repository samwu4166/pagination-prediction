# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import io
import csv
import parsel
import pandas as pd

from autopager.htmlutils import get_xseq_yseq


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
# ../autopager/data
DEFAULT_LABEL_MAP = {
    'PREV': 'PREV',
    'NEXT': 'NEXT',
    'PAGE': 'PAGE',

    'FIRST': 'PAGE',
    'LAST': 'PAGE',
}

class Storage(object):

    def __init__(self, path=DEFAULT_DATA_PATH, label_map=DEFAULT_LABEL_MAP):
        self.path = path
        self.label_map = label_map
        print(path)

    def get_Xy(self, validate=True, contain_button=True, file_type='T'):
        X, y = [], []
        for row in self.iter_records(contain_button, file_type):
            html = self._load_html(row)
            selectors = {key: row[key] for key in self.label_map.keys()}
            root = parsel.Selector(html)
            xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
            yseq = [self.label_map.get(_y, _y) for _y in yseq]
            print(f"Finish: Get Page {row['File Name']} (Encoding: {row['Encoding']})records ... (len: {len(yseq)})")
            X.append(xseq)
            y.append(yseq)
        return X, y
    
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
        X, y = [], []
        for row in self.iter_test_records():
            html = self._load_test_html(row)
            selectors = {key: row[key] for key in self.label_map.keys()}
#             print(selectors)
            root = parsel.Selector(html)
            xseq, yseq = get_xseq_yseq(root, selectors, validate=validate, contain_button=contain_button)
            yseq = [self.label_map.get(_y, _y) for _y in yseq]
            X.append(xseq)
            y.append(yseq)
        return X, y

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
        info_path = os.path.join(self.path, 'test_data/test_data.csv')
        with io.open(info_path, encoding='utf8') as f:
            for row in csv.DictReader(f):
                yield row
#         test_csv = pd.read_csv(info_path)
#         test_csv = test_csv.fillna('N/A')
#         for idx, row in test_csv.iterrows():
#             yield row
    def _load_test_html(self, row):
        data_path = os.path.join(self.path, 'test_data/html')
        path = os.path.join(data_path, str(row['File Name']) + ".html")
        with io.open(path, encoding=row['Encoding']) as f:
            return f.read()

    def _load_html(self, row):
#         data_path = os.path.join(self.path, 'html_2')
        data_path = os.path.join(self.path, 'html_all')
        path = os.path.join(data_path, row['File Name'] + ".html")
        with io.open(path, encoding=row['Encoding']) as f:
            return f.read()
