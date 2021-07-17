#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import re
import sys
from collections import Counter
from itertools import islice
from urllib.parse import urlparse, urlsplit, parse_qs, parse_qsl
import pandas as pd
import numpy as np
import parsel
from sklearn_crfsuite.metrics import flat_classification_report, sequence_accuracy_score

sys.path.insert(0, '..')
from autopager.storage import Storage
from autopager.htmlutils import (get_link_text, get_text_around_selector_list,
                                 get_link_href, get_selector_root)
from autopager.utils import (
    get_domain, normalize_whitespaces, normalize, ngrams, tokenize, ngrams_wb, replace_digits
)
from autopager.model import _num_tokens_feature, _elem_attr
from autopager import AUTOPAGER_LIMITS
from autopager.parserutils import (TagParser, MyHTMLParser, draw_scaled_page, position_check, compare_tag, get_first_tag)
parser = MyHTMLParser()
tagParser = TagParser()


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


# In[ ]:


from tensorflow_addons.layers.crf import CRF
from tensorflow.keras.layers import (Dense, Input, Bidirectional, LSTM, Embedding, Masking, Concatenate,
                                    AveragePooling2D, MaxPooling2D, Reshape, Attention, GlobalAveragePooling1D
                                    , Activation, Conv1D, Conv2D, Flatten, Dropout)


# In[ ]:


from ipywidgets import IntProgress
from IPython.display import display


# In[31]:


import itertools


# In[32]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[14]:


from tensorflow.keras import Model
from tensorflow.data import Dataset


# In[ ]:


from tensorflow.keras import Model
from tensorflow.data import Dataset
import copy


# In[ ]:


import sys
commands = sys.argv
if len(commands) < 3:
    print("python CharCNN.py MODE: Normal|Test GPU_TARGET: 0|1")
    sys.exit(0)
if commands[1].lower() == 'normal':
    train_epoch = 15
    target = 'en'
elif commands[1].lower() == 'test':
    train_epoch = 1
    target = 'ko'
else:
    print("Mode only contains: Normal | Test")
    sys.exit(0)

gpu_target = int(commands[2])

# In[ ]:


print("Mode: ",commands[1].lower())
print("Train_epoch: ",train_epoch)
print("Test_target: ",target)
print("GPU_target :", gpu_target)

# In[ ]:

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus)!=0:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_target], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("No GPUs visible")
# In[14]:


def filter_empty(x, y):
    res_x = [page for page in x if len(x)!= 0]
    res_y = [page for page in y if len(y)!= 0]
    return x, y


# In[ ]:


storage = Storage()


# In[13]:


max_page_seq = 512


# In[ ]:


urls = [rec['Page URL'] for rec in storage.iter_records(language='en',contain_button = True, file_type='T')]
X_raw, y, page_positions = storage.get_Xy(language='en',contain_button = True,  contain_position=True,file_type='T', scaled_page='normal')
print("pages: {}  domains: {}".format(len(urls), len({get_domain(url) for url in urls})))


# In[ ]:


chunks_x, chunks_y, chunk_positions = X_raw, y, page_positions


# In[ ]:


chunks_x, chunks_y = filter_empty(chunks_x, chunks_y)


# In[18]:


from LaserSentenceModel import LaserSentenceModel


# In[19]:


laser = LaserSentenceModel()


# In[22]:


def parseAttribute(html):
    close_index = html.find('>')
    open_text = html[:close_index]
    open_text = open_text.replace('<a ','')
    open_text = open_text.replace('<button','')
    return normalize(open_text)


# In[25]:


def _as_list(generator, limit=None):
    """
    >>> _as_list(ngrams_wb("text", 2, 2), 0)
    []
    >>> _as_list(ngrams_wb("text", 2, 2), 2)
    ['te', 'ex']
    >>> _as_list(ngrams_wb("text", 2, 2))
    ['te', 'ex', 'xt']
    """
    return list(generator if limit is None else islice(generator, 0, limit))

def feat_to_tokens(feat, tokenizer):
    if type(feat) == type([]):
        feat = ' '.join(feat)
    tokens = tokenizer.tokenize(feat)
    return tokens

def num_token_feature_to_class(number):
    if number == '=0':
        return [1, 0, 0, 0]
    elif number == '=1':
        return [0, 1, 0, 0]
    elif number == '=2':
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

def link_to_features(link):
    text = normalize(get_link_text(link))
    href = get_link_href(link)
    if href is None:
        href = ""
    p = urlsplit(href)
    parent = link.xpath('..').extract()
    parent = get_first_tag(parser, parent[0])
    query_parsed = parse_qsl(p.query) #parse query string from path
    query_param_names = [k.lower() for k, v in query_parsed]
    query_param_names_ngrams = _as_list(ngrams_wb(
        " ".join([normalize(name) for name in query_param_names]), 3, 5, True
    ))
    attribute_text = parseAttribute(link.extract())
    # Classes of link itself and all its children.
    # It is common to have e.g. span elements with fontawesome
    # arrow icon classes inside <a> links.
    self_and_children_classes = ' '.join(link.xpath(".//@class").extract())
    parent_classes = ' '.join(link.xpath('../@class').extract())
    css_classes = normalize(parent_classes + ' ' + self_and_children_classes)
    
    token_feature = {
        'text-exact': replace_digits(text.strip()[:100].strip()),
#         'query': query_param_names,
        'query': query_param_names_ngrams,
        'parent-tag': parent,
#         'class': css_classes.split()[:AUTOPAGER_LIMITS.max_css_features],
        'class':_as_list(ngrams_wb(css_classes, 4, 5),
                          AUTOPAGER_LIMITS.max_css_features),
        'text': _as_list(ngrams_wb(replace_digits(text), 2, 5),
                         AUTOPAGER_LIMITS.max_text_features),
        'attribute_text': attribute_text,
    }
    tag_feature = {
        'isdigit': 1 if text.isdigit() is True else 0,
        'isalpha': 1 if text.isalpha() is True else 0,
        'has-href': 0 if href is "" else 1,
        'path-has-page': 1 if 'page' in p.path.lower() else 0,
        'path-has-pageXX': 1 if re.search(r'[/-](?:p|page\w?)/?\d+', p.path.lower()) is not None else 0,
        'path-has-number': 1 if any(part.isdigit() for part in p.path.split('/')) else 0,
        'href-has-year': 1 if re.search('20\d\d', href) is not None else 0,
        'class-has-disabled': 1 if 'disabled' in css_classes else 0,
#         'num-tokens': num_token_feature_to_class(_num_tokens_feature(text)),
    }
    non_token_feature = []
    for k,v in tag_feature.items():
        if type(v) == type([]):
            non_token_feature.extend(v)
        else:
            non_token_feature.append(v)
    return [token_feature, non_token_feature]


def page_to_features(xseq):
    feat_list = [link_to_features(a) for a in xseq]
    around = get_text_around_selector_list(xseq, max_length=15)
    return feat_list

def get_token_tag_features_from_chunks(chunks):
    token_features = []
    tag_features = []
    for idx, page in enumerate(chunks):
        try:
            feat_list = page_to_features(page)
            token_features.append([node[0] for node in feat_list])
            tag_features.append(np.array([node[1] for node in feat_list]))
        except:
            raise Exception(f"Error occured on {idx}")
    return token_features, tag_features

def word_to_vector(word_list, word_vector_method = None):
    if word_vector_method is None:
        print("Need to specified a method.")
        return
    elif word_vector_method == 'FastText':
        if type(word_list) == type([]):
            if len(word_list) == 0:
                return np.zeros(ft.getModel().get_dimension())
            else:
                vectors_array = []
                for word in word_list:
                    vector = ft.getWordVector(word)
                    vectors_array.append(vector)
                mean_vector = np.mean(vectors_array, axis = 0)
                return mean_vector
        else:
            return ft.getWordVector(word_list)
    elif word_vector_method == 'Laser':
        return laser.getSentenceVector(word_list)

def pages_to_word_vector(ft, token_features):
    pages_vector = []
    for page in token_features:
        page_vectors = []
        for node in page:
            classes = word_to_vector(ft, node['class'])
            query = word_to_vector(ft, node['query'])
            p_tag = word_to_vector(ft, node['parent-tag'])
            full_vector = np.concatenate([classes, query, p_tag], axis = 0)
            page_vectors.append(full_vector)
        pages_vector.append(np.array(page_vectors))
    return pages_vector
    
def list_to_dataSet(data, dataType):
    dataset = Dataset.from_generator(lambda: iter(data), dataType)
    return dataset

def zip_dataSet(data):
    data_tuple = tuple(data)
    dataset = Dataset.zip(data_tuple)
    return dataset

def describe_dataset(dataset):
    print(train_dataset.element_spec)
    
def composite_splite_to_train_val(composite_x, y, number):
    x_train = [ data[:-number] for data in composite_x]
    y_train = y[:-number]
    x_val = [ data[-number:] for data in composite_x]
    y_val = y[-number:]
    return x_train, y_train, x_val, y_val

def composite_cut_data(composite_x, y, percent):
    number = round(len(y) * percent)
    new_composite_x = [ data[:number] for data in composite_x]
    new_y = y[:number]
    return new_composite_x, new_y

def data_list_to_dataset(x, y, isValidation = False, batch_size = 1):
    all_data = None
    for data in x:
        dataset = list_to_dataSet(data, tf.float32)
        if all_data == None:
            all_data = dataset
        else:
            all_data = Dataset.zip((all_data, dataset))
    y_ds = list_to_dataSet(y, tf.int32)
    final_set = Dataset.zip((all_data, y_ds))
    if not isValidation:
        final_set = final_set.shuffle(buffer_size=1024).batch(batch_size)
    else:
        final_set = final_set.batch(batch_size)
    return final_set

def composite_list_to_dataset(x, batch_size = 1):
    all_data = None
    for data in x:
        dataset = list_to_dataSet(data, tf.float32)
        if all_data == None:
            all_data = dataset
        else:
            all_data = Dataset.zip((all_data, dataset))
    return all_data.batch(batch_size)


def get_test_attr(tk_train, test_token_features):
    test_attr_pages = [[node['attribute_text'].lower() for node in page] for page in test_token_features]
    test_sequences = [tk_train.texts_to_sequences(page) for page in test_attr_pages]
    test_attr_data = [pad_sequences(test_page, maxlen=256, padding='post') for test_page in test_sequences]
    test_attr_data = [np.array(test_page, dtype='float32') for test_page in test_attr_data]
    return test_attr_data

def prepare_input_ids(page_tokens, max_len):
    pages_class = []
    pages_query = []
    pages_text = []
#     print(len(page_tokens))
    for page in page_tokens:
        class_page = []
        query_page = []
        text_page = []
        for node in page:
            #class
            class_ids = class_tokenizer.tokenize(node['class'])
            class_ids = class_ids + [0] * (max_len-len(class_ids))
            class_page.append(class_ids[:max_len])
            #query
            query_ids = query_tokenizer.tokenize(node['query'])
            query_ids = query_ids + [0] * (max_len-len(query_ids))
            query_page.append(query_ids[:max_len])
            #text
            text_ids = text_tokenizer.tokenize(node['text'])
            text_ids = text_ids + [0] * (max_len-len(text_ids))
            text_page.append(text_ids[:max_len])
        pages_class.append(np.array(class_page))
        pages_query.append(np.array(query_page))
        pages_text.append(np.array(text_page))
    return pages_class, pages_query, pages_text


# In[29]:


token_features, tag_features = get_token_tag_features_from_chunks(chunks_x)
# train_tag_feature_token_list = extract_tokens_from_token_features(token_features)


# In[33]:


fit_attr = []
attr_pages = [[node['attribute_text'].lower() for node in page] for page in token_features]
_ = [fit_attr.extend([node['attribute_text'].lower() for node in page]) for page in token_features]


# In[35]:


tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(fit_attr)


# In[36]:


# -----------------------Skip part start--------------------------
# construct a new vocabulary
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

# Use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy()
# Add 'UNK' to the vocabulary
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
# -----------------------Skip part end----------------------------


# In[37]:


# Convert string to index
train_sequences = [tk.texts_to_sequences(page) for page in attr_pages]


# In[39]:


train_attr_data = [pad_sequences(train_page, maxlen=256, padding='post') for train_page in train_sequences]


# In[40]:


train_attr_data = [np.array(train_page, dtype='float32') for train_page in train_attr_data]


# In[43]:


vocab_size = len(tk.word_index)


# In[44]:


embedding_weights = []
embedding_weights.append(np.zeros(vocab_size))

for char, i in tk.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.array(embedding_weights)


# In[46]:


attr_embedding_size = 69
input_size = 256


# In[47]:


attr_embedding_layer = Embedding(vocab_size+1, 
                                 attr_embedding_size,
                                 input_length=input_size,
                                 weights = [embedding_weights]
                                )


# In[ ]:


token_feature_list = list(token_features[0][0].keys())

def pages_to_word_vector_from_keylist(word_vector_method, token_features, word_to_vec_list):
    print(f"Transform key {word_to_vec_list} to word_vector ... ")
    pages_vector = []
    for idx, page in enumerate(token_features):
        page_vectors = []
        for node in page:
            full_vector_list = []
            for k,v in node.items():
                if k in word_to_vec_list:
                    full_vector_list.append(word_to_vector(v, word_vector_method))
            full_vector = np.concatenate(full_vector_list, axis=0)
            page_vectors.append(full_vector)
        pages_vector.append(np.array(page_vectors))
    print("Finish transforming to word_vector")
    return pages_vector

def sparse_representation_with_map(tag, data_map):
    rt_vec = [0] * len(data_map)
    for idx, map_tag in enumerate(data_map):
        if tag == map_tag[0]:
            rt_vec[idx] = 1
            break
    return rt_vec

def get_ptags_vector(token_features, data_map):
    pages_ptag = []
    for page in token_features:
        ptag_page = []
        for node in page:
            p_tag = node['parent-tag']
            ptag_page.append(sparse_representation_with_map(p_tag, data_map))
        pages_ptag.append(np.array(ptag_page))
    return pages_ptag

top_parent_tags = {}
for page in token_features:
    for node in page:
        p_tag = node['parent-tag']
        if p_tag not in top_parent_tags:
            top_parent_tags[p_tag] = 1
        else:
            top_parent_tags[p_tag] += 1
            
# Create datamap for ptag
sorted_parent_tags = sorted(top_parent_tags.items(),key=lambda x:x[1],reverse=True)
data_map_for_ptag = sorted_parent_tags[:30]



ptags_vector = get_ptags_vector(token_features, data_map_for_ptag)

from collections import OrderedDict

class TagTokenizer:
    def __init__(self, myDict = None):
        rt_dict = {}
        rt_dict['[PAD]'] = 0
        rt_dict['[UNK]'] = 1
        i = 2
        if myDict is not None:
            for k,v in myDict.items():
                rt_dict[k] = i
                i+=1
        self.map = rt_dict
        
    def tokenize(self, word):
        if type(word) == type([]):
            token_list = []
            for _word in word:
                if _word not in self.map:
                    token_list.append(self.map['[UNK]'])
                else:
                    token_list.append(self.map[_word])
            return token_list
        else:
            if word not in self.map:
                return self.map['[UNK]']
            else:
                return self.map[word]
    def get_size(self):
        return len(self.map)

top_thousand_class = {}
top_thousand_query = {}
text_map = {}
for page in token_features:
    for node in page:
        for _class in node['class']:
            if _class in top_thousand_class:
                top_thousand_class[_class]+=1
            else:
                top_thousand_class[_class]=1
        for _query in node['query']:
            if _query in top_thousand_query:
                top_thousand_query[_query]+=1
            else:
                top_thousand_query[_query]=1
        for _text in node['text']:
            if _text not in text_map:
                text_map[_text] = 1

class_tokenizer = TagTokenizer(top_thousand_class)
query_tokenizer = TagTokenizer(top_thousand_query)
text_tokenizer = TagTokenizer(text_map)


ft_full_tokens_emb = np.load('embedding/train/LaserEmb.npy', allow_pickle=True)
train_tag_info_list = tag_features #features which only have tag true/false information


# In[65]:


max_len = 256


# In[66]:


pages_class, pages_query, pages_text = prepare_input_ids(token_features, max_len)


# In[67]:


train_attr_x = ft_full_tokens_emb


# In[68]:


train_ptag = ptags_vector


# In[69]:


train_tag_x = tag_features


# In[70]:


train_tag_x = [ np.concatenate([tag_info,ptags], axis = 1) if len(tag_info)!=0 else np.array([]) for tag_info, ptags in zip(train_tag_x, train_ptag)]


# In[124]:


train_composite_with_token = [train_attr_x, train_attr_data, pages_class, pages_query, train_tag_x]


# In[73]:


labels = ["O", "PREV", "PAGE", "NEXT"]
tag2idx = { label:idx for idx,label in enumerate(labels)}
idx2tag = { idx:label for idx,label in enumerate(labels)}
num_tags = len(labels)


# In[74]:


train_y = [np.array([tag2idx.get(l) for l in lab]) for lab in chunks_y]


# In[125]:


for inputs in train_composite_with_token:
    print(inputs[0].shape)


# ### Model

# In[ ]:


conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]
              ]
dropout_p = 0.25
fully_connected_layers = [512, 512]
ft_shape = (None, 1024)
tag_info_shape = (None, 38)
tag_emb_shape = (None, 256)
HIDDEN_UNITS = 300
embedding_size = 32


NUM_CLASS = num_tags
embbed_output_shape = embedding_size
page_embbed_shape = (-1, embbed_output_shape)
optimizer = keras.optimizers.Adam()


input_ft_embedding = Input(shape=(ft_shape), name="input_ft_embeddings")
input_tag_information = Input(shape=(tag_info_shape), name="input_tag_information")
input_attribute = Input(shape=(tag_emb_shape), name="input_attr")
input_class = Input(shape=(tag_emb_shape), name="input_class")
input_query = Input(shape=(tag_emb_shape), name="input_query")

# Char-CNN Attribute start

attr_emb = Embedding(vocab_size+1,attr_embedding_size,input_length=input_size,weights = [embedding_weights])(input_attribute)
attr_shape = attr_emb.get_shape().as_list()
attr_emb = Reshape((-1, attr_shape[2] * attr_shape[3]))(attr_emb)
attr_emb_merged = Dense(512, activation='relu')(attr_emb)  # 

# Char-CNN Attribute end
ft_FFN = Dense(units = 512, activation = 'relu', name="ft_FFN_01")(input_ft_embedding)
ft_FFN = Dense(units = 256, activation = 'relu', name="ft_FFN_02")(ft_FFN)
ft_FFN = Dense(units = 128, activation = 'relu', name="ft_FFN_out")(ft_FFN)

merged = Concatenate()([ft_FFN, attr_emb_merged, input_tag_information])
model = Bidirectional(LSTM(units = HIDDEN_UNITS//2, return_sequences=True))(merged)

crf=CRF(NUM_CLASS, name='crf_layer')
out =crf(model)
model = Model([input_ft_embedding, input_attribute, input_class, input_query, input_tag_information], out)

loss_fn = crf.get_loss


# In[ ]:


from sklearn.metrics import classification_report
from collections import Counter
def calculate_pages_metric(y_true_pages, y_predict_pages):
    pages_f1 = []
    nexts_f1 = []
    avg_f1 = []
    for y_true, y_predict in zip(y_true_pages, y_predict_pages):
        if len(y_true) == 0:
            break
        report = classification_report(y_true, y_predict,output_dict=True)
#         print(report)
        PAGE = report['2']['f1-score']
        NEXT = report['3']['f1-score']
        pages_f1.append(PAGE)
        nexts_f1.append(NEXT)
        avg_f1.append((PAGE+NEXT)/2)
    return pages_f1, nexts_f1, avg_f1
def calculate_page_metric(y_true, y_predict):    
    report = classification_report(y_true, y_predict,labels=[0,2,3],output_dict=True)
    OTHER = report['0']['f1-score']
    PAGE = report['2']['f1-score']
    NEXT = report['3']['f1-score']
    if 2 in y_true and 3 in y_true:
        AVG = (PAGE+NEXT)/2
    elif 2 in y_true and 3 not in y_true:
        AVG = PAGE
    elif 2 not in y_true and 3 in y_true:
        AVG = NEXT
    else:
        AVG = OTHER
    return AVG


# In[ ]:


def train_on_epoch(epochs, model, optimizer, train_dataset, val_dataset, best_model_method = 'f1-score'):
    import time
    
    epochs = epochs
    best_weights = None
    best_f1_weights = None
    best = np.Inf
    best_loss_history = None
    best_f1 = 0
    best_f1_history = None
    avg_epoch_losses = []
    avg_epoch_f1s = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 50 batches.
#             if step % 50 == 0:
#                 print(
#                     "Training loss (for one batch) at step %d: %.4f"
#                     % (step, float(loss_value))
#                 )
#                 print("Seen so far: %d samples" % ((step + 1) * batch_size))


        # Run a validation loop at the end of each epoch.
        val_losses = []
        val_f1s = []
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            val_avg_f1 = calculate_page_metric(y_batch_val.numpy()[0], val_logits.numpy()[0])
            val_losses.append(val_loss_value)
            val_f1s.append(val_avg_f1)
        average_val_loss = np.average(val_losses)
        average_val_f1 = np.average(val_f1s)
        avg_epoch_losses.append(average_val_loss)
        avg_epoch_f1s.append(average_val_f1)
        if average_val_loss < best:
            best_weights = model.get_weights()
            best = average_val_loss
            best_loss_history = [val_losses, val_f1s]
        if average_val_f1 > best_f1:
            best_f1_weights = model.get_weights()
            best_f1 = average_val_f1
            best_f1_history = [val_losses, val_f1s]
        print("Validation loss: %.4f" % (float(average_val_loss),))
        print("Validation F1: %.4f" % (float(average_val_f1),))
        print("Time taken: %.2fs" % (time.time() - start_time))
    print(f"Best loss: {best}, Best F1: {best_f1}")
    print(f"Training finish, load best weights. {best_model_method}")
    
    if best_model_method == 'loss':
        model.set_weights(best_weights)
    elif best_model_method == 'f1-score':
        model.set_weights(best_f1_weights)
    avg_epoch_result = {"epoch_losses": avg_epoch_losses, "epoch_f1s": avg_epoch_f1s}
    return model, avg_epoch_result


# In[ ]:


def prepare_for_testing(test_X_raw, test_y_raw): #ft-bert -no chunks
    chunks_test_x, chunks_test_y = test_X_raw, test_y_raw
    chunks_test_x, chunks_test_y = filter_empty(chunks_test_x, chunks_test_y)
    test_token_features, test_tag_features = get_token_tag_features_from_chunks(chunks_test_x)
    
    test_ptags_vector = get_ptags_vector(test_token_features, data_map_for_ptag)
    test_ft_emb = pages_to_word_vector_from_keylist('Laser', test_token_features, ['text-exact'])
    test_attr = get_test_attr(tk, test_token_features)
    test_pages_class, test_pages_query, _ = prepare_input_ids(test_token_features, 256)
    test_tag_info_list = test_tag_features

    ## X_test_input
    test_tag_x = [ np.concatenate([tag_info,ptags], axis = 1) if len(tag_info)!=0 else np.array([]) for tag_info, ptags in zip(test_tag_info_list, test_ptags_vector)]
    test_composite_input = [test_ft_emb, test_attr, test_pages_class, test_pages_query, test_tag_x]
    
    ## y_test_input
    y_test = [[tag2idx.get(l) for l in lab] for lab in chunks_test_y]
    y_test = [[idx2tag.get(lab) for lab in page] for page in y_test]
    y_test = np.asarray(y_test)
    
    return test_composite_input, y_test


# In[ ]:


def evaluate_from_batch(model, x, y, evaluate_labels):
    print("Start predicting test data ...")
    test_page_dataset = composite_list_to_dataset(x)
    predicted_y = []
    for pageIdx, batch_x_test in enumerate(test_page_dataset):
        if len(y[pageIdx]) == 0:
            batch_predict_y = np.array([])
        else:
            batch_predict_y = model(batch_x_test)[0].numpy()
        if len(batch_predict_y.shape) != 1:
            tmp = list()
            for lab in batch_predict_y:
                lab = lab.tolist()
                tmp.append(lab.index(max(lab)))
            batch_predict_y = tmp
        predicted_y.append(batch_predict_y)
    print("Start evaluating test data ...")
    predict_y = np.asarray([[idx2tag.get(lab) for lab in page] for page in predicted_y])
    report = flat_classification_report(y, predict_y, labels=evaluate_labels, digits=3,output_dict=True)
    return report


# In[ ]:


def evaluate_model(model, target = "all"):
    TEST_MODEL = model
#     test_languages = storage.get_all_test_languages()
    test_languages = ['en','de','ru','zh','ja','ko']
    if target != "all":
        test_languages = [target]
    reports = {}
    for language in test_languages:
        print("Testing language: ", language)
        test_urls = [rec['Page URL'] for rec in storage.iter_test_records_by_language(language=language)]
        test_X_raw, test_y = storage.get_test_Xy_by_language(language=language)
        print("pages: {}  domains: {}".format(len(test_urls), len({get_domain(url) for url in test_urls})))
        _test_x, _test_y = prepare_for_testing(test_X_raw, test_y)
        report = evaluate_from_batch(TEST_MODEL, _test_x, _test_y, ['PAGE','NEXT'])
        print(pd.DataFrame(report))
        reports[language] = report
        print("===================================")
    return reports

def calculate_macro_avg(reports):
    avg_macro = 0
    for lan, report in reports.items():
        avg_macro+=report['macro avg']['f1-score']
    return avg_macro/len(reports)


# In[ ]:


x_train, y_train, x_val, y_val = composite_splite_to_train_val(train_composite_with_token, train_y, 20)
train_dataset = data_list_to_dataset(x_train, y_train, isValidation=False)
val_dataset = data_list_to_dataset(x_val, y_val, isValidation=True)


# In[ ]:


print("Ready for traininig")


# In[ ]:


model, avg_epoch_result = train_on_epoch(train_epoch, model, optimizer, train_dataset, val_dataset)


# In[ ]:


reports = evaluate_model(model, target)


# In[ ]:


score = calculate_macro_avg(reports)


# In[ ]:


print("Macro F1: ", score)


# In[ ]:


print("================================================================")

