#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow_hub as hub
import numpy as np
import bert
from bert import tokenization
from tensorflow.keras import Model
from tensorflow.data import Dataset
import tensorflow as tf
from ipywidgets import IntProgress
from IPython.display import display

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus)!=0:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("No GPUs visible")

class BertModel:
    def __init__(self, max_seq_length = None):
        if max_seq_length == None:
            print("Need to assign max_seq_length")
            return
        else:
            self.max_seq_length = max_seq_length  
        self.get_multilingual_bert()
        
    def get_multilingual_bert(self):
        max_seq_length = self.max_seq_length
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1",
                                    trainable=False)
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        self.model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_bert_model(self):
        return self.model
    
    def get_masks(self, tokens):
        """Mask for padding"""
        if len(tokens) > self.max_seq_length:
            raise IndexError(f"Token length more than max seq length! {len(tokens)} > {self.max_seq_length}")
        return [1]*len(tokens) + [0] * (self.max_seq_length - len(tokens))

    def get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id += 1
        return segments + [0] * (self.max_seq_length - len(tokens))


    def get_ids(self, tokens):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (self.max_seq_length-len(token_ids))
        return input_ids

    def get_bert_inputs_from_sequences(self, seqs, Token):
        if type(seqs) != type([]):
            print("seqs must be list of seq")
            return
        if Token is False:
            tokens_list = [self.tokenizer.tokenize(seq) for seq in seqs]
        else:
            tokens_list = seqs
        ids = [ np.array(self.get_ids(tokens)) for tokens in tokens_list ]
        masks = [ np.array(self.get_masks(tokens))  for tokens in tokens_list ]
        segments = [ np.array(self.get_segments(tokens))  for tokens in tokens_list ]
        return np.array(ids), np.array(masks), np.array(segments)
    def page_list_to_bert_embedding_list(self, page_list, Token = None):
        '''
        Args:
            page_list - Input pages
            model - pre-trained emb model
            tokenizer - text tokenizer
            max_seq_length - max seq length per node
            Token - Input is Tokenized list or not (raw input data)
        '''
        if Token is None:
            print("Please assign Token argument")
            return
        print(f"Use custom Token: {Token}")
        p = IntProgress(max=len(page_list))
        p.description = '(Init)'
        p.value = 0
        display(p)
        seq_list = []
        for idx, page in enumerate(page_list):
            p.description = f"Task: {idx+1}"
            p.value = idx+1
            page_idx, page_mask, page_seg = self.get_bert_inputs_from_sequences(page, Token)
            pooled_emb, _ = self.model.predict([ page_idx, page_mask, page_seg ])
            seq_list.append(pooled_emb)
        p.description = '(Done)'
        return seq_list


# In[21]:


if __name__ == '__main__':
    bertModel = BertModel(128)
    test_pages = [["hello world","hello world","hello world"],["hello world","hello world"]]
    test_emb = bertModel.page_list_to_bert_embedding_list(test_pages, False)
    print(f"test emb shape: {test_emb[0].shape}")


# In[ ]:




