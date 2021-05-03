#!/usr/bin/env python
# coding: utf-8

# In[8]:


import fasttext.util
import tensorflow as tf
fasttext.util.download_model('en', if_exists='ignore')  # English

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

class FastTextModel:
    def __init__(self, dimension = 100):
        fasttext.util.download_model('en', if_exists='ignore')  # English
        self.ft = fasttext.load_model('cc.en.300.bin')
        if dimension != 300:
            fasttext.util.reduce_model(self.ft, 100)
        print("Current dimension: ", self.ft.get_dimension())
    
    def getModel(self):
        return self.ft
    
    def getWordVector(self, word):
        return self.ft.get_word_vector(word)


# In[9]:


if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script FastTextModel.ipynb')


# In[ ]:




