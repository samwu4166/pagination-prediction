#!/usr/bin/env python
# coding: utf-8

# In[43]:


from laserembeddings import Laser

class LaserSentenceModel:
    def __init__(self, lang = 'en'):
        try:
            self.laser = Laser()
            self.lang = lang
        except Exception as e:
            raise Exception(f"{e}")
    
    def getModel(self):
        return self.laser
    
    def getSentenceVector(self, sents):
        if type(sents) == type([]):
            return self.laser.embed_sentences(sents, lang=self.lang)
        else:
            return self.laser.embed_sentences(sents, lang=self.lang)[0]


# In[10]:


if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script FastTextModel.ipynb')

