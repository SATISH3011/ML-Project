#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("D:/satish/project/Stress Detection Project Using ML/week 3/stress.csv")
df.head()	


# In[3]:


df.describe()


# In[4]:


df.isnull()


# In[5]:


df.isnull().sum()


# In[20]:


import nltk	
import re
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
stemmer=nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))
def clean(text):
    text=str(text).lower()
    text=re.sub('\[.*?\]',' ',text)
    text=re.sub('https?://\S+/www\.\S+',' ',text)
    text=re.sub('<.*?>+',' ',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n',' ',text)
    text=re.sub('\w*\d\w*',' ',text)
    text=[word for word in text.split('') if word not in stopword]
    text=" ".join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["text"]=df["text"].apply(clean)


# In[12]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
text=" ".join(i for i in df.text)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_colour="white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x=np.array(df["text"])
y=np.array(df["label"])

cv=CountVectorizer()
X=cv.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33,random_state=42)


# In[18]:


from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)


# In[19]:


user=input("enter the text")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)


# In[ ]:




