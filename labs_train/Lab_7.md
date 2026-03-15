# Lab 7

In this lab, we'll be covering **text clustering** and **topic modeling** - namely, unsupervised learning techniques with text. 

We are thus moving beyond the "bag of words" representation of text. 

Topic modeling can be thought of as a type of "dimensionality reduction" technique.


```python
import pandas as pd
pd.set_option('display.max_colwidth', 200)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import nltk
from nltk.corpus import treebank 
from nltk.tree import Tree
import string
import re
import os
import argparse
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/leondgarse/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /home/leondgarse/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to
    [nltk_data]     /home/leondgarse/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!





    True




```python
!python --version
```

    Python 3.12.3


## Part 1: Load our corpus


```python
del_ch_csv_path = Path(r"combined_cases_del-ch.csv")

df_del_ch = pd.read_csv(del_ch_csv_path)
```


```python
df_del_ch
```


```python
df = df_del_ch[['court.name_abbreviation',
                  "name_abbreviation", 
                  'decision_date', 
                  'court.name', 
                  'casebody.opinions' ]]
```


```python
type(df['casebody.opinions'][0])
```




    str




```python
from ast import literal_eval
df['casebody.opinions'] = df['casebody.opinions'].apply(literal_eval)
```

    /tmp/ipykernel_503111/844464783.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['casebody.opinions'] = df['casebody.opinions'].apply(literal_eval)



```python
df['text'] = df['casebody.opinions'].apply(lambda x: x[0]['text'] if isinstance(x, list) and len(x) > 0 else '')
```

    /tmp/ipykernel_503111/502085088.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['text'] = df['casebody.opinions'].apply(lambda x: x[0]['text'] if isinstance(x, list) and len(x) > 0 else '')



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>court.name_abbreviation</th>
      <th>name_abbreviation</th>
      <th>decision_date</th>
      <th>court.name</th>
      <th>casebody.opinions</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1814-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Ridoely, Chancellor.
The articles • of agreement, signed by the complainant on the 26th of September, together with the deed referred to, furnish the only' guide by which the real contr...</td>
      <td>Ridoely, Chancellor.\nThe articles • of agreement, signed by the complainant on the 26th of September, together with the deed referred to, furnish the only' guide by which the real contract of the...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1815-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Ridgely, Chancellor.
In the examination of David F. Gordon, in this cause, that witness stated, in substance, what I have heretofore noticed, and which amounts nearly to the evidence wh...</td>
      <td>Ridgely, Chancellor.\nIn the examination of David F. Gordon, in this cause, that witness stated, in substance, what I have heretofore noticed, and which amounts nearly to the evidence which, it is...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Del. Ch.</td>
      <td>Tatem v. Gilpin</td>
      <td>1816-06-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Ridgely, Chancellor.
This is a case which comes within the exceptions to the rule, and in principle is the same as the case of Robinson vs. Lord Byron cited in argument. By the overflow...</td>
      <td>Ridgely, Chancellor.\nThis is a case which comes within the exceptions to the rule, and in principle is the same as the case of Robinson vs. Lord Byron cited in argument. By the overflowing of the...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Del. Ch.</td>
      <td>Woolaston v. Mendenhall</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Read,
of counsel for the defendant, doubted whether such was the intention of the law; but supposed it was intended for a class of cases where administrators, executors, and trustees mi...</td>
      <td>Read,\nof counsel for the defendant, doubted whether such was the intention of the law; but supposed it was intended for a class of cases where administrators, executors, and trustees might be ord...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Del. Ch.</td>
      <td>State v. Gilpin</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Bidgely, Chancellor.
It is a general rule that where any motion or petition is made which is not of course, an affidavit of the facts alleged is necessary. 2 Harrison’s Ch. Pr. 1. Accor...</td>
      <td>Bidgely, Chancellor.\nIt is a general rule that where any motion or petition is made which is not of course, an affidavit of the facts alleged is necessary. 2 Harrison’s Ch. Pr. 1. According to th...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Pre-load objects outside the functions
# Using a set() for stopwords makes lookups nearly instantaneous
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Combined Regex: Matches punctuation and digits
# [^\w\s] matches punctuation, \d matches numbers
CLEAN_PATTERN = re.compile(r'[^\w\s]|\d')

def clean_text_fast(text):
    # Convert to lower case immediately to avoid doing it later
    text = text.lower()
    
    # Use the pre-compiled regex to remove punct and digits in one go
    text = CLEAN_PATTERN.sub('', text)
    
    # Split and process in a single list comprehension
    # This combines tokenization, stopword removal, and lemmatization
    tokens = [LEMMATIZER.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    
    return ' '.join(tokens)

# Apply the optimized function
df['clean_text'] = df['text'].apply(clean_text_fast)
```

    /tmp/ipykernel_503111/3919048582.py:26: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['clean_text'] = df['text'].apply(clean_text_fast)



```python
df = df.drop(columns=['casebody.opinions'] , axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>court.name_abbreviation</th>
      <th>name_abbreviation</th>
      <th>decision_date</th>
      <th>court.name</th>
      <th>text</th>
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1814-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridoely, Chancellor.\nThe articles • of agreement, signed by the complainant on the 26th of September, together with the deed referred to, furnish the only' guide by which the real contract of the...</td>
      <td>ridoely chancellor article agreement signed complainant th september together deed referred furnish guide real contract party ascertained intention party must sought complainant counsel rightly ob...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1815-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridgely, Chancellor.\nIn the examination of David F. Gordon, in this cause, that witness stated, in substance, what I have heretofore noticed, and which amounts nearly to the evidence which, it is...</td>
      <td>ridgely chancellor examination david f gordon cause witness stated substance heretofore noticed amount nearly evidence believed complainant could obtained thomas gordon isaac dunning upon reexamin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Del. Ch.</td>
      <td>Tatem v. Gilpin</td>
      <td>1816-06-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridgely, Chancellor.\nThis is a case which comes within the exceptions to the rule, and in principle is the same as the case of Robinson vs. Lord Byron cited in argument. By the overflowing of the...</td>
      <td>ridgely chancellor case come within exception rule principle case robinson v lord byron cited argument overflowing land complainant deprived use enjoyment cannot build covered water may possibly g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Del. Ch.</td>
      <td>Woolaston v. Mendenhall</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Read,\nof counsel for the defendant, doubted whether such was the intention of the law; but supposed it was intended for a class of cases where administrators, executors, and trustees might be ord...</td>
      <td>read counsel defendant doubted whether intention law supposed intended class case administrator executor trustee might ordered sell order effect object trust c chancellor observed power already re...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Del. Ch.</td>
      <td>State v. Gilpin</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Bidgely, Chancellor.\nIt is a general rule that where any motion or petition is made which is not of course, an affidavit of the facts alleged is necessary. 2 Harrison’s Ch. Pr. 1. According to th...</td>
      <td>bidgely chancellor general rule motion petition made course affidavit fact alleged necessary harrison ch pr according english practicean affidavit service subpoena appear necessary attachment go e...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now let's take a look at the same text above after cleaning up. 
df['clean_text'][0]
```

# Part 2: Clustering algorithms

Some reveiw: **Supervised vs. Unsupervised Learning**

1. Supervised Learning (for example, classification)
   
    In Supervised Learning, the model acts like a student with a teacher (supervisor).

    We provide the computer with a **"labeled"** dataset - meaning every piece of data already has the correct answer attached to it.

    The Goal under supervised learning is to learn the relationship between the data and the labels so the model can predict the label of new, unseen data.

    For example -  showing a model 1,000 photos labeled "Cat" or "Dog" so the model can identify them on its own later. In theory, the model learns the "features" which are predictive of the underlying class - namely what makes a cat a cat, and a dog a dog. 

   In practice, this means that we have a column of "classes/labels" used for training - 1 or 0 - depending on the machine learning task that is in question.

2.  Unsupervised Learning (for example, clustering)

    Unsupervised learning occurs when we have data but **no labels**. There is no "teacher" to provide the right answers.

    Instead, the computer looks for inherent patterns, similarities, or structures within the data itself - for example, based on distance between data points. 

    The Goal with clustering is to group data points into "clusters" based on how similar they are to one another.

    For example, in the "cat" vs "dog" example it would be something like, split up similar looking species into clusters. 

    


**TL;DR** 

Classification - that we studied last time - was "supervised" machine learning - meaning that we knew the "true labels" of data beforehand - and these labels were used to create our classification models. 

Clustering is an example of "unsupervised" machine learning - meaning that there is no human telling the computer the "true labels" 

Clustering is useful when we don't know the classes in our data. 

In NLP and data science settings - and in my experience - unsupervised learning is not really as useful as supervised learning. Unspupervized techniques tend to be used a lot to **describe and understand** data - i.e., find patterns in it that are not clear to humans. But it has a lot of limitations which make supervised learning (i.e human labeling) just a better paradigm, which will be discussed below.

With things like topic modeling, unsupervised learning can actually be used as a means of representing text for downstream classification, as we saw in the Aletras paper.



### **K-means Clustering**

K-Means is one of the most well-known clustering algorithms. 

Each data point (in our example, a legal case) has to be a member of a distinct class. 

The "within-class" sum of squares has to be minimized - the alogirthm tries to seperate clusters based on **means/average** of each "cluster" - hence, "k" means. 

Without getting into math, this means is that for each cluster, k-means measure how "far" (euclidian distance) points are from their cluster mean, square it, and sum everything.

The problem with K-means is the classsical problem of "means" - that each cluster is sensetive to outliers in the data. 

It also assumes that the data has roughly "spherical" clusters that are clearly seperable - anything more complex (i.e., most of real world data) requires more complicated methods. The spherical nature of data is a strong assumption because as we explained in the class, text as represented in our document term matrices is both high dimensional and sparse - meaning that data isn't really going to be spherical, so k-means and other types of methods tend to not work that well with text.  

See this image for a vizualization of the of the K-means algorithm.

<div align="center">
  <img src="https://media0.giphy.com/media/12vVAGkaqHUqCQ/giphy.gif?cid=790b76117b7e712fa0d37536e033c289c372b105a4d0447b&rid=giphy.gif&ct=g" alt="iterative nature">
</div>

The algorithm works as follows:

In NLP, we typically cluster documents represented as vectors (e.g., TF-IDF or embeddings). Each document is a point/vector in a high-dimensional space.

K-means aims to partition documents into *k* clusters by minimizing the within-cluster sum of squared Euclidean distances.

1) Initialization: we choose k (the number of clusters) and **randomly** place k starting points (which are known as "centroids").

2) Assignment: For each document vector, compute its Euclidean distance to every centroid. The document is assigned to the cluster whose centroid is closest in Euclidean distance. This partitions the space into Voronoi regions defined by distance to centroids.

3) Update - The center of each cluster is recalculated by taking a new mean of all points assigned to it.

4) Repeat - These steps continue until the centroids stop moving, meaning the algorithm has "converged" - ie the "within cluster sum of squares of euclidean distances" stopped decreasing.

K means is efficient and intuitive, but its reliance on "averages" makes it vulnerable to extreme data points.

For a vizualization of how Voronoi diagrams are computed with the use of Eculidian distance for similarity metrics:

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d9/Voronoi_growth_euclidean.gif?" width="400">

See [this](https://commons.wikimedia.org/wiki/File:Voronoi_growth_euclidean.gif) to explore how different distance metrics affect cluster computation. 

See [this](https://freakonometrics.hypotheses.org/19156) for more on different distance metrics used for similarity. 



```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.01,
                             max_df=.9,  
                             max_features=1000,
                             ngram_range=(1,2))
```


```python
X = vectorizer.fit_transform(df.clean_text)
```


```python
#  Convert the sparse matrix X to a "dense" array
#  Get the word labels from the vectorizer
tfidf_df = pd.DataFrame(X.toarray(), 
                        columns = vectorizer.get_feature_names_out())


# Show the first 5 rows
tfidf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>absence</th>
      <th>absolute</th>
      <th>accept</th>
      <th>accepted</th>
      <th>accordance</th>
      <th>according</th>
      <th>accordingly</th>
      <th>account</th>
      <th>accounting</th>
      <th>acquired</th>
      <th>...</th>
      <th>witness</th>
      <th>word</th>
      <th>work</th>
      <th>would</th>
      <th>would seem</th>
      <th>writing</th>
      <th>written</th>
      <th>year</th>
      <th>yet</th>
      <th>york</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.080556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.041313</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.075830</td>
      <td>0.0</td>
      <td>0.08218</td>
      <td>0.126795</td>
      <td>0.037160</td>
      <td>0.015296</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.164264</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.134003</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.144043</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.012294</td>
      <td>0.000000</td>
      <td>0.036109</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.015762</td>
      <td>0.022072</td>
      <td>0.0</td>
      <td>0.051433</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.009452</td>
      <td>0.023343</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.040260</td>
      <td>0.017557</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.018070</td>
      <td>0.0</td>
      <td>0.021054</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1000 columns</p>
</div>




```python
# create 10 clusters of similar documents
from sklearn.cluster import KMeans

num_clusters = 10

km = KMeans(n_clusters=num_clusters,
           random_state=42) 

km.fit(X)
doc_clusters = km.labels_.tolist() ## clusters stored in km.labels_
```

K-Means assigns a cluster to each document. Each document is represented as a vector of words. 


```python
print(len(km.labels_))
```

    2361



```python
df['cluster'] = doc_clusters
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>court.name_abbreviation</th>
      <th>name_abbreviation</th>
      <th>decision_date</th>
      <th>court.name</th>
      <th>text</th>
      <th>clean_text</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1814-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridoely, Chancellor.\nThe articles • of agreement, signed by the complainant on the 26th of September, together with the deed referred to, furnish the only' guide by which the real contract of the...</td>
      <td>ridoely chancellor article agreement signed complainant th september together deed referred furnish guide real contract party ascertained intention party must sought complainant counsel rightly ob...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1815-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridgely, Chancellor.\nIn the examination of David F. Gordon, in this cause, that witness stated, in substance, what I have heretofore noticed, and which amounts nearly to the evidence which, it is...</td>
      <td>ridgely chancellor examination david f gordon cause witness stated substance heretofore noticed amount nearly evidence believed complainant could obtained thomas gordon isaac dunning upon reexamin...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Del. Ch.</td>
      <td>Tatem v. Gilpin</td>
      <td>1816-06-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridgely, Chancellor.\nThis is a case which comes within the exceptions to the rule, and in principle is the same as the case of Robinson vs. Lord Byron cited in argument. By the overflowing of the...</td>
      <td>ridgely chancellor case come within exception rule principle case robinson v lord byron cited argument overflowing land complainant deprived use enjoyment cannot build covered water may possibly g...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Del. Ch.</td>
      <td>Woolaston v. Mendenhall</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Read,\nof counsel for the defendant, doubted whether such was the intention of the law; but supposed it was intended for a class of cases where administrators, executors, and trustees might be ord...</td>
      <td>read counsel defendant doubted whether intention law supposed intended class case administrator executor trustee might ordered sell order effect object trust c chancellor observed power already re...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Del. Ch.</td>
      <td>State v. Gilpin</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Bidgely, Chancellor.\nIt is a general rule that where any motion or petition is made which is not of course, an affidavit of the facts alleged is necessary. 2 Harrison’s Ch. Pr. 1. According to th...</td>
      <td>bidgely chancellor general rule motion petition made course affidavit fact alleged necessary harrison ch pr according english practicean affidavit service subpoena appear necessary attachment go e...</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



What does each cluster represent?


```python
df[df['cluster'] == 4]['text'][1]
```




    "Ridgely, Chancellor.\nIn the examination of David F. Gordon, in this cause, that witness stated, in substance, what I have heretofore noticed, and which amounts nearly to the evidence which, it is now believed by the complainant, could he obtained from Thomas Gordon and Isaac Dunning, upon a re-examination of the former and upon taking the deposition of the latter.\nIn the consideration of the case, it seems to me that my observations upon the testimony of David F. Gordon apply to the supposed testimony of Thomas Gordon and Isaac Dunning. The whole would be paroi evidence of a supposed contract made before the article of agreement of the' 26th Sept. 1812: and, if that evidence were before me, I should think myself bound by the rules of law to reject it, or rather not to give any effect to it; because it would vary a subsequent written contract made without any fraud on the part of the defendants.\nThe decree must be entered, as heretofore directed, dismissing the bill."



## **Finding the Optimal number of clusters**

As you probably noticed, one of the bigest problems with unsupervised learning - clustering, dimensionality reduction, topic modeling, etc. - is that we don't inherently know how many clusters exist in the data. This parameter is user-defined. 

There are a lot of methods developed to tackle this question.

**Elbow method** 

To help us visualize the "optimal" number of clusters, the Elbow Method is the most common technique. It plots the *Inertia* (Within-Cluster Sum of Squares) against the number of clusters (*k*).

As we increase *k*, the inertia will always decrease because the clusters become smaller and more specific. The "elbow" is the point where the rate of decrease shifts significantly indicating that adding more clusters no longer provides a substantial improvement in describing the data.

**Sillhouete score** 

Sillhouete score shows the quality of clusters - ie are they dense and seperable. It is bounded between +1 and -1.

A high score (closer to 1) occurs when two conditions are met simultaneously:

High Cohesion (similarity): The documents in Cluster A are very similar to each other (the "average distance" **within** the group is **small**) - ie, how similar is the dot to its within cluster neighbors. 

* The question is - "Is this data point similar to its own cluster" (cohesion).

High Separation (dissimilarity): The documents in Cluster A are very different from those in Cluster B (the distance to the next nearest cluster is **large**).

* The question is - "Is this point dissimilar from other clusters?" (separation)

A low score occurs when clusters overlap, which means the number of clusters is incorrect. 

<div align="center">
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/20250623153050051609/clustering.jpg" width="600">



```python
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

inertia = []
sil_scores = []
k_range = range(2, 40)  # Checking kmeans from 2 to 40 clusters

for k in k_range:
    # Fit KMeans
    km = KMeans(n_clusters=k, 
                init='k-means++', # Instead of pure randomness, k-means++ uses a probabilistic approach to spread out the initial centroids before the actual clustering begins
                n_init=10, 
                max_iter=300, 
                random_state=42)
    labels = km.fit_predict(X)
    
    # Store Inertia (for Elbow) and Silhouette Score
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))
```


```python
# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Elbow Plot (Inertia)
ax1.plot(k_range, inertia, marker='o', color='teal')
ax1.set_title('Elbow Method (Inertia)')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Sum of Sqares)')

# Silhouette Plot
ax2.plot(k_range, sil_scores, marker='s', color='coral')
ax2.set_title('Silhouette Score Method')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Avg Silhouette Score')

plt.tight_layout()
plt.show()
```

Text lives in extremely high-dimensional (lots of words as columns), sparse spaces (lots of 0s in the matrix), where Euclidean distance and "means" become less informative — a classic case of so-called "curse of dimensionality". 

The "curse of dimensionality" technically occurs when the number of observations (documents) is less than the number of dimensions (words). For example, 200 documents/rows/observations vs 3000 words/columns/features/dimensions. Genetic data actually has a similar problem (2 humans, millions of genes/features/columns). 

The "curse of dimensionality" essentially means that our general 2-D and 3-D intuitions about the world no longer apply in higher dimensions. Machine learning also struggles in these spaces. For example, distance between points is really not that meaningfull because everything in higher dimensions is really far away from everything else.

<div align="center">
    <img src="https://www.researchgate.net/publication/342638066/figure/fig3/AS:963443473526793@1606714232416/The-effect-of-the-curse-of-dimensionality-when-projected-in-1-one-dimension-2-two.png" width="600">


To quote wikipedia "The common theme of these problems is that when the dimensionality increases, the volume of the space increases so fast that the available data become **sparse**. In order to obtain a reliable result, the amount of data needed often grows exponentially with the dimensionality." 

Which is why clustering tends to not do so well in these settings. It also explains why LLMs and deep learning methods are so data hungry, seemingly requiring infinite amounts of text. 

As we will find out next week, it turns out that with high dimensional data like text, the more useful metric is not **"Euclidian distance"** (or distance generally), but rather **"Cosine Similarity"** - which we will study next week. In short, cosine similarity looks at whether two vectors point in the same direction, rather than looking at the distance between vectors. 




```python
df.shape
```




    (2361, 7)




```python
import numpy as np
from sklearn.metrics import silhouette_score

# Find the best K automatically
# sil_scores was generated from k_range = range(2, 20)
best_index = np.argmax(sil_scores)
opt_num_cluster = k_range[best_index]

print(f'The optimal number of clusters based on Silhouette Score is: {opt_num_cluster}')
print(f'Highest average Silhouette Score: {sil_scores[best_index]:.4f}')

# Re-fit the model with the optimal K
km = KMeans(n_clusters=opt_num_cluster, 
            init='k-means++', 
            n_init=10, 
            random_state=42)
km.fit(X)
```

    The optimal number of clusters based on Silhouette Score is: 36
    Highest average Silhouette Score: 0.0540





<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=36, n_init=10, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>KMeans</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html">?<span>Documentation for KMeans</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_clusters',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_clusters&nbsp;</td>
            <td class="value">36</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('init',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">init&nbsp;</td>
            <td class="value">&#x27;k-means++&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_init',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_init&nbsp;</td>
            <td class="value">10</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_iter',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_iter&nbsp;</td>
            <td class="value">300</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tol',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">tol&nbsp;</td>
            <td class="value">0.0001</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">0</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">random_state&nbsp;</td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy_x',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">copy_x&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('algorithm',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">algorithm&nbsp;</td>
            <td class="value">&#x27;lloyd&#x27;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




```python
# Assign labels to the DataFrame
# Naming it 'cluster_id' is more accurate than 'cluster_mean'
df['cluster_id'] = km.labels_

# View a specific cluster (e.g., Cluster 32)
cluster_1_samples = df[df['cluster_id'] == 32]['clean_text'].head(10)
print("\nSamples from Cluster 1:")
print(cluster_1_samples)
```

    
    Samples from Cluster 1:
    452     chief justice cause complainant preferred stockholder north american cement corporation one defendant constituting preferred stockholder protective committee committee share preferred stock deposi...
    544     chancellor rule preliminary injunction two case heard together disposing rule shall first take notice suit filed journal square bank building company bill journal square bank building company comp...
    550     chancellor consolidation take place general corporation act stockholder constituent company given election whether abide consolidation withdraw therefrom value stock determined appraisal statute p...
    647     appeal decree court chancery appeal court chancery court demurrer sustained bill complaint del ch upon election complainant file amended bill bill dismissed bill filed obtain appointment appraiser...
    773     chancellor merger agreement entered directorate three corporation january three corporation investment trust meeting stockholder defendant called february purpose approving disapproving merger cas...
    873     chancellor complainant brief confine objection alteration right old six dollar cumulative preferred stock phase alteration concerned attempted destruction right paid cash amount unpaid dividend ac...
    993     vicechancellor respondent surviving corporation merger one maryland three delaware corporation merger became effective august year prior merger complainant owned share preferred stock respondent a...
    1034    harrington chancellor demurrer raise two question whether person record owner corporate stock real owner certificate stockholder within meaning section general corporation law rev code whether eve...
    1059    pearson vicechancellor first proceeding amended statute relating payment stock dissatisfied stockholder merging consolidating corporation ultimate question claimant complied provision section beco...
    1065    pearson vicechancellor corporation surviving merger filed petition determination person entitled appraisal payment share preferred stock appointment appraiser pursuant section corporation law cons...
    Name: clean_text, dtype: object



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>court.name_abbreviation</th>
      <th>name_abbreviation</th>
      <th>decision_date</th>
      <th>court.name</th>
      <th>text</th>
      <th>clean_text</th>
      <th>cluster</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1814-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridoely, Chancellor.\nThe articles • of agreement, signed by the complainant on the 26th of September, together with the deed referred to, furnish the only' guide by which the real contract of the...</td>
      <td>ridoely chancellor article agreement signed complainant th september together deed referred furnish guide real contract party ascertained intention party must sought complainant counsel rightly ob...</td>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1815-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridgely, Chancellor.\nIn the examination of David F. Gordon, in this cause, that witness stated, in substance, what I have heretofore noticed, and which amounts nearly to the evidence which, it is...</td>
      <td>ridgely chancellor examination david f gordon cause witness stated substance heretofore noticed amount nearly evidence believed complainant could obtained thomas gordon isaac dunning upon reexamin...</td>
      <td>4</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Del. Ch.</td>
      <td>Tatem v. Gilpin</td>
      <td>1816-06-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Ridgely, Chancellor.\nThis is a case which comes within the exceptions to the rule, and in principle is the same as the case of Robinson vs. Lord Byron cited in argument. By the overflowing of the...</td>
      <td>ridgely chancellor case come within exception rule principle case robinson v lord byron cited argument overflowing land complainant deprived use enjoyment cannot build covered water may possibly g...</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Del. Ch.</td>
      <td>Woolaston v. Mendenhall</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Read,\nof counsel for the defendant, doubted whether such was the intention of the law; but supposed it was intended for a class of cases where administrators, executors, and trustees might be ord...</td>
      <td>read counsel defendant doubted whether intention law supposed intended class case administrator executor trustee might ordered sell order effect object trust c chancellor observed power already re...</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Del. Ch.</td>
      <td>State v. Gilpin</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>Bidgely, Chancellor.\nIt is a general rule that where any motion or petition is made which is not of course, an affidavit of the facts alleged is necessary. 2 Harrison’s Ch. Pr. 1. According to th...</td>
      <td>bidgely chancellor general rule motion petition made course affidavit fact alleged necessary harrison ch pr according english practicean affidavit service subpoena appear necessary attachment go e...</td>
      <td>6</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Let us define a function which returns returns the top n_terms for each cluster based on the TF-IDF centroid.


```python
def get_top_keywords(n_terms):
    # Get the feature names (the actual words) from the TFIDF vectorizer
    terms = vectorizer.get_feature_names_out()
    
    # Get the coordinates of the cluster centers
    # Shape: (n_clusters, n_features)
    centroids = km.cluster_centers_
    
    for i in range(km.n_clusters):
        print(f"Cluster {i}: ", end="")
        
        # Sort the terms in the centroid by their TF-IDF score (descending)
        # centroids[i].argsort() gives indices of scores from smallest to largest
        # [::-1] reverses it to get the highest scores first
        top_indices = centroids[i].argsort()[::-1][:n_terms]
        
        # Print the words corresponding to those top indices
        top_terms = [terms[ind] for ind in top_indices]
        print(", ".join(top_terms))

# Run it to see the top 10 words for your clusters
get_top_keywords(10)
```

    Cluster 0: state, tax, act, school, law, section, statute, public, delaware, plaintiff
    Cluster 1: decree, appeal, opinion, complainant, said, respondent, upon, petitioner, party, petition
    Cluster 2: value, asset, appraiser, stock, stockholder, share, corporation, earnings, plaintiff, market
    Cluster 3: complainant, bill, defendant, demurrer, suit, answer, said, equity, upon, right
    Cluster 4: legacy, testator, estate, codicil, real estate, residuary, land, executor, testatrix, real
    Cluster 5: option, plan, stock, plaintiff, stockholder, corporation, director, defendant, share, board
    Cluster 6: corporation, stock, stockholder, director, company, defendant, asset, share, corporate, receiver
    Cluster 7: plaintiff, defendant, action, motion, complaint, corporation, order, ad, rule, stockholder
    Cluster 8: estate, executor, administrator, deceased, real estate, said, account, sale, decedent, personal
    Cluster 9: plaintiff, agreement, defendant, judgment, party, child, paragraph, issue, would, contract
    Cluster 10: city, street, ordinance, public, council, town, department, power, act, said
    Cluster 11: testator, estate, trust, death, income, gift, life, trustee, child, widow
    Cluster 12: deed, church, complainant, title, land, conveyance, grantor, property, defendant, trustee
    Cluster 13: surety, bond, judgment, debt, principal, equity, said, creditor, complainant, assignment
    Cluster 14: levy court, levy, county, commission, castle, new castle, plaintiff, castle county, hearing, land
    Cluster 15: dividend, stock, preferred, preferred stock, share, stockholder, amendment, corporation, common, common stock
    Cluster 16: plaintiff, road, land, defendant, use, foot, lot, water, public, street
    Cluster 17: receiver, corporation, company, creditor, appointment, asset, bill, lien, appointed, claim
    Cluster 18: lease, rent, land, defendant, plaintiff, tenant, complainant, term, premise, said
    Cluster 19: partnership, partner, defendant, business, complainant, plaintiff, surviving, account, agreement, asset
    Cluster 20: dower, widow, husband, estate, land, election, real estate, right, intestate, real
    Cluster 21: trust, trustee, income, estate, beneficiary, power, fund, interest, property, mr
    Cluster 22: child, mother, issue, testator, death, heir, estate, trust, said, father
    Cluster 23: creditor, judgment, debt, lien, complainant, debtor, execution, land, sale, equity
    Cluster 24: meeting, director, stockholder, bylaw, election, proxy, corporation, vote, board, stock
    Cluster 25: contract, complainant, defendant, performance, agreement, bill, specific performance, party, specific, right
    Cluster 26: settlement, fee, fund, plaintiff, stockholder, counsel, compensation, management, director, action
    Cluster 27: mortgage, lien, bond, debt, complainant, land, sale, property, said, upon
    Cluster 28: stock, share, corporation, company, certificate, stockholder, director, defendant, agreement, issued
    Cluster 29: lot, restriction, deed, defendant, plaintiff, covenant, land, building, development, foot
    Cluster 30: witness, testimony, evidence, party, bill, affidavit, complainant, examination, commission, cause
    Cluster 31: plaintiff, defendant, agreement, contract, action, business, party, would, sale, property
    Cluster 32: merger, corporation, stockholder, appraisal, stock, share, objection, statute, appraiser, demand
    Cluster 33: husband, wife, estate, marriage, property, tenant, land, law, defendant, right
    Cluster 34: testatrix, trust, item, estate, death, residuary, codicil, life, gift, trustee
    Cluster 35: voting, voting trust, agreement, trust, stock, trustee, corporation, stockholder, certificate, vote


As you can see, there are many similar "clusters".

By the way, because of the random nature of k-means, everybody might get different topics. 


```python
df[df['cluster_id'] == 35]['text'][606]
```

### **K-Medoids**

K-medoid is a more robust cousin to K-means. The difference is instead of means of dots, we're working with centroids based on "medians" - which are less sensetive to outliers. 

Since median values in any distrubution have to be actual observations in the data, K-medoid allows us to pick an actual document which is, in a way most "representative" of a cluster. This is not possible with an "average" document - because such document doesn't exist. 


<div align="center">
    <img src="https://www.researchgate.net/publication/342871651/figure/fig1/AS:912165510864897@1594488613267/The-graphical-representation-of-the-difference-between-the-k-means-and-k-medoids.png" width="600">




```python
!pip install scikit-learn-extra 
```

    Requirement already satisfied: scikit-learn-extra in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (0.3.0)
    Requirement already satisfied: numpy>=1.13.3 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from scikit-learn-extra) (1.26.4)
    Requirement already satisfied: scipy>=0.19.1 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from scikit-learn-extra) (1.13.1)
    Requirement already satisfied: scikit-learn>=0.23.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from scikit-learn-extra) (1.7.2)
    Requirement already satisfied: joblib>=1.2.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from scikit-learn>=0.23.0->scikit-learn-extra) (1.3.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from scikit-learn>=0.23.0->scikit-learn-extra) (3.6.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m26.0[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
## if you are on a windows, scikit-learn-extra might not install
## suggestion - do this lab in 3.11 and install scikit-learn-extra - it works with 3.11
```


```python
from sklearn_extra.cluster import KMedoids
```


```python
kmed_sil_scores = []
kmed_inertia = []
k_range = range(2, 40)  


# KMedoids doesn't use 'k-means++', it uses 'build' by default which is very effective
for k in k_range:
    kmed = KMedoids(
        n_clusters=k,
        metric='cosine', ## we use cosine metrics rather than euclidian for k_med
        init='build',
        random_state=42
    )
    labels = kmed.fit_predict(X)
    kmed_inertia.append(kmed.inertia_)
    kmed_sil_scores.append(silhouette_score(X, labels, metric='cosine'))
```


```python
# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Elbow-like plot (total dissimilarity)
ax1.plot(k_range, kmed_inertia, marker='o', color='teal')
ax1.set_title('K-Medoids Elbow Method (Total Dissimilarity)')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Total Within-Cluster Dissimilarity')

# Silhouette score plot
ax2.plot(k_range, kmed_sil_scores, marker='s', color='coral')
ax2.set_title('K-Medoids Silhouette Score Method')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Average Silhouette Score')

plt.tight_layout()
plt.show()
```


```python
# Find Best K
best_index = np.argmax(kmed_sil_scores)

opt_num_cluster = k_range[best_index]

print(f'Optimal K-Medoids clusters: {opt_num_cluster}')

# Re-fit
kmed_final = KMedoids(n_clusters=opt_num_cluster, 
                      metric='cosine', 
                      init='build', 
                      random_state=42)

kmed_final.fit(X)

```


```python
df['cluster_med'] = kmed_final.labels_
df.head()
```

The benefit of using this algorithm is that we get actual "median" documents at the center of a cluster - ie "representative documents"


```python
def get_kmedoid_keywords(n_terms):
    terms = vectorizer.get_feature_names_out()
    medoids = kmed_final.cluster_centers_   # These are actual medoid documents
    
    for i in range(kmed_final.n_clusters):
        # Convert sparse row to dense
        dense_vector = medoids[i].toarray().flatten()
        
        top_indices = dense_vector.argsort()[::-1][:n_terms]
        top_terms = [terms[ind] for ind in top_indices]
        
        print(f"Cluster {i} Keywords: {', '.join(top_terms)}")
```


```python
get_top_keywords(10)
```


```python
# Get the "Representative" Document for each cluster
medoid_indices = kmed_final.medoid_indices_

print("Representative Document for each Cluster")
for cluster_idx, doc_idx in enumerate(medoid_indices):
    print(f"Cluster {cluster_idx} Exemplar: {df.iloc[doc_idx]['text'][:1000]}...")
```


```python
# Build DataFrame
# Get document counts per cluster as a Series
cluster_counts = df.groupby("cluster_med").size()

representative_docs = pd.DataFrame({
    "cluster_id": range(len(medoid_indices)), # Cluster IDs 0,1,2,...
    "medoid_index": medoid_indices,
    "medoid_text": df.iloc[medoid_indices]["text"].values,
    "doc_count": [cluster_counts[c] for c in range(len(medoid_indices))]
})
```


```python
representative_docs
```


```python
total_docs = representative_docs["doc_count"].sum()
print(total_docs)
```

You can see that all documents belong to a cluster - which is a problem because maybe some documents are just "noise" and don't really belong to a cluster?

### **DBSCAN**

DBSCAN  is a popular, very commonly used and relatively recent clustering algorithm (first published in 1996).

DBSCAN stands for __"Density-Based Spatial Clustering of Applications with Noise"__. As the name suggests, DBSCAN uses "density" measure for points, rather than simple means or centroids. "Density" in this case can be thought of points which are "densely packed together". 


<div align="center">
    <img src="https://ml-explained.com/articles/dbscan-explained/dbscan.gif" width="600">



DBSCAN is very useful for non-circle/globular data as can be seen below, and can pretty much be applied to clusters of any shape. DBScan does not assume spherical data representation, which is again useful for clustering, unlike K-means.

<div align="center">
    <img src="https://miro.medium.com/max/1400/1*rfi9uHjGPdNgXgxe9xWvVw.png" width="600">


There are also different determinations of what means density - as anything outside the "cluster" can be deemed to be noise (or outlier) depending on the parametarization.


Some relevant limiations are that just like K-means clustering, DBScan (like all unsupervised models) doesn't have a knowledge of how many clusters there are in the data. Furthermore, sparse vectors and high dimensional data still pose problems - which seems to be a universal problem. 

<div align="center">
    <img src="https://dashee87.github.io/images/DBSCAN_search.gif?" width="600">
<div>



As can be seen in the .gif image above, the algorithm has two key hyperparameters:

* `eps` or epsilon, meaning "radius of neighborhood" or "the maximum distance between two points to consider them **neighbors**" 

`eps` can be thought of as drawing a circle (or hypersphere in higher dimensions) around each point.

Setting epsilon to a small value can lead to many smaller clusters. 

Setting epsilon to a higher value leads to larger clusters. 

* `min_samples` meaning "minimum points to form a cluster"

A `"core point"` (similar to a centroid) is defined by the algorithm as a point with at least `min_samples` neighbors within distance `eps`.

Intuitively, we draw a circle of radius `eps` around each dot. 

If a dot’s circle contains enough other dots (`min_samples`), it’s a core point.

Circles that overlap get merged into clusters.

Lonely dots outside any circle are `noise` or outliers, that don't belong to any clusters. 




```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.90,  ## if you set high, you'll get larger clusters (less clusters) 
                            ## if you set to low, you'll get more clusters 
                min_samples=5)
dbscan.fit(X)
db_clusters = dbscan.labels_
```


```python
df['cluster_db'] = db_clusters
df.head()
```


```python
df['cluster_db'].unique()
```

What does __-1__ cluster mean? 

Check  the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) - __"Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1."__


```python
df[df['cluster_db']==1]['text'] ## cluster one seems to be about land, which is very good!
```


```python
df["text"][7]
```


```python
from sklearn.metrics import pairwise_distances_argmin_min

def get_dbscan_summary(dbscan_model, tfidf_matrix, vectorizer, df, n_terms=10):
    labels = dbscan_model.labels_
    terms = vectorizer.get_feature_names_out()
    
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    results = []

    for cluster_id in unique_clusters:
        indices = np.where(labels == cluster_id)[0]
        cluster_vectors = tfidf_matrix[indices]

        # Calculate the Centroid
        centroid = np.asarray(cluster_vectors.mean(axis=0)).reshape(1, -1)
        
        # Extract Keywords
        dense_centroid = centroid.flatten()
        top_indices = dense_centroid.argsort()[::-1][:n_terms]
        top_terms = [terms[ind] for ind in top_indices]

        # Find the Exemplar
        # closest_idx_in_cluster is the index relative to 'cluster_vectors'
        closest_idx_in_cluster, _ = pairwise_distances_argmin_min(centroid, cluster_vectors)
        
        # Map back to the original DataFrame index
        global_doc_idx = indices[closest_idx_in_cluster[0]]
        # If your df has a non-integer index or a specific named index:
        original_df_index = df.index[global_doc_idx]

        results.append({
            "cluster_id": cluster_id,
            "doc_count": len(indices),
            "keywords": ", ".join(top_terms),
            "exemplar_text": df.iloc[global_doc_idx]['text'][:2000],
            "exemplar_index": original_df_index  # <--- Added this
        })

    return pd.DataFrame(results)
```


```python
# Execute the function we defined above 
representative_docs = get_dbscan_summary(dbscan, X, vectorizer, df)
```


```python
representative_docs
```


```python
df.iloc[26]["text"]
```

### **Hierarchical DBSCAN**
A modified version of DBScan, it automatically chooses `epsilon`, performing DBSCAN over various epsilon values - and returns the result that gives the best stability over epsilon. For reference see [here](https://github.com/scikit-learn-contrib/hdbscan/).


```python
!pip install hdbscan
#!pip install --upgrade numpy
```


```python
from hdbscan import HDBSCAN

hdbscan = HDBSCAN(min_cluster_size=5)
hdbscan.fit(X)
hdb_clusters = hdbscan.labels_
```


```python
df['cluster_hdb'] = hdb_clusters
df.head()
```


```python
df['cluster_hdb'].unique()
```


```python
df[df['cluster_hdb']==1]['clean_text']
```


```python
from sklearn.metrics import pairwise_distances_argmin_min

def get_hdbscan_summary(hdbscan_model, tfidf_matrix, vectorizer, df, n_terms=10):
    labels = hdbscan_model.labels_
    # Probabilities tell us how strongly a point belongs to its assigned cluster
    probs = hdbscan_model.probabilities_ 
    terms = vectorizer.get_feature_names_out()
    
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    results = []

    for cluster_id in unique_clusters:
        # 1. Get indices of all points in this cluster
        indices = np.where(labels == cluster_id)[0]
        cluster_vectors = tfidf_matrix[indices]
        
        # 2. Calculate the Centroid (Mean Vector) for keywords
        centroid = np.asarray(cluster_vectors.mean(axis=0)).reshape(1, -1)
        dense_centroid = centroid.flatten()
        top_indices = dense_centroid.argsort()[::-1][:n_terms]
        top_terms = [terms[ind] for ind in top_indices]

        # 3. Find the Exemplar using HDBSCAN Probabilities
        # We find the index within 'indices' where the probability is highest
        cluster_probs = probs[indices]
        relative_exemplar_idx = np.argmax(cluster_probs)
        global_doc_idx = indices[relative_exemplar_idx]
        
        # Get the original DataFrame index label
        original_df_index = df.index[global_doc_idx]

        results.append({
            "cluster_id": cluster_id,
            "doc_count": len(indices),
            "exemplar_index": original_df_index,
            "max_probability": cluster_probs[relative_exemplar_idx].round(4),
            "keywords": ", ".join(top_terms),
            "exemplar_text": df.iloc[global_doc_idx]['text'][:2000]
        })

    return pd.DataFrame(results)
```


```python
hdb_representative_docs = get_hdbscan_summary(hdbscan, X, vectorizer, df)
```


```python
hdb_representative_docs
```

### **Hierarchical (Agglomerative) Clustering**
For Agglomerative clustering, each point starts as a cluster, and is combined based on distance to other points. At the end - you get "optimal" number of clusters (depending on where you define the cutoff).

<div align="center">
    <img src="https://dashee87.github.io/images/hierarch.gif" width="700">

It's essentially a "bottom-up" approach. 

* We start by having every single point as a cluster
* We calculate the distance between clusters
* We merge the two clusters that are "closest" to each other - where the  `"linkage"` parameter comes - determining how point sare merged
* We keep merging until only one giant cluster containing everything remains.

The dendogram above shows every merge that happened. 

Unlike other algorithms, Agglomerative Clustering doesn't tell us how many clusters we have; we decide this by "cutting" the tree horizontally.

If we cut the tree near the bottom, we get many small and similar clusters. 

If we cut the tree near the top, we get larger broader clusters. 

here's an explanation of the different linkage parameters:
<div align="center">
    <img src="https://dashee87.github.io/images/hierarch_1.gif" width="700">

Ward's method is the most difficult to understand - but essentially, it asks if two clusters are merged, does this reduce the variance of the data. 

It chooses to merge the two clusters that result in the **smallest increase** in the total within-cluster sum of squares (ESS). It is essentially trying to keep clusters as tight and "neat" as possible.


```python
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=opt_num_cluster, 
                                  metric='euclidean', 
                                  linkage='ward') # ward is considered the best method for heirarchical clustering 

cluster.fit_predict(X.toarray())

clusters = cluster.labels_
```


```python
df['cluster_hie'] = clusters
df.head()
```


```python
df['cluster_hie'].unique()
```


```python
df[df['cluster_hie']==1]['clean_text']
```


```python
def get_agglomerative_summary(agglom_model, tfidf_matrix, vectorizer, df, n_terms=10):
    labels = agglom_model.labels_
    terms = vectorizer.get_feature_names_out()
    
    # Agglomerative clustering assigns every point a label (no -1, unlike DBscan)
    unique_clusters = sorted(set(labels))
    results = []

    for cluster_id in unique_clusters:
        # Isolate the cluster data
        indices = np.where(labels == cluster_id)[0]
        cluster_vectors = tfidf_matrix[indices]

        # Calculate the Centroid (Thematic Center)
        # Note: Ward linkage works on Euclidean distance, so the Mean is the natural center
        centroid = np.asarray(cluster_vectors.mean(axis=0)).reshape(1, -1)
        
        #  Extract Keywords from the centroid
        dense_centroid = centroid.flatten()
        top_indices = dense_centroid.argsort()[::-1][:n_terms]
        top_terms = [terms[ind] for ind in top_indices]

        #  Find the Exemplar (Document closest to the mean)
        # We compare the centroid against the specific cluster vectors
        closest_idx_in_cluster, _ = pairwise_distances_argmin_min(centroid, cluster_vectors)
        global_doc_idx = indices[closest_idx_in_cluster[0]]
        
        # Map to original DataFrame index
        original_df_index = df.index[global_doc_idx]

        results.append({
            "cluster_id": cluster_id,
            "doc_count": len(indices),
            "exemplar_index": original_df_index,
            "keywords": ", ".join(top_terms),
            "exemplar_text": df.iloc[global_doc_idx]['text'][:2000]
        })

    return pd.DataFrame(results)
```


```python
hie_representative_docs = get_agglomerative_summary(cluster, 
                                                    X, vectorizer, df)
```


```python
hie_representative_docs
```

Summary - clustering is useful when we want to study how documents in our corpus are related. A set of documents could be a member of "Cluster 1" or "Cluster 2" - this can be useful.

However, because each document __has__ to be a part of a single cluster (or noise) - this makes the determination of clusters not that intuitive when it comes to text. This is known as  "hard clustering" ie that each document is exclusively belonging to a single cluster - basically an "all or nothing" approach to data. 

The issue is that text doesn't really have super clear clusters. Some documents can be about multiple things - the document could be 40% about share prices and 60% about taxes for example. 

We can address this problem via topic modeling.

# Part 3: Topic Modeling

Humans don't really think in terms of data points or "similarity measures." We mostly think of things in terms of parts-based approaches - your face is not a bunch of pixels, but can be thought of as a combination of "the nose (cluster)", "eyes (cluster)," etc. 



## Latent Dirichlet Allocation (LDA) 

Latent Dirilect Allocation, or LDA, is an approach to model the distribution of words that appear in a body of text.

__"Latent"__ means hidden. We can see the words - but nobody knows the exact topics. LDA uncovers these hidden structures by looking at which words frequently appear together across different documents.

__"Dirichlet"__ means [Dirichlet probability disturbiton](https://en.wikipedia.org/wiki/Dirichlet_distribution)  used for this algorithm. This is all mathy stuff -  but intuitively, it’s a statistical constraint. It forces documents to be about only a few topics and topics to be made of only a few key words. Without this, the math would just turn every document into a messy soup of every single topic.

__"Allocation"__ means allocating words to topics.

Basically, one can think of the LDA algorithm as just allocation of words to topics based on the Dirichlet distribution. 


<div align="center">
    <img src="https://agdal1125.github.io/assets/images/lda_diagram.png" width="800"> 

Unlike standard clustering, LDA is a **Generative Model**. It assumes that every document was created by a specific data genearating process:

* The Topic Distribution: An author first chooses a "blend" of topics (e.g., 60% Coropration, 40% Taxes).

* The Word Distribution: For each topic, there is a list of words with specific probabilities of appearing.

* There is an assumption on the data - that "words" (data points) tend to appear together when certain "topics" are discussed. Thus, each topic "generates" a probability on the "words" that appear in it.

For example, the words "Hogwarts" (0.9 probability), "magic" (0.8 probability), "wizardry" (0.7 probability) appear in a topic on "Harry Potter". 

If you think about it carefully, everything in our lives can be thought of as a "topic". 






```python
from IPython.display import display, HTML, IFrame

# Create the IFrame
iframe = IFrame("https://www.youtube.com/embed/MqPKguO5hDA", width=800, height=450)

# centering
display(HTML(f"""
<div style="text-align: center;">
    {iframe._repr_html_()}
</div>
"""))
```


```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
```


```python
df.head()
```


```python
#  df['clean_text'] has our documents
documents = df['clean_text'].tolist()
```


```python
#  Convert documents to a Document-Term Matrix
vectorizer = CountVectorizer(max_df=0.95, 
                             min_df=2)

X_matrix = vectorizer.fit_transform(documents)
```


```python
#  Fit LDA
n_topics = 20  # Number of topics

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10,                 # Maximum number of iterations over the entire corpus during training
    learning_method='online',    # 'online' updates incrementally (good for large datasets); 'batch' updates after seeing all documents
    learning_offset=50.,         # A smoothing parameter to slow down early learning in 'online' mode; larger = more stable updates initially 
    random_state=0
)

lda.fit(X_matrix)
```


```python
def get_top_words(model, feature_names, n_top_words=10, print_topics=True):
    topics_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics_dict[topic_idx] = top_words
        if print_topics:
            print(f"Topic {topic_idx}: {', '.join(top_words)}")
    return topics_dict

# Usage
feature_names = vectorizer.get_feature_names_out()
topics = get_top_words(lda, feature_names, n_top_words=15)
```


```python
# get the distribution array
topic_dist = lda.transform(X_matrix)
topic_dist


```


```python
topic_dist_df = pd.DataFrame(topic_dist)
```


```python
topic_dist_df = pd.DataFrame(topic_dist)

df = df.reset_index() ## need to reset index to merge properly)
df_w_topics = df.join(topic_dist_df)
df_w_topics.head()
```

__What is the most "representative topic" of topic 8?__


```python

```


```python
topic_of_interest = 8

df_w_topics[['name_abbreviation', 
             'decision_date', 
             'text',
              topic_of_interest]].sort_values(by=[topic_of_interest], ascending=False)
```


```python
!pip install pyldavis
```


```python
# pyLDAvis is a package which allows you to view topic distribution of your text
import pyLDAvis
import pyLDAvis.lda_model
pyLDAvis.enable_notebook()

lda_display = pyLDAvis.lda_model.prepare(lda, 
                                        X_matrix, 
                                        vectorizer)

pyLDAvis.save_html(lda_display, 'lda_visualization.html')
# See lda_visualization.html to explore the LDA based topics

lda_display
```

But is it right to say that there are only 10 topics in the corpus? Again, the limitations of unsupervised learning make themselves visible.

To find the optimal number of topics we can use a measure called `perplexity` - which measures how well the model predicts unseen documents — lower is better.

To understand perplexity, we can think of predicting the next word in a sentence - perplexity measures how well our model can guess it. Lower perplexity = model guesses are usually correct, the model is "less perplexed"


```python
perplexities = []
topic_range = range(5, 40, 1)  # test from 5 to 40 topics

for n_topics in topic_range:
    lda = LatentDirichletAllocation(n_components=n_topics, 
                                    random_state=0)
    lda.fit(X_matrix)
    perplexities.append(lda.perplexity(X_matrix))
    print(n_topics)
```


```python
# Plot
import matplotlib.pyplot as plt

plt.plot(topic_range, perplexities, marker='o')
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity (lower is better) - ie, less perplexed")
plt.title("LDA Perplexity by Number of Topics")
plt.show()
```

Just as before, we look for the “elbow” point — where decreasing perplexity slows down. That’s often a good choice for number of topics.

## NMF: a matrix decomposition method

### **Non-negative Matrix Factorization (NMF)**

Non-Negative Matrix Factorization (Lee and Seung, 1999) is a "Matrix decomposition" method with an imposed non-negativity constraint, meaning that all values in the resulting matrix must be non-zero.     


Recall from the classification lab, that the Document-Term Matrix is full of zeroes - but it has __no negative values__ meaning we can use non-negative matrix factorization. 

The intuition is that we define our Document Term Matrix (X) as being composed of 2 other matrices (W) and (H).

X ~= W * H

where

* X → Our Document-Term Matrix (documents × words)

* W → Document-topic matrix (documents × topics) 

* H → Topic-term matrix (topics × words)


<div align="center">
    <img src="https://www.researchgate.net/publication/374780969/figure/fig4/AS:11431281199413809@1697595827929/Non-Negative-Matrix-Factorisation-NMF-topic-modelling-algorithm.png" width="800">


In essence, NMF discovers hidden “topics” in the documents by factorizing the document term matrix into two other matrices.

As we saw before, it was initially used for image processing. 

<div align="center">
    <img src="https://blog.acolyer.org/wp-content/uploads/2019/02/nmf-fig-1.jpeg?w=640" width="800">








```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Vectorizer
tfidf_vec = TfidfVectorizer(
    max_df=0.95,      # Ignore words that appear in > 95% of docs (corpus-specific stop words)
    min_df=2,         # Ignore words that appear in only 1 document
    stop_words='english'
)

X = tfidf_vec.fit_transform(documents)
```


```python
# Fit NMF 
from sklearn.decomposition import NMF

n_topics = 40

model = NMF(n_components=n_topics, 
            init="nndsvd",  ## nndsvd better for sparse data
            random_state=42)

W = model.fit_transform(X) # w matrix

H = model.components_  # H matrix
```


```python
# get vocabulary from the vectorizer
vocab = tfidf_vec.get_feature_names_out()

# Extract Results 
def get_descriptors(features, H, top_n):
    top_indices = np.argsort(H, axis=1)[:, ::-1][:, :top_n]
    return features[top_indices]

# Display Topics
top_words_list = get_descriptors(vocab, H, 10)

for i, words in enumerate(top_words_list):
    print(f"Topic {i+1:02d}: {', '.join(words)}")
```


```python
# Function to return indices instead of just strings
def get_top_doc_indices(W, topic_index, top_n):
    # Get indices of the documents with the highest weights for this topic
    return np.argsort(W[:, topic_index])[::-1][:top_n]

topic_of_interest = 7
n_docs = 10

#  Get the top indices
top_indices = get_top_doc_indices(W, topic_of_interest, n_docs)

# Print summary and keep the indices for later use
print(f"--- Top Documents for Topic {topic_of_interest + 1} ---")
for i, idx in enumerate(top_indices):
    case_name = df.iloc[idx]['name_abbreviation']
    print(f"{i+1:02d}. [Index: {idx}] {case_name}")

# 3. Example: Read the full text of the #1 document in this topic
top_doc_index = top_indices[0]
full_text = df.iloc[top_doc_index]['text']
print(f"\n--- Full Text of Top Case ---\n{full_text[:500]}...") # Printing first 500 chars
```


```python
# Create topic column names
topic_cols = [f"topic_{i}" for i in range(n_topics)]

# Copy dataframe and attach topic weights
df_w_topics_nmf = df.copy()
df_w_topics_nmf[topic_cols] = W
```


```python
df_w_topics_nmf
```


```python
topic_of_interest = 8

df_w_topics_nmf[
    ['name_abbreviation', 'decision_date', 'text', f'topic_{topic_of_interest}']
].sort_values(by=f'topic_{topic_of_interest}', ascending=False)
```

# Part 4: Topics as Features

What's more interesting about topic modeling is that we can use them as features for classification. In fact, it's just another represnetation technique

Let's do the classification we did last class, but instead of bag of words representation, we will use topics.


```python
# Define the path using pathlib
del_ch_csv_path = Path("combined_cases_del-ch.csv")
del_csv_path = Path("combined_cases_del.csv")

# Read the CSV file into a pandas DataFrame
df_del_ch = pd.read_csv(del_ch_csv_path)
df_del = pd.read_csv(del_csv_path)
```


```python
combined_df = pd.concat([df_del_ch, df_del], 
                        ignore_index=True)

# Display the combined DataFrame
combined_df
```


```python
len(combined_df)
```


```python
df = combined_df[['court.name_abbreviation',
                  "name_abbreviation", 'name',
                  'decision_date', 
                  'court.name', 
                  'casebody.opinions' ]]
```


```python
court_name_counts = df["court.name"].value_counts()
print(court_name_counts)
```


```python
## Extract Court Information
df = df[df["court.name"].isin(["Delaware Court of Chancery" , "Delaware Superior Court"])]
```


```python
len(df)
```


```python
# Convert decision_date to datetime format
df['decision_date'] = pd.to_datetime(df['decision_date'])
```


```python

# Filter for cases on or after January 1, 1945
df = df[df['decision_date'] >= '1945-01-01']
```


```python
len(df)
```


```python
df['casebody.opinions'] = df['casebody.opinions'].apply(literal_eval)
```


```python
df['text'] = df['casebody.opinions'].apply(lambda x: x[0]['text'] if isinstance(x, list) and len(x) > 0 else '')
```


```python
df['clean_text'] = df['text'].apply(clean_text_fast)
```


```python
df["court_name_number"] = df['court.name'].replace({"Delaware Court of Chancery" : 1 , 
                                        'Delaware Superior Court': 0})

# make sure your column is not called "court" - otherwise you might confuse yourself as to the word count because there will be a column called court
```

So far so good. Now we must split the data BEFORE applying TF-IDF and NMF transformations, in order to prevent data leakage. 


```python
# Step 1: SPLIT FIRST
# Split the raw text and labels before any vectorization or NMF

from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(
    df, 
    df['court_name_number'], 
    test_size=0.2, 
    random_state=1, 
    stratify=df['court_name_number']
)
```


```python
# Step 2: FIT TF-IDF ON TRAIN ONLY
tfidf_vec = TfidfVectorizer(max_df=0.8, # initialize the TFIDF vectorizer
                            min_df=4)
```


```python
X_train_tfidf = tfidf_vec.fit_transform(df_train['clean_text']) ## fit transform - both. fit learns the parameters, transform uses what is learned in fit to transform data.

# Transform test data using the training vocabulary
X_test_tfidf = tfidf_vec.transform(df_test['clean_text']) ## only use transform
```


```python
# what the data looks like 
X_train_tfidf_df = pd.DataFrame(
    X_train_tfidf.toarray(), 
    columns=tfidf_vec.get_feature_names_out(),
    index=df_train.index
)

# View the result
X_train_tfidf_df.head()
```


```python
vocab = np.array(tfidf_vec.get_feature_names_out())
```


```python
# Step 3. FIT NMF ON TRAIN ONLY
n_topics = 40

# Intialize NMF
nmf = NMF(n_components=n_topics, 
          init="nndsvd", 
          random_state=42)

W_train = nmf.fit_transform(X_train_tfidf)  # both fit transform

# Transform test data using the training topics
W_test = nmf.transform(X_test_tfidf) # ONLY transform 

H = nmf.components_  ## topic word matrix 
```


```python
W_train.shape
H.shape
```


```python
X_train_final = pd.DataFrame(W_train, index=df_train.index)
X_test_final = pd.DataFrame(W_test, index=df_test.index)
```


```python
X_train_final
```


```python
# Create topic column names
topic_cols = [f"topic_{i}" for i in range(n_topics)]

# Convert W matrices to DataFrames
X_train_final = pd.DataFrame(W_train, index=y_train.index, columns=topic_cols)
X_test_final = pd.DataFrame(W_test, index=y_test.index, columns=topic_cols)

# Add topic features back to original data
df_train = pd.concat([y_train, X_train_final], axis=1)
df_test = pd.concat([y_test, X_test_final], axis=1)
```


```python
df_train
```


```python
# Define feature columns
topic_cols = [f"topic_{i}" for i in range(40)]

# Training data
X_train_final = df_train[topic_cols]
y_train_final = df_train['court_name_number']

# Testing data
X_test_final = df_test[topic_cols]
y_test_final = df_test['court_name_number']
```


```python
# import Support Vector Machines
from sklearn.svm import SVC
```


```python
# Initialize the SVM model (you can change the kernel to 'linear', 'poly', 'rbf', etc.)
svm = SVC(kernel='linear',
         random_state = 42)  # 'linear' kernel for linear SVM

# Train the SVM model on the training data
svm.fit(X_train_final, 
        y_train_final)
```


```python
y_train_pred = svm.predict(X_train_final)
```


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
```


```python
print("TRAIN Metrics")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1 Score:", f1_score(y_train, y_train_pred))
```


```python
# Predict the labels for the Test set
y_test_pred = svm.predict(X_test_final)
```


```python
print("\nTEST Metrics")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1 Score:", f1_score(y_test, y_test_pred))
```


```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
```


```python
cv_score = cross_val_score(svm,
                           X_train_final,
                           y_train_final, 
                           cv=5)
print(cv_score)
```


```python
cm = confusion_matrix(y_test_final, 
                      y_test_pred, 
                      labels=svm.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Chancery", "Superior"])
disp.plot()
```


```python
X_train_df
```


```python
weights = svm.coef_[0]

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Weight': weights
})
```


```python
# Helper to get top words for a topic
def get_topic_words(topic_idx, H, vocab, top_n=5):
    top_word_indices = np.argsort(H[topic_idx, :])[::-1][:top_n]
    return ", ".join(vocab[top_word_indices])

# Create feature importance DataFrame from trained SVM
weights = svm.coef_[0]  # binary classification
feature_importance_df = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Weight': weights
})

# Extract topic index from feature name
feature_importance_df['Topic_Index'] = feature_importance_df['Feature'].str.extract(r'topic_(\d+)').astype(int)

# Map top words from NMF H
feature_importance_df['Top_Words'] = feature_importance_df['Topic_Index'].apply(
    lambda x: get_topic_words(x, H, vocab, top_n=10)
)

# Assign court label based on weight sign
feature_importance_df['Court'] = feature_importance_df['Weight'].apply(
    lambda w: 'Del Chancery' if w > 0 else 'Delaware Superior'
)

# Final sorted report
final_report = feature_importance_df[['Feature', 'Weight', 'Court', 'Top_Words']].sort_values(
    by='Weight', ascending=False
).reset_index(drop=True)
```


```python
final_report
```


```python

```
