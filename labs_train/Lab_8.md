# Lab 8

This lab will introduce "word embeddings" (or a mapping of a word into a vector space), beginning with the word2vec algorithm - which started a revolution in NLP in 2013. 

This revolution is still ongoing, and it is currently continuing with "contextualized word embeddings" and "large language models" which even outperform humans at certain tasks (GPT, BERT, etc). 

But the **fundamentals of large language models are essentially the same as word2vec** - which is why it's important to understand word2vec word embeddings before proceeding to more "modern" representations. For instance, if you read the [Huggingface Primer on  LLMs](https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=what_are_embeddings?), you will see that "embeddings are the semantic backbone of LLMs." The idea of word embeddings comes directly from word2vec. 


So far we have focused on bag-of-word approaches i.e representations containing **documents** (rows) and **word counts** (columns). If we want TFIDF matrix, we apply TFIDF weighting. 


   Recall the example Document-Term Matrices that we used 
<div align="center">
<img src="https://static.cambridge.org/binary/version/id/urn:cambridge.org:id:binary:20240920053855882-0406:S1351324923000244:S1351324923000244_tab1.png?pub-status=live" width="500">
</div>

There are several important problems of classical text representation methods:

1. Words are simple counts - and documents are vectors of word counts. The bag of words representation has no information of context - the columns are simply ordered alphabetically, reflecting the fact that word order and thus **context** is completely disregarded  - hence "bag of words."
2. A document is a thus point in a space where each unique word in the corpus is a different "dimension".
3. This creates a "sparsity problem": because most documents only use a tiny fraction of the available vocabulary, these vectors are sparse (ie. contain mostly zeros). For most words, the column is going to be full of 0s - reflecting the fact that not every word is used in every document.
4. Mathematically, this representation can create a problem known as "orthogonality" - which just means that all word vectors may be orthogonal to each other (if they don't share documents), ie, at 90-degrees angles.
5. Thus, in a standard DTM, words have less information about context "perfectly independent" (orthogonal).
6. Geometrically, this representation does not capture any information about similarity or semantics or meaning of words (vectors).

Thus, in a DTM, the words "Apple" and "iPhone" are just as mathematically "distant"  as "Apple" and "Kangaroo" due to orthogonality. The DTM  records occurrence - but it lacks any information about semantics.

## More conceptual points 
As we should be familiar by now, in a Document Term Matrix (DTM) rows represent documents, and the columns words. 


### Words as Vectors vs Documents as Vectors

Let us look at another DTM again:
<div align="center">
<img src="https://www.ferventlearning.com/wp-content/uploads/2020/12/articleImagery_IAWNLP-ABF-28_4-scaled.jpg?495a33&495a33" width="500">
</div>
    
**w1** is **[4, 0, 0, 3, 0, 1]** vector, and **w2** is **[0, 1, 5, 0, 2, 0]** vector. 

Both word vectors do not share any "dimmensions" or document co-occurances (ie don't appear in the same document). So the cosine caculation here would mean they are orthogonal (90). If they shared one document, there would be non-zero cosine similarity, but it wouldn't give us much information about semantics, only showing that they appear in the same document. 

Cosine similarity between two orthogonal vectors is 0 - they are at 90 degrees with one another. Thus, even if we wanted to get "similarity" between two **words** in a DTM, we wouldn't get much information. 

But we still have more luck using cosine similarity with **documents**- because a document vector is the sum of its word vectors. 

If two documents share many of the same "dimensions" (words), their vectors will point in a similar direction in that high-dimensional space, allowing us to use Cosine Similarity effectively to find related documents.

**Document** comparison using cosine similarity using a DTM representation is thus actually a fruitful task. 


### Word2vec and [distributional hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics)

As we will see in a bit, word2vec allows us to represent *words* as __dense__ vectors (as opposed to __sparse__ vectors full of 0s). 

In word2vec, each word is __embedded__ in a vector space of a fixed dimension (usually 300) where __similar words__ are located together in a vector space. This allows us to do similarity calculation between words (using cosine) - thus gaining insight into their semantic content. The fact that each word is embedded as a vector in a 300 dimensional vector space is why each word represented by this method is called a __word embedding__. 

For large language models like BERT, the representation of words is a 768-dimensional vector. 

## Getting the data


```python
import numpy as np
import pandas as pd
from pathlib import Path
```


```python
del_ch_csv_path = Path("combined_cases_del-ch.csv")

df_del_ch = pd.read_csv(del_ch_csv_path)
```


```python
df_del_ch.head()
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
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>name</th>
      <th>name_abbreviation</th>
      <th>decision_date</th>
      <th>docket_number</th>
      <th>first_page</th>
      <th>last_page</th>
      <th>citations</th>
      <th>cites_to</th>
      <th>...</th>
      <th>provenance.date_added</th>
      <th>provenance.source</th>
      <th>provenance.batch</th>
      <th>casebody.judges</th>
      <th>casebody.parties</th>
      <th>casebody.opinions</th>
      <th>casebody.attorneys</th>
      <th>casebody.corrections</th>
      <th>casebody.head_matter</th>
      <th>source_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>457812</td>
      <td>Richard C. Dale, vs. Rowland Smith and Joseph ...</td>
      <td>Dale v. Smith</td>
      <td>1814-08-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>11</td>
      <td>[{'type': 'official', 'cite': '1 Del. Ch. 1'}]</td>
      <td>[]</td>
      <td>...</td>
      <td>2019-08-29</td>
      <td>Harvard</td>
      <td>2018</td>
      <td>[]</td>
      <td>['Richard C. Dale, vs. Rowland Smith and Josep...</td>
      <td>[{'text': "Ridoely, Chancellor.\nThe articles ...</td>
      <td>['Read, Rodney and Van Dyke, for complainant.'...</td>
      <td>NaN</td>
      <td>Richard C. Dale, vs. Rowland Smith and Joseph ...</td>
      <td>del-ch\extracted_files\1\json\0001-01.json</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>457807</td>
      <td>Richard C. Dale vs. Rowland Smith and Joseph T...</td>
      <td>Dale v. Smith</td>
      <td>1815-08-01</td>
      <td>NaN</td>
      <td>11</td>
      <td>13</td>
      <td>[{'type': 'official', 'cite': '1 Del. Ch. 11'}]</td>
      <td>[]</td>
      <td>...</td>
      <td>2019-08-29</td>
      <td>Harvard</td>
      <td>2018</td>
      <td>[]</td>
      <td>['Richard C. Dale vs. Rowland Smith and Joseph...</td>
      <td>[{'text': "Ridgely, Chancellor.\nIn the examin...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>Richard C. Dale vs. Rowland Smith and Joseph T...</td>
      <td>del-ch\extracted_files\1\json\0011-01.json</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>457832</td>
      <td>Charles Tatem and James Canby. vs. Joshua Gilp...</td>
      <td>Tatem v. Gilpin</td>
      <td>1816-06-01</td>
      <td>NaN</td>
      <td>13</td>
      <td>23</td>
      <td>[{'type': 'official', 'cite': '1 Del. Ch. 13'}]</td>
      <td>[{'cite': '6 Cranch 51', 'category': 'reporter...</td>
      <td>...</td>
      <td>2019-08-29</td>
      <td>Harvard</td>
      <td>2018</td>
      <td>[]</td>
      <td>['Charles Tatem and James Canby. vs. Joshua Gi...</td>
      <td>[{'text': "Ridgely, Chancellor.\nThis is a cas...</td>
      <td>['Broom, and Read, for defendants.', 'McLane a...</td>
      <td>NaN</td>
      <td>Charles Tatem and James Canby. vs. Joshua Gilp...</td>
      <td>del-ch\extracted_files\1\json\0013-01.json</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>457846</td>
      <td>Jeremiah Woolaston, et al, vs. Thomas Mendenhall</td>
      <td>Woolaston v. Mendenhall</td>
      <td>1817-08-01</td>
      <td>NaN</td>
      <td>23</td>
      <td>25</td>
      <td>[{'type': 'official', 'cite': '1 Del. Ch. 23'}]</td>
      <td>[{'cite': '4 Del. Laws 444', 'category': 'laws...</td>
      <td>...</td>
      <td>2019-08-29</td>
      <td>Harvard</td>
      <td>2018</td>
      <td>[]</td>
      <td>['Jeremiah Woolaston, et al, vs. Thomas Menden...</td>
      <td>[{'text': 'Read,\nof counsel for the defendant...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>Jeremiah Woolaston, et al, vs. Thomas Mendenha...</td>
      <td>del-ch\extracted_files\1\json\0023-01.json</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>457847</td>
      <td>The State, vs. Joshua Gilpin and Thomas Gilpin</td>
      <td>State v. Gilpin</td>
      <td>1817-08-01</td>
      <td>NaN</td>
      <td>25</td>
      <td>31</td>
      <td>[{'type': 'official', 'cite': '1 Del. Ch. 25'}]</td>
      <td>[]</td>
      <td>...</td>
      <td>2019-08-29</td>
      <td>Harvard</td>
      <td>2018</td>
      <td>[]</td>
      <td>['The State, vs. Joshua Gilpin and Thomas Gilp...</td>
      <td>[{'text': "Bidgely, Chancellor.\nIt is a gener...</td>
      <td>['Read,in support of the exception.', 'Van Dyk...</td>
      <td>NaN</td>
      <td>The State, vs. Joshua Gilpin and Thomas Gilpin...</td>
      <td>del-ch\extracted_files\1\json\0025-01.json</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
df = df_del_ch[['court.name_abbreviation',
                  "name_abbreviation", 
                  'decision_date', 
                  'court.name', 
                  'casebody.opinions' ]]
```


```python
from ast import literal_eval
df['casebody.opinions'] = df['casebody.opinions'].apply(literal_eval)
```

    /var/folders/r9/cwv87v850tz71nb16n0kmzf80000gn/T/ipykernel_2973/844464783.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['casebody.opinions'] = df['casebody.opinions'].apply(literal_eval)



```python
df['text'] = df['casebody.opinions'].apply(lambda x: x[0]['text'] if isinstance(x, list) and len(x) > 0 else '')
```

    /var/folders/r9/cwv87v850tz71nb16n0kmzf80000gn/T/ipykernel_2973/502085088.py:1: SettingWithCopyWarning: 
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
The articles •...</td>
      <td>Ridoely, Chancellor.\nThe articles • of agreem...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Del. Ch.</td>
      <td>Dale v. Smith</td>
      <td>1815-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Ridgely, Chancellor.
In the examina...</td>
      <td>Ridgely, Chancellor.\nIn the examination of Da...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Del. Ch.</td>
      <td>Tatem v. Gilpin</td>
      <td>1816-06-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Ridgely, Chancellor.
This is a case...</td>
      <td>Ridgely, Chancellor.\nThis is a case which com...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Del. Ch.</td>
      <td>Woolaston v. Mendenhall</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Read,
of counsel for the defendant,...</td>
      <td>Read,\nof counsel for the defendant, doubted w...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Del. Ch.</td>
      <td>State v. Gilpin</td>
      <td>1817-08-01</td>
      <td>Delaware Court of Chancery</td>
      <td>[{'text': 'Bidgely, Chancellor.
It is a genera...</td>
      <td>Bidgely, Chancellor.\nIt is a general rule tha...</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    2361



## Dot Product and Cosine similarity for documents based on simple count and TFIDF Document Term Matrix representations

According to Wikpiedia, "Cosine similarity is a measure of similarity between two sequences of numbers." 

<div align="center">
<img src="https://storage.googleapis.com/lds-media/images/cosine-similarity-vectors.original.jpg" width="900">
</div>



If there is a small angle between two vectors (which is just a sequence of numbers) - that means they are similar. 

Don't underestimate the usefulness of cosine similarity measures. Imagine you have to find the most similar case to another case in a corpus. How would you go about it? Cosine similarity can help here.

The idea is very simple:

* Cosine(10 degrees) gives you 0.98 __"cosine similarity"__ measure. This can be interpreted as vectors are "98% similar". 

* Cosine(90 degrees) gives you a 0 __"cosine similiarty"__. This can be interpreted as vectors are "0% similar". This is known as "orthogonal vectors" - ie unrelated. 

Cosine similarity is an extension of the so-called **dot product** (also known as scalar product, or inner product), which just means that we get a single number out of two sequences of numbers (vectors).

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$


```python
import numpy as np

A = np.array([1, 2, 3])
B = np.array([100, 200, 103])

dot_product = np.dot(A, B)

print(dot_product)
```

    809


$$\mathbf{a} \cdot \mathbf{b} = (1 \times 10) + (2 \times 20) + (3 \times 13) = 89$$

The dot product will be bigger if vectors point in similar directions, and small (or negative) if the vectors point in different directions.
So to make them dissimilar, you want opposite signs or very different directions.

Cosine similarity is basically the **normalized dot product**. We divide the dot product by the product of the **norms** or **magnitudes** or **lengths** of each vector  

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

where the norm is calculated as the square root of the sum of the squares of its components:

$$\|\mathbf{a}\| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2} = \sqrt{\sum_{i=1}^{n} a_i^2}$$


```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

print(cos_sim)
```

    0.9746318461970762


Step 1 -  calculate the dot product:
$$\mathbf{a} \cdot \mathbf{b} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = \mathbf{32}$$

Step 2 - get the length of the vectors:

$$\|\mathbf{a}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.742$$

$$\|\mathbf{b}\| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{16 + 25 + 36} = \sqrt{77} \approx 8.775$$

Step 3 - calculate the cosine similarity (normalized dot product):

$$\cos(\theta) = \frac{32}{\sqrt{14} \times \sqrt{77}} = \frac{32}{\sqrt{1078}} \approx \frac{32}{32.833} \approx \mathbf{0.9746}$$

A cosine similarity of **0.9746** indicates that the two vectors are pointing in nearly the same direction in vector space. Because it is bounded between 1 and -1, when it's positive, I tend to think of it as a percentage (ie, 90% similar). However this interpretation is not entirely accurate because cosine similarity can be negative. 


```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

A = np.array([[1, 2, 3]])
B = np.array([[4, 5, 6]])

cos_sim = cosine_similarity(A, B)

print(cos_sim)
```

    [[0.97463185]]



```python
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, PointDrawTool, CustomJS, Label, LabelSet

# Initial vectors
ax, ay = 52, 2
bx, by = -10, -4 

# ColumnDataSource
source = ColumnDataSource(data=dict(
    x=[0,0], y=[0,0],
    xe=[ax,bx], ye=[ay,by],
    color=["#3182bd","#31a354"],
    text=[f"({ax:.1f},{ay:.1f})", f"({bx:.1f},{by:.1f})"]
))

# Figure
p = figure(x_range=(-60, 60), y_range=(-40, 40), width=700, height=500,
           title="Vectors with Angle θ")

# Draw vectors
p.segment(x0='x', y0='y', x1='xe', y1='ye', source=source,
          line_width=4, color='color', line_alpha=0.8)
coords_labels = LabelSet(x='xe', y='ye', text='text', source=source,
                         x_offset=5, y_offset=5, text_font_size='10pt')
p.add_layout(coords_labels)

# Dot, Cos, Angle labels
dot_label = Label(x=-50, y=35, x_units='data', y_units='data',
                  text="", background_fill_alpha=0.8)
cos_label = Label(x=-50, y=32, x_units='data', y_units='data',
                  text="", background_fill_alpha=0.8)
angle_label = Label(x=-50, y=29, x_units='data', y_units='data',
                    text="", background_fill_alpha=0.8)
p.add_layout(dot_label)
p.add_layout(cos_label)
p.add_layout(angle_label)

# Arc source
arc_source = ColumnDataSource(data=dict(x=[],y=[]))
p.line('x','y', source=arc_source, line_color='orange', line_width=2)

# Draggable points
render = p.scatter('xe','ye', source=source, size=15, color='color', marker='circle')

# JS Callback
callback = CustomJS(args=dict(s=source, dl=dot_label, cl=cos_label,
                              al=angle_label, arc=arc_source), code="""
    const d = s.data;
    for (let i=0; i<d['xe'].length; i++){
        d['x'][i] = 0; d['y'][i] = 0;
        d['text'][i] = `(${d['xe'][i].toFixed(1)},${d['ye'][i].toFixed(1)})`;
    }

    const ax=d['xe'][0], ay=d['ye'][0];
    const bx=d['xe'][1], by=d['ye'][1];

    const dot = ax*bx + ay*by;
    const magA = Math.sqrt(ax*ax + ay*ay);
    const magB = Math.sqrt(bx*bx + by*by);
    const cos_sim = dot / (magA*magB);
    const angle_rad = Math.acos(Math.min(Math.max(cos_sim,-1),1));
    const angle_deg = angle_rad*180/Math.PI;

    dl.text = `Dot Product: ${dot.toFixed(2)}`;
    cl.text = `Cosine: ${cos_sim.toFixed(4)}`;
    al.text = `Angle θ: ${angle_deg.toFixed(1)}°`;

    // Compute robust arc from vector A to B
    const r = 5;
    let angleA = Math.atan2(ay, ax);
    let angleB = Math.atan2(by, bx);
    if(angleB < angleA){ angleB += 2*Math.PI; }

    const n = 30;
    const arc_x = [];
    const arc_y = [];
    for(let i=0;i<=n;i++){
        const a = angleA + i*(angleB-angleA)/n;
        arc_x.push(r*Math.cos(a));
        arc_y.push(r*Math.sin(a));
    }
    arc.data['x'] = arc_x;
    arc.data['y'] = arc_y;

    s.change.emit();
    arc.change.emit();
""")

source.js_on_change('data', callback)

# Draw tool
draw_tool = PointDrawTool(renderers=[render], num_objects=2)
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool

show(p)
```


```python
!pip install dash
```

    Requirement already satisfied: dash in /opt/anaconda3/lib/python3.13/site-packages (4.0.0)
    Requirement already satisfied: Flask<3.2,>=1.0.4 in /opt/anaconda3/lib/python3.13/site-packages (from dash) (3.1.0)
    Requirement already satisfied: Werkzeug<3.2 in /opt/anaconda3/lib/python3.13/site-packages (from dash) (3.1.3)
    Requirement already satisfied: plotly>=5.0.0 in /opt/anaconda3/lib/python3.13/site-packages (from dash) (5.24.1)
    Requirement already satisfied: importlib-metadata in /opt/anaconda3/lib/python3.13/site-packages (from dash) (8.5.0)
    Requirement already satisfied: typing_extensions>=4.1.1 in /opt/anaconda3/lib/python3.13/site-packages (from dash) (4.12.2)
    Requirement already satisfied: requests in /opt/anaconda3/lib/python3.13/site-packages (from dash) (2.32.3)
    Requirement already satisfied: retrying in /opt/anaconda3/lib/python3.13/site-packages (from dash) (1.4.2)
    Requirement already satisfied: nest-asyncio in /opt/anaconda3/lib/python3.13/site-packages (from dash) (1.6.0)
    Requirement already satisfied: setuptools in /opt/anaconda3/lib/python3.13/site-packages (from dash) (72.1.0)
    Requirement already satisfied: Jinja2>=3.1.2 in /opt/anaconda3/lib/python3.13/site-packages (from Flask<3.2,>=1.0.4->dash) (3.1.6)
    Requirement already satisfied: itsdangerous>=2.2 in /opt/anaconda3/lib/python3.13/site-packages (from Flask<3.2,>=1.0.4->dash) (2.2.0)
    Requirement already satisfied: click>=8.1.3 in /opt/anaconda3/lib/python3.13/site-packages (from Flask<3.2,>=1.0.4->dash) (8.1.8)
    Requirement already satisfied: blinker>=1.9 in /opt/anaconda3/lib/python3.13/site-packages (from Flask<3.2,>=1.0.4->dash) (1.9.0)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /opt/anaconda3/lib/python3.13/site-packages (from Werkzeug<3.2->dash) (3.0.2)
    Requirement already satisfied: tenacity>=6.2.0 in /opt/anaconda3/lib/python3.13/site-packages (from plotly>=5.0.0->dash) (9.0.0)
    Requirement already satisfied: packaging in /opt/anaconda3/lib/python3.13/site-packages (from plotly>=5.0.0->dash) (24.2)
    Requirement already satisfied: zipp>=3.20 in /opt/anaconda3/lib/python3.13/site-packages (from importlib-metadata->dash) (3.21.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.13/site-packages (from requests->dash) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.13/site-packages (from requests->dash) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/lib/python3.13/site-packages (from requests->dash) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.13/site-packages (from requests->dash) (2025.8.3)



```python
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

# Initial vectors
v1 = np.array([52, 2, 10], dtype=float)
v2 = np.array([10, 4, 20], dtype=float)

# Metrics
def compute_metrics(v1, v2):
    dot = np.dot(v1, v2)
    cos = dot / (np.linalg.norm(v1)*np.linalg.norm(v2))
    angle_deg = np.degrees(np.arccos(np.clip(cos,-1,1)))
    return dot, cos, angle_deg

# Create 3D figure
def create_figure(v1, v2):
    dot, cos, angle_deg = compute_metrics(v1, v2)
    
    fig = go.Figure()
    # Vector A
    fig.add_trace(go.Scatter3d(
        x=[0,v1[0]], y=[0,v1[1]], z=[0,v1[2]],
        mode='lines+markers+text', line=dict(width=6,color='blue'),
        marker=dict(size=4,color='blue'),
        text=[None,f"({v1[0]:.1f},{v1[1]:.1f},{v1[2]:.1f})"],
        textposition='top center', name='Vector A'
    ))
    # Vector B
    fig.add_trace(go.Scatter3d(
        x=[0,v2[0]], y=[0,v2[1]], z=[0,v2[2]],
        mode='lines+markers+text', line=dict(width=6,color='green'),
        marker=dict(size=4,color='green'),
        text=[None,f"({v2[0]:.1f},{v2[1]:.1f},{v2[2]:.1f})"],
        textposition='top center', name='Vector B'
    ))
    
    # Angle arc
    n = 20
    if np.linalg.norm(v1)>0 and np.linalg.norm(v2)>0:
        u = v1/np.linalg.norm(v1)
        v = v2/np.linalg.norm(v2)
        axis = np.cross(u,v)
        if np.linalg.norm(axis)>1e-6:
            axis = axis/np.linalg.norm(axis)
            theta = np.arccos(np.clip(np.dot(u,v),-1,1))
            r = 5
            arc_points = []
            for t in np.linspace(0,theta,n):
                point = u*np.cos(t) + np.cross(axis,u)*np.sin(t) + axis*np.dot(axis,u)*(1-np.cos(t))
                arc_points.append(point*r)
            arc_points = np.array(arc_points)
            fig.add_trace(go.Scatter3d(
                x=arc_points[:,0], y=arc_points[:,1], z=arc_points[:,2],
                mode='lines', line=dict(color='orange', width=4),
                name='Angle θ Arc'
            ))
    
    fig.update_layout(
        title=f"Dot: {dot:.2f} | Cos: {cos:.4f} | Angle θ: {angle_deg:.1f}°",
        scene=dict(xaxis=dict(title='X', range=[-60,60]),
                   yaxis=dict(title='Y', range=[-60,60]),
                   zaxis=dict(title='Z', range=[-60,60])),
        width=900, height=700
    )
    return fig

# App layout
app = Dash(__name__)
app.layout = html.Div([
    html.H2("3D Vectors with Angle"),
    html.Div([
        # Left column: vector entries
        html.Div([
            html.Div([html.Label("Vector A X:"), dcc.Input(id='ax', type='number', value=v1[0])], style={'margin':5}),
            html.Div([html.Label("Vector A Y:"), dcc.Input(id='ay', type='number', value=v1[1])], style={'margin':5}),
            html.Div([html.Label("Vector A Z:"), dcc.Input(id='az', type='number', value=v1[2])], style={'margin':5}),
            html.Br(),
            html.Div([html.Label("Vector B X:"), dcc.Input(id='bx', type='number', value=v2[0])], style={'margin':5}),
            html.Div([html.Label("Vector B Y:"), dcc.Input(id='by', type='number', value=v2[1])], style={'margin':5}),
            html.Div([html.Label("Vector B Z:"), dcc.Input(id='bz', type='number', value=v2[2])], style={'margin':5}),
        ], style={'flex': '0 0 200px'}),

        # Right column: 3D graph
        html.Div([
            dcc.Graph(id='vector-plot', figure=create_figure(v1,v2))
        ], style={'flex': '1', 'margin-left':'20px'})
    ], style={'display':'flex', 'flex-direction':'row'})
])

# Callback
@app.callback(
    Output('vector-plot','figure'),
    Input('ax','value'), Input('ay','value'), Input('az','value'),
    Input('bx','value'), Input('by','value'), Input('bz','value')
)
def update_plot(ax, ay, az, bx, by, bz):
    v1_new = np.array([ax, ay, az], dtype=float)
    v2_new = np.array([bx, by, bz], dtype=float)
    return create_figure(v1_new, v2_new)

if __name__ == '__main__':
    app.run(debug=True)
```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8050/"
    frameborder="0"
    allowfullscreen

></iframe>



Let us try similarity measurements for documents in a DTM.


```python
from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(min_df=0.1,
                         max_df=.9,  
                         max_features=10000,
                         stop_words='english',
                         ngram_range=(1,1))
```


```python
X_tf = tf_vectorizer.fit_transform(df['text'])

tf = pd.DataFrame(data = X_tf.toarray(), 
                  columns = tf_vectorizer.get_feature_names_out())

tf.head()
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
      <th>00</th>
      <th>000</th>
      <th>10</th>
      <th>100</th>
      <th>11</th>
      <th>115</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>...</th>
      <th>witnesses</th>
      <th>word</th>
      <th>words</th>
      <th>work</th>
      <th>worth</th>
      <th>writing</th>
      <th>written</th>
      <th>year</th>
      <th>years</th>
      <th>york</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1155 columns</p>
</div>




```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=0.1,
                                   max_df=.9,  
                                   max_features=1000,
                                   stop_words='english',
                                   ngram_range=(1,1))
```


```python
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

tf_idf = pd.DataFrame(data = X_tfidf.toarray(), 
                      columns = tfidf_vectorizer.get_feature_names_out())

tf_idf.head()
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
      <th>00</th>
      <th>000</th>
      <th>10</th>
      <th>100</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>...</th>
      <th>witnesses</th>
      <th>word</th>
      <th>words</th>
      <th>work</th>
      <th>worth</th>
      <th>writing</th>
      <th>written</th>
      <th>year</th>
      <th>years</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.053882</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.094467</td>
      <td>0.144597</td>
      <td>0.037362</td>
      <td>0.015820</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.155677</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.027374</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011873</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.022249</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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

```


```python
# the cosine similarity measures similarity between rows of a matrix - making it into a Square matrix.

cos_sim_tf = cosine_similarity(X_tf)
cos_sim_tfidf = cosine_similarity(X_tfidf)
```

### CountVectorizer cosine similarity matrix


```python
cv_cos_sim = pd.DataFrame(data = cos_sim_tf, 
                          columns = df['name_abbreviation'],
                          index = df['name_abbreviation'])

cv_cos_sim
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
      <th>name_abbreviation</th>
      <th>Dale v. Smith</th>
      <th>Dale v. Smith</th>
      <th>Tatem v. Gilpin</th>
      <th>Woolaston v. Mendenhall</th>
      <th>State v. Gilpin</th>
      <th>Clayton v. Mitchell</th>
      <th>Rodney v. Shankland</th>
      <th>Warner v. Allee</th>
      <th>Philip v. Wood</th>
      <th>Thompson v. Lynam</th>
      <th>...</th>
      <th>Slaughter v. Moore</th>
      <th>Walter v. Peninsula Cut Stone Co.</th>
      <th>Williamson v. McMonagle</th>
      <th>Emmons v. Curlett</th>
      <th>Jacobs v. Wilmington Trust Co.</th>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <th>Dayett v. Willitts</th>
      <th>In re McFarlin</th>
      <th>In re the Real Estate of Donaghy</th>
      <th>In re Tomlinson</th>
    </tr>
    <tr>
      <th>name_abbreviation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dale v. Smith</th>
      <td>1.000000</td>
      <td>0.327434</td>
      <td>0.144093</td>
      <td>0.099391</td>
      <td>0.155623</td>
      <td>0.140167</td>
      <td>0.309922</td>
      <td>0.128332</td>
      <td>0.428791</td>
      <td>0.184770</td>
      <td>...</td>
      <td>0.145207</td>
      <td>0.091564</td>
      <td>0.196714</td>
      <td>0.249375</td>
      <td>0.132173</td>
      <td>0.117521</td>
      <td>0.152195</td>
      <td>0.146265</td>
      <td>0.223285</td>
      <td>0.178340</td>
    </tr>
    <tr>
      <th>Dale v. Smith</th>
      <td>0.327434</td>
      <td>1.000000</td>
      <td>0.091212</td>
      <td>0.159717</td>
      <td>0.129208</td>
      <td>0.054254</td>
      <td>0.158348</td>
      <td>0.093933</td>
      <td>0.124189</td>
      <td>0.052447</td>
      <td>...</td>
      <td>0.100606</td>
      <td>0.041566</td>
      <td>0.092906</td>
      <td>0.044632</td>
      <td>0.031702</td>
      <td>0.052062</td>
      <td>0.082261</td>
      <td>0.065774</td>
      <td>0.044478</td>
      <td>0.047838</td>
    </tr>
    <tr>
      <th>Tatem v. Gilpin</th>
      <td>0.144093</td>
      <td>0.091212</td>
      <td>1.000000</td>
      <td>0.436965</td>
      <td>0.316888</td>
      <td>0.232772</td>
      <td>0.158208</td>
      <td>0.137373</td>
      <td>0.077107</td>
      <td>0.254108</td>
      <td>...</td>
      <td>0.187861</td>
      <td>0.149652</td>
      <td>0.379865</td>
      <td>0.202001</td>
      <td>0.140095</td>
      <td>0.173931</td>
      <td>0.203933</td>
      <td>0.140139</td>
      <td>0.226336</td>
      <td>0.347184</td>
    </tr>
    <tr>
      <th>Woolaston v. Mendenhall</th>
      <td>0.099391</td>
      <td>0.159717</td>
      <td>0.436965</td>
      <td>1.000000</td>
      <td>0.176897</td>
      <td>0.102450</td>
      <td>0.130154</td>
      <td>0.089948</td>
      <td>0.080866</td>
      <td>0.085377</td>
      <td>...</td>
      <td>0.079619</td>
      <td>0.103488</td>
      <td>0.203486</td>
      <td>0.089262</td>
      <td>0.073725</td>
      <td>0.172892</td>
      <td>0.133224</td>
      <td>0.101150</td>
      <td>0.112800</td>
      <td>0.126823</td>
    </tr>
    <tr>
      <th>State v. Gilpin</th>
      <td>0.155623</td>
      <td>0.129208</td>
      <td>0.316888</td>
      <td>0.176897</td>
      <td>1.000000</td>
      <td>0.163856</td>
      <td>0.161356</td>
      <td>0.178519</td>
      <td>0.112386</td>
      <td>0.305053</td>
      <td>...</td>
      <td>0.126375</td>
      <td>0.107762</td>
      <td>0.253598</td>
      <td>0.107477</td>
      <td>0.087476</td>
      <td>0.168815</td>
      <td>0.131479</td>
      <td>0.132203</td>
      <td>0.117149</td>
      <td>0.142197</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <td>0.117521</td>
      <td>0.052062</td>
      <td>0.173931</td>
      <td>0.172892</td>
      <td>0.168815</td>
      <td>0.242520</td>
      <td>0.230403</td>
      <td>0.200236</td>
      <td>0.124329</td>
      <td>0.127230</td>
      <td>...</td>
      <td>0.478613</td>
      <td>0.396707</td>
      <td>0.176729</td>
      <td>0.191912</td>
      <td>0.281313</td>
      <td>1.000000</td>
      <td>0.238622</td>
      <td>0.223453</td>
      <td>0.173327</td>
      <td>0.181026</td>
    </tr>
    <tr>
      <th>Dayett v. Willitts</th>
      <td>0.152195</td>
      <td>0.082261</td>
      <td>0.203933</td>
      <td>0.133224</td>
      <td>0.131479</td>
      <td>0.249881</td>
      <td>0.290773</td>
      <td>0.205691</td>
      <td>0.140967</td>
      <td>0.203008</td>
      <td>...</td>
      <td>0.191887</td>
      <td>0.400984</td>
      <td>0.148236</td>
      <td>0.355137</td>
      <td>0.307823</td>
      <td>0.238622</td>
      <td>1.000000</td>
      <td>0.217928</td>
      <td>0.249920</td>
      <td>0.343404</td>
    </tr>
    <tr>
      <th>In re McFarlin</th>
      <td>0.146265</td>
      <td>0.065774</td>
      <td>0.140139</td>
      <td>0.101150</td>
      <td>0.132203</td>
      <td>0.101005</td>
      <td>0.286497</td>
      <td>0.185652</td>
      <td>0.183870</td>
      <td>0.213466</td>
      <td>...</td>
      <td>0.168872</td>
      <td>0.276021</td>
      <td>0.281588</td>
      <td>0.199450</td>
      <td>0.260446</td>
      <td>0.223453</td>
      <td>0.217928</td>
      <td>1.000000</td>
      <td>0.186306</td>
      <td>0.175127</td>
    </tr>
    <tr>
      <th>In re the Real Estate of Donaghy</th>
      <td>0.223285</td>
      <td>0.044478</td>
      <td>0.226336</td>
      <td>0.112800</td>
      <td>0.117149</td>
      <td>0.223481</td>
      <td>0.188734</td>
      <td>0.147393</td>
      <td>0.237049</td>
      <td>0.325356</td>
      <td>...</td>
      <td>0.162269</td>
      <td>0.233893</td>
      <td>0.197005</td>
      <td>0.395732</td>
      <td>0.189550</td>
      <td>0.173327</td>
      <td>0.249920</td>
      <td>0.186306</td>
      <td>1.000000</td>
      <td>0.379008</td>
    </tr>
    <tr>
      <th>In re Tomlinson</th>
      <td>0.178340</td>
      <td>0.047838</td>
      <td>0.347184</td>
      <td>0.126823</td>
      <td>0.142197</td>
      <td>0.358627</td>
      <td>0.195566</td>
      <td>0.121079</td>
      <td>0.125428</td>
      <td>0.249325</td>
      <td>...</td>
      <td>0.181356</td>
      <td>0.189520</td>
      <td>0.227140</td>
      <td>0.393169</td>
      <td>0.246876</td>
      <td>0.181026</td>
      <td>0.343404</td>
      <td>0.175127</td>
      <td>0.379008</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>2361 rows × 2361 columns</p>
</div>



We can sort the column values to get the "top similar" cases.


```python
cv_cos_sim.sort_values(by='Clayton v. Mitchell', 
                          ascending=False).head(15)
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
      <th>name_abbreviation</th>
      <th>Dale v. Smith</th>
      <th>Dale v. Smith</th>
      <th>Tatem v. Gilpin</th>
      <th>Woolaston v. Mendenhall</th>
      <th>State v. Gilpin</th>
      <th>Clayton v. Mitchell</th>
      <th>Rodney v. Shankland</th>
      <th>Warner v. Allee</th>
      <th>Philip v. Wood</th>
      <th>Thompson v. Lynam</th>
      <th>...</th>
      <th>Slaughter v. Moore</th>
      <th>Walter v. Peninsula Cut Stone Co.</th>
      <th>Williamson v. McMonagle</th>
      <th>Emmons v. Curlett</th>
      <th>Jacobs v. Wilmington Trust Co.</th>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <th>Dayett v. Willitts</th>
      <th>In re McFarlin</th>
      <th>In re the Real Estate of Donaghy</th>
      <th>In re Tomlinson</th>
    </tr>
    <tr>
      <th>name_abbreviation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Clayton v. Mitchell</th>
      <td>0.140167</td>
      <td>0.054254</td>
      <td>0.232772</td>
      <td>0.102450</td>
      <td>0.163856</td>
      <td>1.000000</td>
      <td>0.259759</td>
      <td>0.141406</td>
      <td>0.185418</td>
      <td>0.103682</td>
      <td>...</td>
      <td>0.323558</td>
      <td>0.118820</td>
      <td>0.253790</td>
      <td>0.338259</td>
      <td>0.227030</td>
      <td>0.242520</td>
      <td>0.249881</td>
      <td>0.101005</td>
      <td>0.223481</td>
      <td>0.358627</td>
    </tr>
    <tr>
      <th>Parker v. Yerger</th>
      <td>0.233359</td>
      <td>0.062570</td>
      <td>0.256116</td>
      <td>0.061114</td>
      <td>0.084231</td>
      <td>0.548747</td>
      <td>0.214493</td>
      <td>0.039701</td>
      <td>0.180837</td>
      <td>0.090585</td>
      <td>...</td>
      <td>0.169380</td>
      <td>0.145275</td>
      <td>0.198393</td>
      <td>0.503488</td>
      <td>0.275761</td>
      <td>0.191352</td>
      <td>0.307183</td>
      <td>0.038674</td>
      <td>0.385565</td>
      <td>0.586396</td>
    </tr>
    <tr>
      <th>Burton v. Willen</th>
      <td>0.277917</td>
      <td>0.162488</td>
      <td>0.304036</td>
      <td>0.106141</td>
      <td>0.157467</td>
      <td>0.538206</td>
      <td>0.399191</td>
      <td>0.308237</td>
      <td>0.261741</td>
      <td>0.115973</td>
      <td>...</td>
      <td>0.397925</td>
      <td>0.266696</td>
      <td>0.237432</td>
      <td>0.521195</td>
      <td>0.414866</td>
      <td>0.300329</td>
      <td>0.376075</td>
      <td>0.239349</td>
      <td>0.319349</td>
      <td>0.522179</td>
    </tr>
    <tr>
      <th>Agostini v. Colonial Trust Co.</th>
      <td>0.153347</td>
      <td>0.035934</td>
      <td>0.176084</td>
      <td>0.112862</td>
      <td>0.223433</td>
      <td>0.526354</td>
      <td>0.175650</td>
      <td>0.161561</td>
      <td>0.194777</td>
      <td>0.085930</td>
      <td>...</td>
      <td>0.203739</td>
      <td>0.136137</td>
      <td>0.394087</td>
      <td>0.279464</td>
      <td>0.152379</td>
      <td>0.202191</td>
      <td>0.253072</td>
      <td>0.095321</td>
      <td>0.173874</td>
      <td>0.279059</td>
    </tr>
    <tr>
      <th>Owens v. Owens</th>
      <td>0.120500</td>
      <td>0.040152</td>
      <td>0.288128</td>
      <td>0.084972</td>
      <td>0.060809</td>
      <td>0.512340</td>
      <td>0.248589</td>
      <td>0.186667</td>
      <td>0.080047</td>
      <td>0.083708</td>
      <td>...</td>
      <td>0.176218</td>
      <td>0.210734</td>
      <td>0.152776</td>
      <td>0.468961</td>
      <td>0.447760</td>
      <td>0.197118</td>
      <td>0.364141</td>
      <td>0.131411</td>
      <td>0.282713</td>
      <td>0.489922</td>
    </tr>
    <tr>
      <th>Cannon v. Hudson</th>
      <td>0.169070</td>
      <td>0.022732</td>
      <td>0.323679</td>
      <td>0.047019</td>
      <td>0.092276</td>
      <td>0.500627</td>
      <td>0.256458</td>
      <td>0.092025</td>
      <td>0.099185</td>
      <td>0.156113</td>
      <td>...</td>
      <td>0.146440</td>
      <td>0.178312</td>
      <td>0.184661</td>
      <td>0.509033</td>
      <td>0.300749</td>
      <td>0.165668</td>
      <td>0.419605</td>
      <td>0.091048</td>
      <td>0.347906</td>
      <td>0.598148</td>
    </tr>
    <tr>
      <th>Mayor of Wilmington v. Addicks</th>
      <td>0.170625</td>
      <td>0.054486</td>
      <td>0.296911</td>
      <td>0.060314</td>
      <td>0.144225</td>
      <td>0.486467</td>
      <td>0.163417</td>
      <td>0.095338</td>
      <td>0.078622</td>
      <td>0.065788</td>
      <td>...</td>
      <td>0.438020</td>
      <td>0.127094</td>
      <td>0.178655</td>
      <td>0.417955</td>
      <td>0.260477</td>
      <td>0.335639</td>
      <td>0.267241</td>
      <td>0.098506</td>
      <td>0.273784</td>
      <td>0.514089</td>
    </tr>
    <tr>
      <th>Willey v. Tindal</th>
      <td>0.176679</td>
      <td>0.042779</td>
      <td>0.290780</td>
      <td>0.053289</td>
      <td>0.061399</td>
      <td>0.481879</td>
      <td>0.264341</td>
      <td>0.125278</td>
      <td>0.149262</td>
      <td>0.144763</td>
      <td>...</td>
      <td>0.184313</td>
      <td>0.235247</td>
      <td>0.151531</td>
      <td>0.560962</td>
      <td>0.384395</td>
      <td>0.214184</td>
      <td>0.401505</td>
      <td>0.166098</td>
      <td>0.395073</td>
      <td>0.623488</td>
    </tr>
    <tr>
      <th>Hayes v. Hayes</th>
      <td>0.119431</td>
      <td>0.049566</td>
      <td>0.470384</td>
      <td>0.305535</td>
      <td>0.201335</td>
      <td>0.474132</td>
      <td>0.199578</td>
      <td>0.134508</td>
      <td>0.137189</td>
      <td>0.112518</td>
      <td>...</td>
      <td>0.157218</td>
      <td>0.199547</td>
      <td>0.269947</td>
      <td>0.368210</td>
      <td>0.267295</td>
      <td>0.223082</td>
      <td>0.350189</td>
      <td>0.216169</td>
      <td>0.317322</td>
      <td>0.473352</td>
    </tr>
    <tr>
      <th>Electropure Sales Corp. v. Foremost Dairies, Inc.</th>
      <td>0.314171</td>
      <td>0.180410</td>
      <td>0.128487</td>
      <td>0.098272</td>
      <td>0.215047</td>
      <td>0.471378</td>
      <td>0.232936</td>
      <td>0.110746</td>
      <td>0.251633</td>
      <td>0.040384</td>
      <td>...</td>
      <td>0.223273</td>
      <td>0.102060</td>
      <td>0.359918</td>
      <td>0.264231</td>
      <td>0.153508</td>
      <td>0.253690</td>
      <td>0.174338</td>
      <td>0.084276</td>
      <td>0.140974</td>
      <td>0.219281</td>
    </tr>
    <tr>
      <th>Leary v. King</th>
      <td>0.298490</td>
      <td>0.111224</td>
      <td>0.239779</td>
      <td>0.060679</td>
      <td>0.070730</td>
      <td>0.470224</td>
      <td>0.229535</td>
      <td>0.086271</td>
      <td>0.166658</td>
      <td>0.102360</td>
      <td>...</td>
      <td>0.226620</td>
      <td>0.125046</td>
      <td>0.180313</td>
      <td>0.472103</td>
      <td>0.323014</td>
      <td>0.200729</td>
      <td>0.286985</td>
      <td>0.228759</td>
      <td>0.398325</td>
      <td>0.530241</td>
    </tr>
    <tr>
      <th>Wolcott v. Shaw</th>
      <td>0.177610</td>
      <td>0.029378</td>
      <td>0.266714</td>
      <td>0.038259</td>
      <td>0.084829</td>
      <td>0.464620</td>
      <td>0.209954</td>
      <td>0.128095</td>
      <td>0.114864</td>
      <td>0.093571</td>
      <td>...</td>
      <td>0.204446</td>
      <td>0.197124</td>
      <td>0.170229</td>
      <td>0.508792</td>
      <td>0.457188</td>
      <td>0.205961</td>
      <td>0.390820</td>
      <td>0.156768</td>
      <td>0.330681</td>
      <td>0.578740</td>
    </tr>
    <tr>
      <th>Forman v. Ford</th>
      <td>0.213042</td>
      <td>0.075746</td>
      <td>0.338021</td>
      <td>0.036992</td>
      <td>0.186203</td>
      <td>0.463439</td>
      <td>0.188739</td>
      <td>0.100435</td>
      <td>0.069244</td>
      <td>0.101619</td>
      <td>...</td>
      <td>0.222654</td>
      <td>0.111791</td>
      <td>0.208148</td>
      <td>0.453624</td>
      <td>0.286863</td>
      <td>0.166817</td>
      <td>0.289075</td>
      <td>0.078810</td>
      <td>0.316915</td>
      <td>0.568895</td>
    </tr>
    <tr>
      <th>In re Estate of Journey</th>
      <td>0.190323</td>
      <td>0.040858</td>
      <td>0.296981</td>
      <td>0.063851</td>
      <td>0.060981</td>
      <td>0.461717</td>
      <td>0.255767</td>
      <td>0.171102</td>
      <td>0.169113</td>
      <td>0.217206</td>
      <td>...</td>
      <td>0.194564</td>
      <td>0.230586</td>
      <td>0.181980</td>
      <td>0.560559</td>
      <td>0.395557</td>
      <td>0.186959</td>
      <td>0.354895</td>
      <td>0.148341</td>
      <td>0.445334</td>
      <td>0.658540</td>
    </tr>
    <tr>
      <th>Reybold v. Reybold</th>
      <td>0.129002</td>
      <td>0.024640</td>
      <td>0.256309</td>
      <td>0.074204</td>
      <td>0.043806</td>
      <td>0.455215</td>
      <td>0.199583</td>
      <td>0.100118</td>
      <td>0.044443</td>
      <td>0.041736</td>
      <td>...</td>
      <td>0.150836</td>
      <td>0.126524</td>
      <td>0.140627</td>
      <td>0.469138</td>
      <td>0.351163</td>
      <td>0.156656</td>
      <td>0.295417</td>
      <td>0.082925</td>
      <td>0.304560</td>
      <td>0.548296</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 2361 columns</p>
</div>



### TFIDF cosine similarity matrix


```python
tfidf_cos_sim = pd.DataFrame(data = cos_sim_tfidf, 
                             columns = df['name_abbreviation'],
                             index = df['name_abbreviation'])
```


```python
tfidf_cos_sim
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
      <th>name_abbreviation</th>
      <th>Dale v. Smith</th>
      <th>Dale v. Smith</th>
      <th>Tatem v. Gilpin</th>
      <th>Woolaston v. Mendenhall</th>
      <th>State v. Gilpin</th>
      <th>Clayton v. Mitchell</th>
      <th>Rodney v. Shankland</th>
      <th>Warner v. Allee</th>
      <th>Philip v. Wood</th>
      <th>Thompson v. Lynam</th>
      <th>...</th>
      <th>Slaughter v. Moore</th>
      <th>Walter v. Peninsula Cut Stone Co.</th>
      <th>Williamson v. McMonagle</th>
      <th>Emmons v. Curlett</th>
      <th>Jacobs v. Wilmington Trust Co.</th>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <th>Dayett v. Willitts</th>
      <th>In re McFarlin</th>
      <th>In re the Real Estate of Donaghy</th>
      <th>In re Tomlinson</th>
    </tr>
    <tr>
      <th>name_abbreviation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dale v. Smith</th>
      <td>1.000000</td>
      <td>0.276312</td>
      <td>0.135193</td>
      <td>0.092811</td>
      <td>0.112932</td>
      <td>0.097511</td>
      <td>0.265474</td>
      <td>0.097482</td>
      <td>0.416723</td>
      <td>0.163832</td>
      <td>...</td>
      <td>0.118707</td>
      <td>0.056855</td>
      <td>0.151287</td>
      <td>0.265400</td>
      <td>0.100853</td>
      <td>0.093248</td>
      <td>0.121720</td>
      <td>0.106331</td>
      <td>0.244379</td>
      <td>0.145211</td>
    </tr>
    <tr>
      <th>Dale v. Smith</th>
      <td>0.276312</td>
      <td>1.000000</td>
      <td>0.085617</td>
      <td>0.086401</td>
      <td>0.095099</td>
      <td>0.042488</td>
      <td>0.128184</td>
      <td>0.060454</td>
      <td>0.103198</td>
      <td>0.040434</td>
      <td>...</td>
      <td>0.088963</td>
      <td>0.023779</td>
      <td>0.068625</td>
      <td>0.037353</td>
      <td>0.023675</td>
      <td>0.035132</td>
      <td>0.066241</td>
      <td>0.039022</td>
      <td>0.036230</td>
      <td>0.059313</td>
    </tr>
    <tr>
      <th>Tatem v. Gilpin</th>
      <td>0.135193</td>
      <td>0.085617</td>
      <td>1.000000</td>
      <td>0.298718</td>
      <td>0.295520</td>
      <td>0.187345</td>
      <td>0.141842</td>
      <td>0.123914</td>
      <td>0.070201</td>
      <td>0.308958</td>
      <td>...</td>
      <td>0.156250</td>
      <td>0.088286</td>
      <td>0.367342</td>
      <td>0.133349</td>
      <td>0.094900</td>
      <td>0.140901</td>
      <td>0.117562</td>
      <td>0.113938</td>
      <td>0.139185</td>
      <td>0.211895</td>
    </tr>
    <tr>
      <th>Woolaston v. Mendenhall</th>
      <td>0.092811</td>
      <td>0.086401</td>
      <td>0.298718</td>
      <td>1.000000</td>
      <td>0.108586</td>
      <td>0.100957</td>
      <td>0.095666</td>
      <td>0.065325</td>
      <td>0.062108</td>
      <td>0.079158</td>
      <td>...</td>
      <td>0.069573</td>
      <td>0.054765</td>
      <td>0.141970</td>
      <td>0.071925</td>
      <td>0.078851</td>
      <td>0.149798</td>
      <td>0.105568</td>
      <td>0.058778</td>
      <td>0.058825</td>
      <td>0.068174</td>
    </tr>
    <tr>
      <th>State v. Gilpin</th>
      <td>0.112932</td>
      <td>0.095099</td>
      <td>0.295520</td>
      <td>0.108586</td>
      <td>1.000000</td>
      <td>0.131660</td>
      <td>0.128040</td>
      <td>0.133150</td>
      <td>0.087910</td>
      <td>0.289919</td>
      <td>...</td>
      <td>0.101633</td>
      <td>0.076165</td>
      <td>0.239578</td>
      <td>0.090137</td>
      <td>0.066383</td>
      <td>0.116288</td>
      <td>0.081193</td>
      <td>0.087294</td>
      <td>0.079234</td>
      <td>0.095589</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <td>0.093248</td>
      <td>0.035132</td>
      <td>0.140901</td>
      <td>0.149798</td>
      <td>0.116288</td>
      <td>0.171463</td>
      <td>0.194953</td>
      <td>0.140388</td>
      <td>0.101209</td>
      <td>0.101619</td>
      <td>...</td>
      <td>0.462121</td>
      <td>0.415066</td>
      <td>0.124137</td>
      <td>0.139059</td>
      <td>0.215696</td>
      <td>1.000000</td>
      <td>0.178517</td>
      <td>0.167448</td>
      <td>0.124241</td>
      <td>0.121509</td>
    </tr>
    <tr>
      <th>Dayett v. Willitts</th>
      <td>0.121720</td>
      <td>0.066241</td>
      <td>0.117562</td>
      <td>0.105568</td>
      <td>0.081193</td>
      <td>0.171296</td>
      <td>0.228375</td>
      <td>0.190934</td>
      <td>0.120657</td>
      <td>0.191833</td>
      <td>...</td>
      <td>0.133347</td>
      <td>0.338720</td>
      <td>0.101977</td>
      <td>0.238325</td>
      <td>0.196623</td>
      <td>0.178517</td>
      <td>1.000000</td>
      <td>0.166350</td>
      <td>0.146195</td>
      <td>0.250501</td>
    </tr>
    <tr>
      <th>In re McFarlin</th>
      <td>0.106331</td>
      <td>0.039022</td>
      <td>0.113938</td>
      <td>0.058778</td>
      <td>0.087294</td>
      <td>0.079552</td>
      <td>0.218451</td>
      <td>0.143612</td>
      <td>0.144219</td>
      <td>0.150529</td>
      <td>...</td>
      <td>0.125824</td>
      <td>0.187866</td>
      <td>0.185141</td>
      <td>0.161146</td>
      <td>0.230827</td>
      <td>0.167448</td>
      <td>0.166350</td>
      <td>1.000000</td>
      <td>0.127811</td>
      <td>0.184411</td>
    </tr>
    <tr>
      <th>In re the Real Estate of Donaghy</th>
      <td>0.244379</td>
      <td>0.036230</td>
      <td>0.139185</td>
      <td>0.058825</td>
      <td>0.079234</td>
      <td>0.125207</td>
      <td>0.157789</td>
      <td>0.104890</td>
      <td>0.242881</td>
      <td>0.318390</td>
      <td>...</td>
      <td>0.114273</td>
      <td>0.154284</td>
      <td>0.153437</td>
      <td>0.355135</td>
      <td>0.110276</td>
      <td>0.124241</td>
      <td>0.146195</td>
      <td>0.127811</td>
      <td>1.000000</td>
      <td>0.286089</td>
    </tr>
    <tr>
      <th>In re Tomlinson</th>
      <td>0.145211</td>
      <td>0.059313</td>
      <td>0.211895</td>
      <td>0.068174</td>
      <td>0.095589</td>
      <td>0.203946</td>
      <td>0.178793</td>
      <td>0.099603</td>
      <td>0.126978</td>
      <td>0.255136</td>
      <td>...</td>
      <td>0.118503</td>
      <td>0.120719</td>
      <td>0.149596</td>
      <td>0.224100</td>
      <td>0.138372</td>
      <td>0.121509</td>
      <td>0.250501</td>
      <td>0.184411</td>
      <td>0.286089</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>2361 rows × 2361 columns</p>
</div>




```python
tfidf_cos_sim.sort_values(by='Clayton v. Mitchell', 
                          ascending=False).head(15)


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
      <th>name_abbreviation</th>
      <th>Dale v. Smith</th>
      <th>Dale v. Smith</th>
      <th>Tatem v. Gilpin</th>
      <th>Woolaston v. Mendenhall</th>
      <th>State v. Gilpin</th>
      <th>Clayton v. Mitchell</th>
      <th>Rodney v. Shankland</th>
      <th>Warner v. Allee</th>
      <th>Philip v. Wood</th>
      <th>Thompson v. Lynam</th>
      <th>...</th>
      <th>Slaughter v. Moore</th>
      <th>Walter v. Peninsula Cut Stone Co.</th>
      <th>Williamson v. McMonagle</th>
      <th>Emmons v. Curlett</th>
      <th>Jacobs v. Wilmington Trust Co.</th>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <th>Dayett v. Willitts</th>
      <th>In re McFarlin</th>
      <th>In re the Real Estate of Donaghy</th>
      <th>In re Tomlinson</th>
    </tr>
    <tr>
      <th>name_abbreviation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Clayton v. Mitchell</th>
      <td>0.097511</td>
      <td>0.042488</td>
      <td>0.187345</td>
      <td>0.100957</td>
      <td>0.131660</td>
      <td>1.000000</td>
      <td>0.269079</td>
      <td>0.141190</td>
      <td>0.153409</td>
      <td>0.101605</td>
      <td>...</td>
      <td>0.240680</td>
      <td>0.083957</td>
      <td>0.246097</td>
      <td>0.213884</td>
      <td>0.133136</td>
      <td>0.171463</td>
      <td>0.171296</td>
      <td>0.079552</td>
      <td>0.125207</td>
      <td>0.203946</td>
    </tr>
    <tr>
      <th>Owens v. Owens</th>
      <td>0.074230</td>
      <td>0.046287</td>
      <td>0.178056</td>
      <td>0.066474</td>
      <td>0.042645</td>
      <td>0.484794</td>
      <td>0.245852</td>
      <td>0.195312</td>
      <td>0.077176</td>
      <td>0.077038</td>
      <td>...</td>
      <td>0.135383</td>
      <td>0.156537</td>
      <td>0.114789</td>
      <td>0.296380</td>
      <td>0.312981</td>
      <td>0.148989</td>
      <td>0.247492</td>
      <td>0.122612</td>
      <td>0.163089</td>
      <td>0.267193</td>
    </tr>
    <tr>
      <th>Hayes v. Hayes</th>
      <td>0.089733</td>
      <td>0.037919</td>
      <td>0.257135</td>
      <td>0.172667</td>
      <td>0.142270</td>
      <td>0.457975</td>
      <td>0.184558</td>
      <td>0.142782</td>
      <td>0.121234</td>
      <td>0.113258</td>
      <td>...</td>
      <td>0.128584</td>
      <td>0.156854</td>
      <td>0.175658</td>
      <td>0.226919</td>
      <td>0.170509</td>
      <td>0.154500</td>
      <td>0.252311</td>
      <td>0.143085</td>
      <td>0.212837</td>
      <td>0.273952</td>
    </tr>
    <tr>
      <th>Gamble v. Harris</th>
      <td>0.154412</td>
      <td>0.087277</td>
      <td>0.126098</td>
      <td>0.043562</td>
      <td>0.108685</td>
      <td>0.457090</td>
      <td>0.290703</td>
      <td>0.176022</td>
      <td>0.142940</td>
      <td>0.145114</td>
      <td>...</td>
      <td>0.147961</td>
      <td>0.212768</td>
      <td>0.142544</td>
      <td>0.196637</td>
      <td>0.130253</td>
      <td>0.128100</td>
      <td>0.268221</td>
      <td>0.120115</td>
      <td>0.239255</td>
      <td>0.224501</td>
    </tr>
    <tr>
      <th>Burton v. Willen</th>
      <td>0.230941</td>
      <td>0.154361</td>
      <td>0.229286</td>
      <td>0.083322</td>
      <td>0.133816</td>
      <td>0.431018</td>
      <td>0.397008</td>
      <td>0.286245</td>
      <td>0.248788</td>
      <td>0.104328</td>
      <td>...</td>
      <td>0.352527</td>
      <td>0.198211</td>
      <td>0.191565</td>
      <td>0.376396</td>
      <td>0.331449</td>
      <td>0.250787</td>
      <td>0.273899</td>
      <td>0.223246</td>
      <td>0.209291</td>
      <td>0.370056</td>
    </tr>
    <tr>
      <th>Kempski v. Leszczynski</th>
      <td>0.142850</td>
      <td>0.025228</td>
      <td>0.050302</td>
      <td>0.031891</td>
      <td>0.029384</td>
      <td>0.419086</td>
      <td>0.195442</td>
      <td>0.027438</td>
      <td>0.142192</td>
      <td>0.117320</td>
      <td>...</td>
      <td>0.080092</td>
      <td>0.034642</td>
      <td>0.061467</td>
      <td>0.188518</td>
      <td>0.067577</td>
      <td>0.037110</td>
      <td>0.070516</td>
      <td>0.064475</td>
      <td>0.302551</td>
      <td>0.109408</td>
    </tr>
    <tr>
      <th>Cannon v. Hudson</th>
      <td>0.126065</td>
      <td>0.020422</td>
      <td>0.267698</td>
      <td>0.039617</td>
      <td>0.088360</td>
      <td>0.416681</td>
      <td>0.248896</td>
      <td>0.138339</td>
      <td>0.110487</td>
      <td>0.183791</td>
      <td>...</td>
      <td>0.124637</td>
      <td>0.172494</td>
      <td>0.206951</td>
      <td>0.344684</td>
      <td>0.190451</td>
      <td>0.147058</td>
      <td>0.318134</td>
      <td>0.093276</td>
      <td>0.246310</td>
      <td>0.394177</td>
    </tr>
    <tr>
      <th>Hutchison v. Roberts</th>
      <td>0.087935</td>
      <td>0.034643</td>
      <td>0.122177</td>
      <td>0.027231</td>
      <td>0.049416</td>
      <td>0.414534</td>
      <td>0.361063</td>
      <td>0.196045</td>
      <td>0.293914</td>
      <td>0.097805</td>
      <td>...</td>
      <td>0.180005</td>
      <td>0.174903</td>
      <td>0.124856</td>
      <td>0.328562</td>
      <td>0.210622</td>
      <td>0.099427</td>
      <td>0.264210</td>
      <td>0.117086</td>
      <td>0.145364</td>
      <td>0.300454</td>
    </tr>
    <tr>
      <th>Parker v. Yerger</th>
      <td>0.202842</td>
      <td>0.064103</td>
      <td>0.184657</td>
      <td>0.068809</td>
      <td>0.057861</td>
      <td>0.409092</td>
      <td>0.184913</td>
      <td>0.035552</td>
      <td>0.202089</td>
      <td>0.119305</td>
      <td>...</td>
      <td>0.154639</td>
      <td>0.129849</td>
      <td>0.174627</td>
      <td>0.345681</td>
      <td>0.164208</td>
      <td>0.158186</td>
      <td>0.184969</td>
      <td>0.043798</td>
      <td>0.351410</td>
      <td>0.395671</td>
    </tr>
    <tr>
      <th>Agostini v. Colonial Trust Co.</th>
      <td>0.109672</td>
      <td>0.020871</td>
      <td>0.153893</td>
      <td>0.071611</td>
      <td>0.164226</td>
      <td>0.394943</td>
      <td>0.133511</td>
      <td>0.170558</td>
      <td>0.175207</td>
      <td>0.093044</td>
      <td>...</td>
      <td>0.150765</td>
      <td>0.102212</td>
      <td>0.385845</td>
      <td>0.166821</td>
      <td>0.101174</td>
      <td>0.120694</td>
      <td>0.225687</td>
      <td>0.065934</td>
      <td>0.103912</td>
      <td>0.162314</td>
    </tr>
    <tr>
      <th>Denash v. Security Trust &amp; Safe Deposit Co.</th>
      <td>0.200080</td>
      <td>0.222772</td>
      <td>0.056087</td>
      <td>0.042173</td>
      <td>0.088638</td>
      <td>0.391788</td>
      <td>0.235927</td>
      <td>0.100664</td>
      <td>0.226905</td>
      <td>0.065849</td>
      <td>...</td>
      <td>0.226449</td>
      <td>0.073096</td>
      <td>0.145284</td>
      <td>0.124447</td>
      <td>0.152551</td>
      <td>0.145540</td>
      <td>0.071443</td>
      <td>0.109677</td>
      <td>0.076704</td>
      <td>0.193686</td>
    </tr>
    <tr>
      <th>In re Estate of Journey</th>
      <td>0.142313</td>
      <td>0.043691</td>
      <td>0.195763</td>
      <td>0.057257</td>
      <td>0.040626</td>
      <td>0.371712</td>
      <td>0.246092</td>
      <td>0.161996</td>
      <td>0.168236</td>
      <td>0.241165</td>
      <td>...</td>
      <td>0.156233</td>
      <td>0.174353</td>
      <td>0.135107</td>
      <td>0.412986</td>
      <td>0.337317</td>
      <td>0.139253</td>
      <td>0.234548</td>
      <td>0.147849</td>
      <td>0.375184</td>
      <td>0.469764</td>
    </tr>
    <tr>
      <th>Electropure Sales Corp. v. Foremost Dairies, Inc.</th>
      <td>0.286986</td>
      <td>0.166884</td>
      <td>0.110751</td>
      <td>0.065892</td>
      <td>0.139784</td>
      <td>0.355808</td>
      <td>0.205987</td>
      <td>0.102653</td>
      <td>0.229845</td>
      <td>0.039429</td>
      <td>...</td>
      <td>0.210787</td>
      <td>0.073032</td>
      <td>0.266624</td>
      <td>0.183443</td>
      <td>0.110401</td>
      <td>0.200125</td>
      <td>0.129024</td>
      <td>0.074268</td>
      <td>0.098537</td>
      <td>0.134063</td>
    </tr>
    <tr>
      <th>Reybold v. Reybold</th>
      <td>0.077336</td>
      <td>0.019132</td>
      <td>0.173396</td>
      <td>0.094127</td>
      <td>0.036695</td>
      <td>0.354173</td>
      <td>0.186002</td>
      <td>0.120511</td>
      <td>0.056311</td>
      <td>0.034136</td>
      <td>...</td>
      <td>0.126279</td>
      <td>0.093063</td>
      <td>0.114446</td>
      <td>0.295367</td>
      <td>0.262326</td>
      <td>0.128099</td>
      <td>0.175434</td>
      <td>0.080139</td>
      <td>0.185284</td>
      <td>0.331260</td>
    </tr>
    <tr>
      <th>Pickering v. Day</th>
      <td>0.227865</td>
      <td>0.148615</td>
      <td>0.128182</td>
      <td>0.078752</td>
      <td>0.161543</td>
      <td>0.350399</td>
      <td>0.246400</td>
      <td>0.211389</td>
      <td>0.383099</td>
      <td>0.066148</td>
      <td>...</td>
      <td>0.332404</td>
      <td>0.138095</td>
      <td>0.235864</td>
      <td>0.186205</td>
      <td>0.174944</td>
      <td>0.259677</td>
      <td>0.210147</td>
      <td>0.109112</td>
      <td>0.122325</td>
      <td>0.227433</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 2361 columns</p>
</div>



Notice, there's a bit of a difference between TF and TFIDF similarity scores. 

Let's actually read those "similar cases" to Clayton v. Mitchell - are they actually similar? 

### Finding similar case


```python
def get_index(case):    
    return df.name_abbreviation[df.name_abbreviation == case].index.tolist()[0]     
```


```python
case_of_interest = 'Clayton v. Mitchell'
source_idx = get_index(case_of_interest)

# Print the Original Case for context
print(f"CASE OF INTEREST: {case_of_interest}")
if isinstance(source_idx, int):
    print(df.at[source_idx, 'text'][:1500] + "...") # Showing a bit more for context
print("\n" )


# Run the Similarity Comparison Loop
matrices = [
    ("CountVectorizer", cv_cos_sim),
    ("TF-IDF", tfidf_cos_sim)
]

for label, sim_df in matrices:
    # Find the name and score of the most similar case
    # .index[1] skips the case itself
    most_similar_name = sim_df[case_of_interest].sort_values(ascending=False).index[1]
    score = sim_df[case_of_interest].sort_values(ascending=False).values[1]
    
    similar_idx = get_index(most_similar_name)
    
    print(f"Representation: {label}")
    print(f"MOST SIMILAR CASE: {most_similar_name}")
    print(f"SIMILARITY SCORE: {score:.4f}")
    print("-" * 30)
    
    if isinstance(similar_idx, int):
        text_snippet = df.at[similar_idx, 'text'][:1000] + "..."
        print(text_snippet)
    print("\n")
```

    CASE OF INTEREST: Clayton v. Mitchell
    Bidgely, Chancellor.
    The affidavit states that John Mitchell is about to depart the State and to reside .in the State of Maryland, with the view and intention of avoiding the payment of such sum of money as may be decreed to the complainants ; that the said John Mitchell has declared his intention to remove out of the State ; that he has purchased a large tract of land in the State of Maryland, as this deponent has heard and believes ; that the said John Mitchell has sold and disposed of his share of the mill in the bill mentioned ; and that this deponent verily believes that the said John Mitchell is indebted to the complainant on account of the matter charged in the said bill, in the sum of $16,000 and upwards, and that if the said John Mitchell is permitted to leave the State, the complainants are in danger of losing their just demands :
    There are two objections to this affidavit.
    First, that the affidavit does not sufficiently set out the facts or state the circumstances on which it is grounded ; and,
    Second, that the debt is not positively sworn to.
    As to the first objection, the affidavit refers to the bill and to the matters charged in the bill to show the demand of the complainant or the ground of his complaint. If the complainants shall make out their case, then they will be entitled to relief; and consequently, those facts or circumstances referred to by the affidavit state sufficiently or lay a sufficient ground of complaint, which does not rest on the opinion of t...
    
    
    Representation: CountVectorizer
    MOST SIMILAR CASE: Parker v. Yerger
    SIMILARITY SCORE: 0.5487
    ------------------------------
    Wolcott, Chancellor.
    Let the decree be entered as follows:
    And now, to-wit, this 25th day of February, A. D. 1895, the above-stated cause having come on to be heard upon bill and answer, and the arguments of counsel having been made thereon, and it appearing to the court that the said James Parker, said complainant, is seized of, in and to the said lands and premises, in said bill of complaint described, in his demesne as of fee, and that he can convey a clear and unincumbered title thereto, it is hereby ordered, adjudged and decreed that the said Hiram Yerger, said defendant, specifically perform the said agreement for the sale of the lands and premises in said bill of complaint set forth, and that said defendant pay unto the said complainant the remainder of said purchase money in said agreement for sale mentioned; and it is further ordered and decreed that upon the payment of the whole of said purchase money by said defendant to said complainant, the said complainant shall make, exe...
    
    
    Representation: TF-IDF
    MOST SIMILAR CASE: Owens v. Owens
    SIMILARITY SCORE: 0.4848
    ------------------------------
    Seitz, Chancellor:
    Plaintiff asks to have this court eject her husband from the furnished apartment last used by them as the marital home; she having moved therefrom. This apartment is one of three apartments in the Triplex Apartment house. She also seeks to prevent him from using a portion of the garage for business storage purposes, which garage is a part of the so-called Duplex Apartment house. Finally, she wants this court to restrain defendant from interfering with her. apartment rental business. The real property and furnishings involved are “owned” solely by plaintiff.
    Defendant does not challenge this court’s jurisdiction to entertain this action which sounds in equitable ejectment. However, defendant insists that plaintiff has failed to establish her right to have him ejected from the marital apartment or from the use of a portion of the garage. The defendant also counterclaimed against plaintiff for certain sums of money on the basis of work done for or money advanced to or f...
    
    


Both seem to be talking about real estate?

Let's try another case called "Braasch v. Galdi Securities Corp.". 

In theory, if similarity metrics are correct, we should get some other corporate case.


```python
case_of_interest = 'Braasch v. Galdi Securities Corp.'
source_idx = get_index(case_of_interest)

# Print the Original Case for context
print(f"CASE OF INTEREST: {case_of_interest}")
if isinstance(source_idx, int):
    print(df.at[source_idx, 'text'][:1500] + "...") # Showing a bit more for context
print("\n" )


# Run the Similarity Comparison Loop
matrices = [
    ("CountVectorizer", cv_cos_sim),
    ("TF-IDF", tfidf_cos_sim)
]

for label, sim_df in matrices:
    # Find the name and score of the most similar case
    # .index[1] skips the case itself
    most_similar_name = sim_df[case_of_interest].sort_values(ascending=False).index[1]  ## change to see 2nd most similar, etc
    score = sim_df[case_of_interest].sort_values(ascending=False).values[1]
    
    similar_idx = get_index(most_similar_name)
    
    print(f"Representation: {label}")
    print(f"MOST SIMILAR CASE: {most_similar_name}")
    print(f"SIMILARITY SCORE: {score:.4f}")
    print("-" * 30)
    
    if isinstance(similar_idx, int):
        text_snippet = df.at[similar_idx, 'text'][:1000] + "..."
        print(text_snippet)
    print("\n")
```

    CASE OF INTEREST: Braasch v. Galdi Securities Corp.
    Short, Vice Chancellor:
    This case is before the court on defendants’ motion to dismiss for failure to state a claim on which relief can be granted.
    Plaintiffs are the owners of 5400 shares of the common stock of American Sumatra Tobacco Corporation (American Sumatra), a Delaware corporation. They here sue (1) individually on their own behalf; (2) representatively on behalf of all other stockholders of the corporation similarly situated, including those who sold their shares to the defendant N. V. Deli Maatschappij (Deli), a corporation of the Kingdom of the Netherlands, pursuant to an offer to buy made to all common stockholders; and (3) derivatively on behalf of American Sumatra.
    On June 28, 1960, Deli was the owner of more than fifty per cent of the common stock of American Sumatra. On that date it made an offer to all other stockholders to buy 202,338 shares of the common stock of American Sumatra at $17 per share. The offer resulted in Deli acquiring in excess of 200,000 additional shares of American Sumatra and increasing its stock ownership to more than ninety per cent of the outstanding shares of the company. On October 21, 1960 Deli organized, as its wholly owned subsidiary, Tobacco Holdings, Inc., a Delaware corporation, and thereupon transferred to Tobacco Holdings, Inc. all its shares of American Sumatra. In November, 1960 American Sumatra was merged into Tobacco Holdings, Inc. pursuant to the provisions of 8 Del.C. § 253. Immediately thereafter, the name of Tobacc...
    
    
    Representation: CountVectorizer
    MOST SIMILAR CASE: Abelow v. Symonds
    SIMILARITY SCORE: 0.6984
    ------------------------------
    Marvel, Vice Chancellor:
    This action by holders of common stock of Midstates Oil Corporation as originally designed sought an order restraining the proposed sale by that corporation of its assets and properties to the defendant, Middle States Petroleum Corporation, the owner of 95.93% of the common stock of Midstates. The original complaint was filed on December 29, 1958, the eve of a stockholders’ meeting called to approve such proposed sale of assets and consequent liquidation of Midstates under a plan which provided for the payment of $1,125 for each share of such stock to be surrendered under the plan. Also named as defendants in the action were present and former officers and directors of Midstates and Middle States, and Tennessee Gas Transmission Company, a corporation allegedly in control of Middle States which, according to the complaint, had through an exercise of such control brought about an exchange of stock whereby Tennessee Gas had become the holder of 92% of the outstand...
    
    
    Representation: TF-IDF
    MOST SIMILAR CASE: Abelow v. Symonds
    SIMILARITY SCORE: 0.6651
    ------------------------------
    Marvel, Vice Chancellor:
    This action by holders of common stock of Midstates Oil Corporation as originally designed sought an order restraining the proposed sale by that corporation of its assets and properties to the defendant, Middle States Petroleum Corporation, the owner of 95.93% of the common stock of Midstates. The original complaint was filed on December 29, 1958, the eve of a stockholders’ meeting called to approve such proposed sale of assets and consequent liquidation of Midstates under a plan which provided for the payment of $1,125 for each share of such stock to be surrendered under the plan. Also named as defendants in the action were present and former officers and directors of Midstates and Middle States, and Tennessee Gas Transmission Company, a corporation allegedly in control of Middle States which, according to the complaint, had through an exercise of such control brought about an exchange of stock whereby Tennessee Gas had become the holder of 92% of the outstand...
    
    


They both seem to be talking about common stock, etc. 

Seems like cosine similarity is working. 

Note also that both representations give the same result. 

__NOTE:__  Recall that we could also represent documents as "topics" rather than words - this can also be used for cosine similarity purposes.

## word2vec: some theory
Now that we know what cosine similarity is  we can move on to Word2Vec. 

The theory behind the word2vec algorithm relies on two theoretical foundations - 

1. **[Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics),** - ie that "meaning" of words (semantics) is known by the context in which the word is used.


2. **[Language Modeling](https://thegradient.pub/content/images/2019/10/lm-1.png)** - a task in NLP where you predict the next word in a sequence based on probabilities

![image](https://thegradient.pub/content/images/2019/10/lm-1.png)

### Distributional Semantics 

Word2vec algorithm is based on a Linguistic theory called "distributional hypothesis". 

This theory can be summerized by the linguist Firth's famous statement that **You shall know a word by the company it keeps**. This should not be surprising to anyone who encountered a weird word in a book - we usually tend to re-read the sentence and look for other words around the word we don't know. __Thus, we kinda get an idea of what the word is by looking at other words around it.__ 

Distributional hypothesis suggest that:

* meaning isn't a fixed/static definition inside a dictionary, but rather a result of how a word is "distributed" across a body of text.
* The distributional hypothesis very convenient for Computer Scienctists because it has a solid statistical basis - if the words "coffee" and "tea" both frequently appear near words like "drink," "cup," "hot," and "morning," a computer can mathematically determine they are related, even if it has no idea what a liquid is.
* The other theory of meaning is known as "formal semantics" - which is based studying the the relationship between a word and the world, where a word is a  "symbol" that points to a specific object or set. Meaning of a word is thus a "denotation" (the object/set of objects that the symbol points to) in the real world. Thus, a dog is a set of objects 

For an example of how distributional hypothesis works in the real world, consider Lewis Carroll's poem [Jobberwocky](https://www.poetryfoundation.org/poems/42916/jabberwocky).

### Language modeling as a Cloze reading comprehension task/test

Distributional semantics thus assumes a **distributional hypothesis**. In simple terms, distributional hypothesis argues that the usage of words is **a distribution.** Not only that, but the distribution is constrained/changes depending on various contexts. What this means is that there's the probability of a word appearing __depends on a given context.__

#### Consider the following examples  of constraining the distirbution of words:

1) **You are a ___________** 

How many words can logically fill in the blank here? ie what is the **"distribution"**, probability wise, what words can fit here?


2) **I am typing on a _____** 

How many words can fit in the blank space? - probably only a couple - a typewriter/computer/my phone. The distribution of words here is more constrained. We can also logically conclude that the word definitely has to be a noun.

3) **I like drinking ____** 

How many words can fit in the blank space?

4) **I like drinking  ____, very hot** 

Compared to the previous example, less words can fit here - for example, we can't talk about alcoholic drinks any more, unless you drink it very hot.

* Thus, words that co-occur in the same context have similar meanings/functions/usages (you can't drink a "table" for example - our language is logically constrained and the context determines the word that you we would use).
* (By the way, recall, that we did something similar in the kindgergarden known as the "Cloze Test" - this is how basic this stuff is).

**This "fill in the blank exercise" above, when done by computers is called a "language modeling task"**

<div align="center">
<img src="https://miro.medium.com/max/1400/1*_MrDp6w3Xc-yLuCTbco0xw.png" width="80%" />
</div>


###  Language modeling with word2vec

Having established that the "fill in the blanks" exercise that we did in the kindergarden is unironically the way language modeling works we can move on to the actual algorithm architecture. 


There are two model architectures for word2vec:

* The first of the architectures is called __"continious bag of words"__ (CBOW) which predicts the **current word based on the context** (ie a word is blanked out and the algorithm looks at the data to see which words fit best/highest probability), 

* The second architecture, __"Skip-gram"__ (SGNS) predicts __surrounding words given the current word__ (it's literally  called **"skipgram"** ie - you "skip" an "ngram").

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:678/1*KrMpN-9V1mJRiOIHBLrGfw.png" width="80%" />
</div>

### Word2vec (and language modeling) as Self-supervised learning

Conceptually, the Word2Vec algorithm is a **supervised prediction task** using only raw text.

In CBOW:
* Input: the context words surrounding a target word.

* Output: the target word itself, represented as a probability distribution over the vocabulary. Higher probability words are more likely to "fill in the blank"

* During training, the model learns to assign a high probability to the **correct target word** (label = 1) and low probabilities to other words (label = 0).

Since the labels are derived automatically from the text, Word2Vec is thus **self-supervised**: it uses the structure of language to create its own training signal without requiring manually labeled data.

During training, the model learns to assign a high probability to the correct target word (label = 1) and low probabilities to other words (label = 0).
you use as input the context word vectors (from the DTM) and for output, you know what the word is - and you use this to update the values in the __"hidden layer"__ - which subsequently becomes our "word embedding" - ie a dense representation of a word (rather than sparse). The vectors __keep updating__ until the algorithm gets matching predictions for the output word from the context words (in the case of CBOW). 


When trained on very large corpora (like all of English Wikipedia) it can capture very interesting analogies and relationships such as finding that the vector closest to the output of the operation 'king' - 'man' + 'woman' is 'queen' 

<div align="center">
<img src="https://static.packt-cdn.com/products/9781787287600/graphics/d4b8d439-e136-44f7-895d-71de1d84342c.png" width="80%" />
</div>


This is precisely the "analogical reasoning" that we humans are so good at. 

And remember - __analogy is one of the key tasks of a judge__ 

In essence - word2vec can capture "conceptual relationships" using simple linear algebra - after learning the vectors from the data.




In both CBOW and SGNS You set the window size (context size around a target word) and the algorithm does this for all the words. 

Recall that theoretically speaking when we learn word vector representations via context information - as humans, we kinda do **the same thing as a concordances,** only on a much larger scale.  One can say that word2vec is actually fundamentally based on concordances.

<div align="center">
<img src="https://www.researchgate.net/profile/Kwabena-Sarfo-Kantankah/publication/337577141/figure/fig1/AS:830626072633345@1575048096309/Sample-concordance-lines-of-actually.png" width="80%" />
</div>


```python
# Continous bag of words example
# Our raw text data
example_sentences = [
    "the chef cooked a delicious meal",
    "the baker baked fresh bread",
    "the waiter served the cold drinks",
    "the customer ate a tasty sandwich",
    "the cat chased the small mouse",
    "the dog barked at the mailman",
    "the sun rose in the morning",
    "the moon shone in the night",
    "the gardener planted a red rose",
    "the boy played with a blue ball"
]

sentences = []
sentences += [s.split() for s in example_sentences]
```


```python

```


```python
window_size = 2 # Adjusted to 2 for a 5-word sentence
data = []

for sentence in sentences:
    for i, target_word in enumerate(sentence):
        # Identify the context (all neighbors within the window)
        context = []
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(sentence))):
            if i != j:                # Don't include the target word itself in the context
                context.append(sentence[j])
        
        # Skip if the context is empty (usually at start/end of tiny sentences)
        if context:
            # CBOW: Context (Input) -> Target Word (Output)
            data.append([context, target_word])

# Representing as the Input/Output Pandas DataFrame
df_cbow = pd.DataFrame(data, columns=['Input (Context Words)', 'Output (Target Word)'])

df_cbow
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
      <th>Input (Context Words)</th>
      <th>Output (Target Word)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[chef, cooked]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[the, cooked, a]</td>
      <td>chef</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[the, chef, a, delicious]</td>
      <td>cooked</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[chef, cooked, delicious, meal]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[cooked, a, meal]</td>
      <td>delicious</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[a, delicious]</td>
      <td>meal</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[baker, baked]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[the, baked, fresh]</td>
      <td>baker</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[the, baker, fresh, bread]</td>
      <td>baked</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[baker, baked, bread]</td>
      <td>fresh</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[baked, fresh]</td>
      <td>bread</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[waiter, served]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[the, served, the]</td>
      <td>waiter</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[the, waiter, the, cold]</td>
      <td>served</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[waiter, served, cold, drinks]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[served, the, drinks]</td>
      <td>cold</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[the, cold]</td>
      <td>drinks</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[customer, ate]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[the, ate, a]</td>
      <td>customer</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[the, customer, a, tasty]</td>
      <td>ate</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[customer, ate, tasty, sandwich]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[ate, a, sandwich]</td>
      <td>tasty</td>
    </tr>
    <tr>
      <th>22</th>
      <td>[a, tasty]</td>
      <td>sandwich</td>
    </tr>
    <tr>
      <th>23</th>
      <td>[cat, chased]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[the, chased, the]</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>25</th>
      <td>[the, cat, the, small]</td>
      <td>chased</td>
    </tr>
    <tr>
      <th>26</th>
      <td>[cat, chased, small, mouse]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>27</th>
      <td>[chased, the, mouse]</td>
      <td>small</td>
    </tr>
    <tr>
      <th>28</th>
      <td>[the, small]</td>
      <td>mouse</td>
    </tr>
    <tr>
      <th>29</th>
      <td>[dog, barked]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>30</th>
      <td>[the, barked, at]</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>31</th>
      <td>[the, dog, at, the]</td>
      <td>barked</td>
    </tr>
    <tr>
      <th>32</th>
      <td>[dog, barked, the, mailman]</td>
      <td>at</td>
    </tr>
    <tr>
      <th>33</th>
      <td>[barked, at, mailman]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>34</th>
      <td>[at, the]</td>
      <td>mailman</td>
    </tr>
    <tr>
      <th>35</th>
      <td>[sun, rose]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>36</th>
      <td>[the, rose, in]</td>
      <td>sun</td>
    </tr>
    <tr>
      <th>37</th>
      <td>[the, sun, in, the]</td>
      <td>rose</td>
    </tr>
    <tr>
      <th>38</th>
      <td>[sun, rose, the, morning]</td>
      <td>in</td>
    </tr>
    <tr>
      <th>39</th>
      <td>[rose, in, morning]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>40</th>
      <td>[in, the]</td>
      <td>morning</td>
    </tr>
    <tr>
      <th>41</th>
      <td>[moon, shone]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>42</th>
      <td>[the, shone, in]</td>
      <td>moon</td>
    </tr>
    <tr>
      <th>43</th>
      <td>[the, moon, in, the]</td>
      <td>shone</td>
    </tr>
    <tr>
      <th>44</th>
      <td>[moon, shone, the, night]</td>
      <td>in</td>
    </tr>
    <tr>
      <th>45</th>
      <td>[shone, in, night]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>46</th>
      <td>[in, the]</td>
      <td>night</td>
    </tr>
    <tr>
      <th>47</th>
      <td>[gardener, planted]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>48</th>
      <td>[the, planted, a]</td>
      <td>gardener</td>
    </tr>
    <tr>
      <th>49</th>
      <td>[the, gardener, a, red]</td>
      <td>planted</td>
    </tr>
    <tr>
      <th>50</th>
      <td>[gardener, planted, red, rose]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>51</th>
      <td>[planted, a, rose]</td>
      <td>red</td>
    </tr>
    <tr>
      <th>52</th>
      <td>[a, red]</td>
      <td>rose</td>
    </tr>
    <tr>
      <th>53</th>
      <td>[boy, played]</td>
      <td>the</td>
    </tr>
    <tr>
      <th>54</th>
      <td>[the, played, with]</td>
      <td>boy</td>
    </tr>
    <tr>
      <th>55</th>
      <td>[the, boy, with, a]</td>
      <td>played</td>
    </tr>
    <tr>
      <th>56</th>
      <td>[boy, played, a, blue]</td>
      <td>with</td>
    </tr>
    <tr>
      <th>57</th>
      <td>[played, with, blue, ball]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>58</th>
      <td>[with, a, ball]</td>
      <td>blue</td>
    </tr>
    <tr>
      <th>59</th>
      <td>[a, blue]</td>
      <td>ball</td>
    </tr>
  </tbody>
</table>
</div>




```python
import random

# Build vocab from all sentences
vocab = list({word for sent in sentences for word in sent})

# Simulate predictions
def mock_predict(context_words):
    """Fake prediction: pick a random word from vocab."""
    return random.choice(vocab)

# Add a 'Predicted' column
df_cbow['Predicted Word'] = df_cbow['Input (Context Words)'].apply(mock_predict)

# Check correctness
df_cbow['Correct?'] = df_cbow['Predicted Word'] == df_cbow['Output (Target Word)']

df_cbow
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
      <th>Input (Context Words)</th>
      <th>Output (Target Word)</th>
      <th>Predicted Word</th>
      <th>Correct?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[chef, cooked]</td>
      <td>the</td>
      <td>ate</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[the, cooked, a]</td>
      <td>chef</td>
      <td>planted</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[the, chef, a, delicious]</td>
      <td>cooked</td>
      <td>mouse</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[chef, cooked, delicious, meal]</td>
      <td>a</td>
      <td>with</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[cooked, a, meal]</td>
      <td>delicious</td>
      <td>small</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[a, delicious]</td>
      <td>meal</td>
      <td>at</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[baker, baked]</td>
      <td>the</td>
      <td>waiter</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[the, baked, fresh]</td>
      <td>baker</td>
      <td>baker</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[the, baker, fresh, bread]</td>
      <td>baked</td>
      <td>sandwich</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[baker, baked, bread]</td>
      <td>fresh</td>
      <td>mouse</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[baked, fresh]</td>
      <td>bread</td>
      <td>rose</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[waiter, served]</td>
      <td>the</td>
      <td>bread</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[the, served, the]</td>
      <td>waiter</td>
      <td>delicious</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[the, waiter, the, cold]</td>
      <td>served</td>
      <td>a</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[waiter, served, cold, drinks]</td>
      <td>the</td>
      <td>meal</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[served, the, drinks]</td>
      <td>cold</td>
      <td>gardener</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[the, cold]</td>
      <td>drinks</td>
      <td>the</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[customer, ate]</td>
      <td>the</td>
      <td>sun</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[the, ate, a]</td>
      <td>customer</td>
      <td>at</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[the, customer, a, tasty]</td>
      <td>ate</td>
      <td>mouse</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[customer, ate, tasty, sandwich]</td>
      <td>a</td>
      <td>shone</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[ate, a, sandwich]</td>
      <td>tasty</td>
      <td>served</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>[a, tasty]</td>
      <td>sandwich</td>
      <td>the</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>[cat, chased]</td>
      <td>the</td>
      <td>customer</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[the, chased, the]</td>
      <td>cat</td>
      <td>moon</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>[the, cat, the, small]</td>
      <td>chased</td>
      <td>fresh</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>[cat, chased, small, mouse]</td>
      <td>the</td>
      <td>served</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>[chased, the, mouse]</td>
      <td>small</td>
      <td>planted</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>[the, small]</td>
      <td>mouse</td>
      <td>in</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>[dog, barked]</td>
      <td>the</td>
      <td>baker</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>[the, barked, at]</td>
      <td>dog</td>
      <td>ball</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>[the, dog, at, the]</td>
      <td>barked</td>
      <td>morning</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>[dog, barked, the, mailman]</td>
      <td>at</td>
      <td>cat</td>
      <td>False</td>
    </tr>
    <tr>
      <th>33</th>
      <td>[barked, at, mailman]</td>
      <td>the</td>
      <td>mouse</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>[at, the]</td>
      <td>mailman</td>
      <td>blue</td>
      <td>False</td>
    </tr>
    <tr>
      <th>35</th>
      <td>[sun, rose]</td>
      <td>the</td>
      <td>morning</td>
      <td>False</td>
    </tr>
    <tr>
      <th>36</th>
      <td>[the, rose, in]</td>
      <td>sun</td>
      <td>baker</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>[the, sun, in, the]</td>
      <td>rose</td>
      <td>blue</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>[sun, rose, the, morning]</td>
      <td>in</td>
      <td>planted</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>[rose, in, morning]</td>
      <td>the</td>
      <td>the</td>
      <td>True</td>
    </tr>
    <tr>
      <th>40</th>
      <td>[in, the]</td>
      <td>morning</td>
      <td>boy</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>[moon, shone]</td>
      <td>the</td>
      <td>chef</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42</th>
      <td>[the, shone, in]</td>
      <td>moon</td>
      <td>waiter</td>
      <td>False</td>
    </tr>
    <tr>
      <th>43</th>
      <td>[the, moon, in, the]</td>
      <td>shone</td>
      <td>waiter</td>
      <td>False</td>
    </tr>
    <tr>
      <th>44</th>
      <td>[moon, shone, the, night]</td>
      <td>in</td>
      <td>red</td>
      <td>False</td>
    </tr>
    <tr>
      <th>45</th>
      <td>[shone, in, night]</td>
      <td>the</td>
      <td>cat</td>
      <td>False</td>
    </tr>
    <tr>
      <th>46</th>
      <td>[in, the]</td>
      <td>night</td>
      <td>waiter</td>
      <td>False</td>
    </tr>
    <tr>
      <th>47</th>
      <td>[gardener, planted]</td>
      <td>the</td>
      <td>chased</td>
      <td>False</td>
    </tr>
    <tr>
      <th>48</th>
      <td>[the, planted, a]</td>
      <td>gardener</td>
      <td>moon</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49</th>
      <td>[the, gardener, a, red]</td>
      <td>planted</td>
      <td>dog</td>
      <td>False</td>
    </tr>
    <tr>
      <th>50</th>
      <td>[gardener, planted, red, rose]</td>
      <td>a</td>
      <td>ate</td>
      <td>False</td>
    </tr>
    <tr>
      <th>51</th>
      <td>[planted, a, rose]</td>
      <td>red</td>
      <td>customer</td>
      <td>False</td>
    </tr>
    <tr>
      <th>52</th>
      <td>[a, red]</td>
      <td>rose</td>
      <td>with</td>
      <td>False</td>
    </tr>
    <tr>
      <th>53</th>
      <td>[boy, played]</td>
      <td>the</td>
      <td>tasty</td>
      <td>False</td>
    </tr>
    <tr>
      <th>54</th>
      <td>[the, played, with]</td>
      <td>boy</td>
      <td>at</td>
      <td>False</td>
    </tr>
    <tr>
      <th>55</th>
      <td>[the, boy, with, a]</td>
      <td>played</td>
      <td>baked</td>
      <td>False</td>
    </tr>
    <tr>
      <th>56</th>
      <td>[boy, played, a, blue]</td>
      <td>with</td>
      <td>blue</td>
      <td>False</td>
    </tr>
    <tr>
      <th>57</th>
      <td>[played, with, blue, ball]</td>
      <td>a</td>
      <td>in</td>
      <td>False</td>
    </tr>
    <tr>
      <th>58</th>
      <td>[with, a, ball]</td>
      <td>blue</td>
      <td>cooked</td>
      <td>False</td>
    </tr>
    <tr>
      <th>59</th>
      <td>[a, blue]</td>
      <td>ball</td>
      <td>small</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# CBOW model: sg=0 -> CBOW
from gensim.models import Word2Vec
model = Word2Vec(sentences,   # list of tokenized sentences
                 vector_size=5,  # Word vector dimensionality  
                 window=2,  # Context window size     
                 min_count=1, # minimum word count - ignore words that appear less than once 
                 sg=0)
```


```python
# Train for multiple epochs
model.train(sentences, 
            total_examples=len(sentences), # 
            epochs=10)  #  # number of times to go through the dataset
          # output is (trained_word_count, raw_word_count).
```




    (127, 600)



See this [post](https://stackoverflow.com/questions/76300983/how-to-interpret-word2vec-train-output)


```python
vectors = {word: model.wv[word] for word in model.wv.key_to_index}
vectors_df = pd.DataFrame.from_dict(vectors, orient='index')
vectors_df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>the</th>
      <td>-0.010457</td>
      <td>0.005250</td>
      <td>0.102446</td>
      <td>0.179594</td>
      <td>-0.185364</td>
    </tr>
    <tr>
      <th>a</th>
      <td>-0.142546</td>
      <td>0.130141</td>
      <td>0.180161</td>
      <td>-0.101365</td>
      <td>-0.075317</td>
    </tr>
    <tr>
      <th>in</th>
      <td>0.147139</td>
      <td>-0.029705</td>
      <td>-0.090012</td>
      <td>0.129747</td>
      <td>-0.096810</td>
    </tr>
    <tr>
      <th>rose</th>
      <td>-0.036268</td>
      <td>0.057604</td>
      <td>0.019893</td>
      <td>-0.165776</td>
      <td>-0.188701</td>
    </tr>
    <tr>
      <th>ball</th>
      <td>0.146327</td>
      <td>0.101453</td>
      <td>0.135293</td>
      <td>0.015081</td>
      <td>0.127311</td>
    </tr>
    <tr>
      <th>blue</th>
      <td>-0.068210</td>
      <td>-0.018442</td>
      <td>0.115617</td>
      <td>-0.151060</td>
      <td>-0.078563</td>
    </tr>
    <tr>
      <th>with</th>
      <td>-0.150268</td>
      <td>-0.018444</td>
      <td>0.190968</td>
      <td>-0.146771</td>
      <td>-0.046459</td>
    </tr>
    <tr>
      <th>played</th>
      <td>-0.038811</td>
      <td>0.161621</td>
      <td>-0.118537</td>
      <td>0.000741</td>
      <td>-0.095021</td>
    </tr>
    <tr>
      <th>boy</th>
      <td>-0.192068</td>
      <td>0.100312</td>
      <td>-0.175176</td>
      <td>-0.088079</td>
      <td>-0.000675</td>
    </tr>
    <tr>
      <th>red</th>
      <td>-0.005880</td>
      <td>-0.153306</td>
      <td>0.192259</td>
      <td>0.099813</td>
      <td>0.184810</td>
    </tr>
    <tr>
      <th>planted</th>
      <td>-0.163280</td>
      <td>0.090571</td>
      <td>-0.082231</td>
      <td>0.015780</td>
      <td>0.170316</td>
    </tr>
    <tr>
      <th>gardener</th>
      <td>-0.089411</td>
      <td>0.090719</td>
      <td>-0.135464</td>
      <td>-0.071444</td>
      <td>0.188286</td>
    </tr>
    <tr>
      <th>night</th>
      <td>-0.031501</td>
      <td>0.006317</td>
      <td>-0.082636</td>
      <td>-0.153988</td>
      <td>-0.029952</td>
    </tr>
    <tr>
      <th>shone</th>
      <td>0.049123</td>
      <td>-0.017158</td>
      <td>0.111141</td>
      <td>-0.055953</td>
      <td>0.045918</td>
    </tr>
    <tr>
      <th>moon</th>
      <td>0.109236</td>
      <td>0.166956</td>
      <td>-0.028995</td>
      <td>-0.184301</td>
      <td>0.087726</td>
    </tr>
    <tr>
      <th>morning</th>
      <td>0.011427</td>
      <td>0.148969</td>
      <td>-0.016202</td>
      <td>-0.052892</td>
      <td>-0.174917</td>
    </tr>
    <tr>
      <th>sun</th>
      <td>-0.017189</td>
      <td>0.056627</td>
      <td>0.108056</td>
      <td>0.140947</td>
      <td>-0.114042</td>
    </tr>
    <tr>
      <th>mailman</th>
      <td>0.037121</td>
      <td>0.121848</td>
      <td>-0.095756</td>
      <td>-0.062154</td>
      <td>0.136370</td>
    </tr>
    <tr>
      <th>at</th>
      <td>0.032476</td>
      <td>0.004295</td>
      <td>0.069767</td>
      <td>0.003955</td>
      <td>0.192974</td>
    </tr>
    <tr>
      <th>barked</th>
      <td>0.101011</td>
      <td>-0.177887</td>
      <td>-0.140657</td>
      <td>0.017618</td>
      <td>0.128211</td>
    </tr>
    <tr>
      <th>dog</th>
      <td>-0.172551</td>
      <td>0.073778</td>
      <td>0.104016</td>
      <td>0.114471</td>
      <td>0.149565</td>
    </tr>
    <tr>
      <th>mouse</th>
      <td>-0.123347</td>
      <td>0.022194</td>
      <td>0.121059</td>
      <td>-0.056803</td>
      <td>-0.123304</td>
    </tr>
    <tr>
      <th>small</th>
      <td>-0.008053</td>
      <td>-0.167410</td>
      <td>-0.111962</td>
      <td>0.142237</td>
      <td>0.067007</td>
    </tr>
    <tr>
      <th>chased</th>
      <td>0.144538</td>
      <td>0.136010</td>
      <td>0.150661</td>
      <td>-0.075770</td>
      <td>-0.011101</td>
    </tr>
    <tr>
      <th>cat</th>
      <td>0.046829</td>
      <td>-0.090163</td>
      <td>0.167978</td>
      <td>-0.197526</td>
      <td>0.135251</td>
    </tr>
    <tr>
      <th>sandwich</th>
      <td>0.058274</td>
      <td>-0.098662</td>
      <td>0.087989</td>
      <td>-0.034793</td>
      <td>0.134228</td>
    </tr>
    <tr>
      <th>tasty</th>
      <td>0.199285</td>
      <td>-0.087196</td>
      <td>-0.011932</td>
      <td>-0.113975</td>
      <td>0.077013</td>
    </tr>
    <tr>
      <th>ate</th>
      <td>0.055714</td>
      <td>0.137844</td>
      <td>0.122075</td>
      <td>0.190730</td>
      <td>0.185482</td>
    </tr>
    <tr>
      <th>customer</th>
      <td>0.157985</td>
      <td>-0.139743</td>
      <td>-0.183023</td>
      <td>-0.007118</td>
      <td>-0.061956</td>
    </tr>
    <tr>
      <th>drinks</th>
      <td>0.157620</td>
      <td>0.119370</td>
      <td>-0.030634</td>
      <td>0.029639</td>
      <td>0.036255</td>
    </tr>
    <tr>
      <th>cold</th>
      <td>0.156327</td>
      <td>-0.190129</td>
      <td>-0.004070</td>
      <td>0.069299</td>
      <td>-0.018739</td>
    </tr>
    <tr>
      <th>served</th>
      <td>0.167728</td>
      <td>0.180348</td>
      <td>0.130756</td>
      <td>-0.014508</td>
      <td>0.154625</td>
    </tr>
    <tr>
      <th>waiter</th>
      <td>-0.170811</td>
      <td>0.064788</td>
      <td>-0.092450</td>
      <td>-0.102401</td>
      <td>0.071761</td>
    </tr>
    <tr>
      <th>bread</th>
      <td>0.107310</td>
      <td>0.155608</td>
      <td>-0.115248</td>
      <td>0.148403</td>
      <td>0.132626</td>
    </tr>
    <tr>
      <th>fresh</th>
      <td>-0.074399</td>
      <td>-0.174517</td>
      <td>0.109016</td>
      <td>0.129808</td>
      <td>-0.015534</td>
    </tr>
    <tr>
      <th>baked</th>
      <td>-0.134197</td>
      <td>-0.141672</td>
      <td>-0.049879</td>
      <td>0.102757</td>
      <td>-0.073174</td>
    </tr>
    <tr>
      <th>baker</th>
      <td>-0.187625</td>
      <td>0.077137</td>
      <td>0.098068</td>
      <td>-0.129208</td>
      <td>0.024539</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>-0.041690</td>
      <td>0.000785</td>
      <td>-0.197457</td>
      <td>0.053478</td>
      <td>-0.095012</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>0.021377</td>
      <td>-0.031026</td>
      <td>0.044223</td>
      <td>-0.158189</td>
      <td>-0.054195</td>
    </tr>
    <tr>
      <th>cooked</th>
      <td>0.052959</td>
      <td>0.107506</td>
      <td>-0.047820</td>
      <td>-0.190854</td>
      <td>0.090210</td>
    </tr>
    <tr>
      <th>chef</th>
      <td>0.001515</td>
      <td>0.062136</td>
      <td>-0.135774</td>
      <td>-0.028299</td>
      <td>0.153415</td>
    </tr>
  </tbody>
</table>
</div>




```python
similar_to_land = model.wv.most_similar("chef", topn=5)
similar_to_land
```




    [('gardener', 0.9258278012275696),
     ('mailman', 0.9167701601982117),
     ('planted', 0.7344727516174316),
     ('bread', 0.6474943161010742),
     ('cooked', 0.6081644892692566)]




```python
# Look at similarity
model.wv.similarity('chef', 'waiter')  # high similarity because contexts overlap
```




    np.float32(0.58342105)




```python
model.wv.similarity('delicious', 'meal')   # lower similarity
```




    np.float32(-0.31912485)



Bonus: in the "Improving Distributional Similarity with Lessons Learned from Word Embeddings" paper, Goldberg points out that: 
"state-of-the-art embedding methods are all based on the same bag-of-contexts representation of words. Furthermore, analysis by Levy and Goldberg (2014c) shows that word2vec’s SGNS is implicitly factorizing a word-context PMI matrix." 


```python
from collections import defaultdict

window_size = 10

# vocabulary
vocab = sorted({word for sent in sentences for word in sent})

# initialize matrix
word_context_counts = pd.DataFrame(
    0,
    index=vocab,
    columns=vocab
)

# fill counts
for sentence in sentences:
    for i, target in enumerate(sentence):
        for j in range(max(i-window_size,0), min(i+window_size+1,len(sentence))):
            if i != j:
                context = sentence[j]
                word_context_counts.loc[target, context] += 1

word_context_counts
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
      <th>a</th>
      <th>at</th>
      <th>ate</th>
      <th>baked</th>
      <th>baker</th>
      <th>ball</th>
      <th>barked</th>
      <th>blue</th>
      <th>boy</th>
      <th>bread</th>
      <th>...</th>
      <th>rose</th>
      <th>sandwich</th>
      <th>served</th>
      <th>shone</th>
      <th>small</th>
      <th>sun</th>
      <th>tasty</th>
      <th>the</th>
      <th>waiter</th>
      <th>with</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>at</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ate</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>baked</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>baker</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ball</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>barked</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>blue</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>boy</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bread</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cat</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>chased</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>chef</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cold</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cooked</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>customer</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>dog</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>drinks</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fresh</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gardener</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>in</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>mailman</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>moon</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>morning</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>mouse</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>night</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>planted</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>played</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>red</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>rose</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sandwich</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>served</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>shone</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>small</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sun</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tasty</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>the</th>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>waiter</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>with</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>41 rows × 41 columns</p>
</div>




```python
import numpy as np

# total counts
total = word_context_counts.values.sum()

# probabilities
p_wc = word_context_counts / total
p_w = word_context_counts.sum(axis=1) / total
p_c = word_context_counts.sum(axis=0) / total

# compute PMI
pmi = np.log2(p_wc.div(p_w, axis=0).div(p_c, axis=1))

pmi = pmi.replace([-np.inf, np.inf], 0).fillna(0)

pmi
```

    /opt/anaconda3/lib/python3.13/site-packages/pandas/core/internals/blocks.py:393: RuntimeWarning:
    
    divide by zero encountered in log2
    





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
      <th>a</th>
      <th>at</th>
      <th>ate</th>
      <th>baked</th>
      <th>baker</th>
      <th>ball</th>
      <th>barked</th>
      <th>blue</th>
      <th>boy</th>
      <th>bread</th>
      <th>...</th>
      <th>rose</th>
      <th>sandwich</th>
      <th>served</th>
      <th>shone</th>
      <th>small</th>
      <th>sun</th>
      <th>tasty</th>
      <th>the</th>
      <th>waiter</th>
      <th>with</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.261125</td>
      <td>0.000000</td>
      <td>1.261125</td>
      <td>1.261125</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.524159</td>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.524159</td>
      <td>-0.382731</td>
      <td>0.000000</td>
      <td>1.261125</td>
    </tr>
    <tr>
      <th>at</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ate</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>baked</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.238405</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.238405</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009586</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>baker</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.238405</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.238405</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009586</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ball</th>
      <td>1.261125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.575376</td>
      <td>0.000000</td>
      <td>3.068480</td>
    </tr>
    <tr>
      <th>barked</th>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>blue</th>
      <td>1.261125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.575376</td>
      <td>0.000000</td>
      <td>3.068480</td>
    </tr>
    <tr>
      <th>boy</th>
      <td>1.261125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.575376</td>
      <td>0.000000</td>
      <td>3.068480</td>
    </tr>
    <tr>
      <th>bread</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.238405</td>
      <td>4.238405</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009586</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>cat</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>chased</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>chef</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>cold</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>3.594549</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>cooked</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>customer</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>dog</th>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>drinks</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>3.594549</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>fresh</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.238405</td>
      <td>4.238405</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.238405</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009586</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>gardener</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>in</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mailman</th>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>moon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>morning</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mouse</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>night</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>planted</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>played</th>
      <td>1.261125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.575376</td>
      <td>0.000000</td>
      <td>3.068480</td>
    </tr>
    <tr>
      <th>red</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>rose</th>
      <td>0.524159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>0.272620</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sandwich</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>served</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>3.594549</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>shone</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>small</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sun</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tasty</th>
      <td>1.524159</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.312342</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>the</th>
      <td>-0.382731</td>
      <td>0.687658</td>
      <td>-0.312342</td>
      <td>0.009586</td>
      <td>0.009586</td>
      <td>-0.575376</td>
      <td>0.687658</td>
      <td>-0.575376</td>
      <td>-0.575376</td>
      <td>0.009586</td>
      <td>...</td>
      <td>0.272620</td>
      <td>-0.312342</td>
      <td>0.687658</td>
      <td>0.687658</td>
      <td>0.687658</td>
      <td>0.687658</td>
      <td>-0.312342</td>
      <td>-0.897305</td>
      <td>0.687658</td>
      <td>-0.575376</td>
    </tr>
    <tr>
      <th>waiter</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.594549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.687658</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>with</th>
      <td>1.261125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>3.068480</td>
      <td>3.068480</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.575376</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>41 rows × 41 columns</p>
</div>




```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create the mask for the upper triangle

plt.figure(figsize=(10, 8))

# Plot the heatmap
sns.heatmap(
    pmi, 
    cmap="Blues", 
    square=True, 
    cbar_kws={"shrink": .8}
)


plt.xticks(rotation=90) # # Rotate the x-axis labels to 90 degrees
plt.yticks(rotation=0) # Keep y-axis horizontal for readability

plt.title("Word–Context Matrix")
plt.show()
```


    
![png](output_71_0.png)
    


## Word2Vec on a corpus

For this part, we'll be using the [Gensim library](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py) but on a larger scale.

Note that word2vec requires sentences as inputs.







```python
#!pip install gensim
```


```python
import string
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Downloads if not already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stoplist = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def normalize_text(doc):
    # Lowercase
    doc = doc.lower()
    # Replace apostrophes smartly: don't -> dont (or you could expand contractions)
    doc = doc.replace("'", "")
    # Remove punctuation
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(doc)
    # Remove stopwords
    words = [w for w in words if w not in stoplist]
    # Normalize numbers
    words = ['#' if w.isdigit() else w for w in words]
    # Lemmatize to reduce plurals, verb forms
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

def get_sentences(doc):
    return [normalize_text(sent) for sent in sent_tokenize(doc)]
```

    [nltk_data] Downloading package stopwords to /Users/ilya/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /Users/ilya/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /Users/ilya/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to /Users/ilya/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!



```python
# Process all documents
sentences = []
for doc in df['text']:
    sentences += get_sentences(doc)
```


```python
sentences[2]
```




    ['intention', 'party', 'must', 'sought']




```python
# Train Word2Vec

w2v_model = Word2Vec(sentences,  # list of tokenized sentences
               workers = 4, # Number of threads to run in parallel
               vector_size=300,  # Word vector dimensionality     
               min_count = 2, # Minimum word count  
               window = 5 # Context window size      
               )

```


```python
words = list(w2v_model.wv.index_to_key)
words[:10]
```




    ['#',
     '’',
     '“',
     'court',
     'case',
     'defendant',
     'v',
     '”',
     'plaintiff',
     'corporation']




```python
## how many words in vocab
print(len(words))

```

    23955



```python
## Print actual values of word embedding - this is the hidden leayer aka the word embedding we "learned"

len(w2v_model.wv['court']) # vector for "judge"
```




    300




```python
print(w2v_model.wv.get_vector('law'))
```

    [ 0.9142291  -0.5339833   0.2746493   1.6266046  -1.4366103   0.2603867
     -0.15430531  0.1199854   0.59910214  1.5280026  -0.00253317  1.1297374
      1.5747863   0.90380627  1.1895101   1.8506919   0.02008733 -0.8329556
      1.3753487   0.01776134  1.6296716   1.8351139  -0.8404059  -0.2473482
     -1.8868259   1.6391777   0.27477214  1.3487777  -0.7893847   0.7299489
      0.5312579  -0.4982083   1.0236554   0.4254822  -0.39474556  1.3628973
     -1.6368765   0.22922496  0.11263172 -0.7456108  -0.6449713   1.184895
      0.04869631 -0.20527153  0.14652984 -0.30126476 -1.8176981  -0.5080804
      1.1989799  -0.93489486  0.9161819   0.27906275  0.4472068   0.7387095
      1.6330967   0.6448473   1.6423601  -0.02903326 -1.08663    -0.28002417
      0.18039395  0.4912914  -0.11659174  0.23906723 -1.301045    0.17664467
     -1.0220689   0.20170367 -0.33506063  0.27670196 -1.8289362  -1.2048304
     -0.6472456  -0.27458644  1.102625    0.9770931  -0.82818806  1.1903697
      1.3852067  -0.66933036  0.99508584 -0.07454011 -1.5749639  -0.5693207
      1.9952767  -0.77321935 -0.53357893  1.4263663   0.78017426 -0.5974957
      0.34652615 -0.38890484 -0.82229644 -0.22486444 -0.13686663  0.12986824
     -1.865931    0.2768646  -0.42940944  0.15431798  0.49784434  0.53499585
     -0.6606207  -0.37105197 -0.7167458   1.5317025   0.3379159  -0.91838133
      1.6956378   0.5076267  -0.7673857  -0.9937738  -0.6071015   0.264697
      0.54657084 -0.30531207 -0.6156733  -0.4636815  -0.3158972  -0.6899009
      1.1489292  -0.04443556 -0.5124903  -0.3851733  -0.42479604  0.34241125
      0.304768    0.68787587 -0.23445871  1.3441304   0.68061066  0.15843451
      0.01338791  0.66958714  1.0222815  -0.02951255  1.6088663   1.0948019
     -2.255534    1.3940135  -0.9992296  -0.616814   -2.43472     0.73655367
     -0.41726598 -0.35370463  2.6356685   0.43315518 -0.05484107  1.0468937
     -0.8666376  -0.22402944 -0.13019805 -0.46363646 -0.9217416   0.8044542
     -1.275156   -0.03919093 -0.2790564   0.4204044  -0.36975548  0.05832778
      1.853241   -0.44709703  0.14988048 -0.22739041  0.6290748   0.11702125
     -2.3357298   0.8449681   0.7774565   1.3990434   0.24663658 -0.34503615
     -0.16281335  0.76077443 -1.0117522  -0.22807012  0.59455967 -0.44472072
     -0.7619128  -1.0852524  -1.0645987  -1.1419905   0.56541365 -0.21369044
      1.1108748  -0.42489395  0.3499504   1.5156496   1.3625325  -1.1314123
     -0.88648665 -1.4146216  -2.5360062   0.81980634  0.5455882   0.05309995
      0.61974645 -1.1629848   0.65841514 -0.83003354  0.699303    0.25169438
      0.951373   -0.5744904  -0.1513834  -0.28339097  0.2802393   0.05918315
     -1.5173397  -0.04288187  1.491896    0.22486655 -0.20661305 -0.40705794
     -1.3950093  -0.39273056 -0.17204033 -0.30577293 -0.3615274   0.98785394
      1.3941454  -0.15129913  0.07395262  0.235033   -0.15506795  1.1279753
      0.09363301  0.32630134 -0.02538142  0.7192137   0.68517137  1.6393092
     -0.8036893  -0.57317823  1.2954115   1.6727571  -1.0900651  -0.8724475
      0.360359    0.3665332  -0.31870413  0.7770077   2.002217   -0.34551606
     -0.60654044 -0.7112517  -1.0062205  -0.00325779 -0.6277088   1.0779939
      0.77482    -0.4268534  -0.29765105  0.26981634 -0.41696206  0.25736484
      0.39704475 -0.6188184  -1.3456203   0.8921844  -0.9298685  -0.54930305
      1.2036667  -0.22109127 -0.5647482   0.8844034  -0.6869105  -0.85274327
     -0.34987122  1.011651   -0.4223056   0.42648906 -0.17310329  0.2049562
     -0.84387004  0.5411525  -1.1910585  -0.39789864  0.91132927 -0.10390548
     -0.8463993  -1.1285564   1.2119623  -0.702703   -0.55759215 -0.36153534
     -0.24959517 -1.4164532  -1.2311907  -0.5376028  -1.1405644  -0.19580835
      0.48375377 -1.4336989  -0.34761634 -0.05921376 -0.88338083 -0.12672643]



```python
## Cosine similarity between two vectors
print(w2v_model.wv.similarity('crime', 'law'))
```

    0.17217113



```python
## Most similar words
w2v_model.wv.similar_by_word('crime')
```




    [('punishment', 0.7317579388618469),
     ('violator', 0.6987003087997437),
     ('malicious', 0.6969221830368042),
     ('offender', 0.6910900473594666),
     ('offense', 0.6908679008483887),
     ('nonsupport', 0.6875366568565369),
     ('80a1', 0.6862915754318237),
     ('optometry', 0.6861080527305603),
     ('unlicensed', 0.6847260594367981),
     ('criminal', 0.6791772246360779)]




```python
## We can even see which words are not fitting in a given pattern
w2v_model.wv.doesnt_match("he committed a crime english with a weapon".split())
```




    'english'




```python
## vector addition - we can add vectors to get to a new "vector" (that might not exist)

vector = w2v_model.wv.get_vector('corporation') - w2v_model.wv.get_vector('money') 
w2v_model.wv.similar_by_vector(vector)


```




    [('corporation', 0.6297608613967896),
     ('merger', 0.48571059107780457),
     ('facto', 0.4731733798980713),
     ('consolidation', 0.44776788353919983),
     ('belle', 0.44713094830513),
     ('american', 0.4337545335292816),
     ('whollyowned', 0.43374884128570557),
     ('corp', 0.4329778552055359),
     ('registration', 0.4289991855621338),
     ('planter', 0.4229223430156708)]




```python
vector = w2v_model.wv.get_vector('judge') + w2v_model.wv.get_vector('opinion')  ## decision?
w2v_model.wv.similar_by_vector(vector)
```




    [('opinion', 0.8364631533622742),
     ('judge', 0.7775453329086304),
     ('reviewing', 0.6333453059196472),
     ('dictum', 0.6329247951507568),
     ('chief', 0.627659797668457),
     ('lord', 0.6190498471260071),
     ('writer', 0.6166452169418335),
     ('learned', 0.6113904118537903),
     ('welsh', 0.6107160449028015),
     ('reviewed', 0.6087403297424316)]




```python
w2v_model.wv.most_similar(positive=['law', 'court'], negative = ['crime'])
```




    [('apply', 0.4351530373096466),
     ('determined', 0.4169975221157074),
     ('proceed', 0.41324371099472046),
     ('concluded', 0.4093601405620575),
     ('recognize', 0.39917832612991333),
     ('follow', 0.3980320692062378),
     ('judicial', 0.3901236653327942),
     ('nevertheless', 0.38917818665504456),
     ('disposed', 0.3857983350753784),
     ('expressly', 0.3841733932495117)]



## Visualizing word2vec word embeddings
Once we have our word embedding model, we can viszualize it using the standard techniques - such as PCA and TSNE. 

The problem is that we're reducing from 300 dimensions to 2. Note that PCA has a unique singular representation whereas TSNE (another representation method) is a bit more complex, so it will always have a different representation every time you print the graph. 






```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# This code is adapted from https://github.com/drelhaj/NLP_ML_Visualization_Tutorial/blob/master/6_Word_embeddings_Tutorial.ipynb
def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']
    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=20).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig = plt.subplots()
    #fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    
```


```python
word = 'crime'
tsnescatterplot(w2v_model, word,
                [t[0] for t in w2v_model.wv.most_similar(positive=[word], 
                                                         topn=20)][10:])
```


    
![png](output_90_0.png)
    


### PCA plot of words


```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Take top N words
top_words = words[:200]
vectors = [w2v_model.wv[w] for w in top_words]

# Reduce dimensions
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

plt.figure(figsize=(12,8))
for i, word in enumerate(top_words):
    x, y = coords[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word, fontsize=9)
plt.show()
```


    
![png](output_92_0.png)
    



```python
# save word2vec model
# w2v_model.save('w2v_model_vectors.pkl')
```

## [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py)
Doc2Vec is the same thing as word2vec, but with an extra column representation  for a given document.

Doc2Vec can be used for classification as an alternative document representation. Unlike topics, it's harder to interpret the output (becasue we're dealing with some  vectors)

<div align="center">
<img src="https://miro.medium.com/max/640/0*x-gtU4UlO8FAsRvL." width="60%" />
</div>







```python
from nltk import word_tokenize

docs = []
for i, row in df.iterrows():
    docs += [word_tokenize(row['text'])]

```


```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

doc_iterator = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

d2v_model = Doc2Vec(doc_iterator, # list of tokenized documents
                   workers = 4, # Number of threads to run in parallel
                   vector_size = 300,  # Word vector dimensionality     
                   min_count = 2, # Minimum word count  
                   window = 10 # Context window size      
                   #max_vocab_size =  10000
                  )
```


```python
# d2v_model.save('d2v-vectors.pkl')
```


```python
# matrix of all document vectors:
doc2vec_matrix = d2v_model.dv.vectors
doc2vec_matrix.shape
```




    (2361, 300)




```python
d2v_matrix = pd.DataFrame(data = doc2vec_matrix, 
                          index = df['name_abbreviation'])
```


```python
d2v_matrix.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>290</th>
      <th>291</th>
      <th>292</th>
      <th>293</th>
      <th>294</th>
      <th>295</th>
      <th>296</th>
      <th>297</th>
      <th>298</th>
      <th>299</th>
    </tr>
    <tr>
      <th>name_abbreviation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dale v. Smith</th>
      <td>-0.343278</td>
      <td>0.188420</td>
      <td>0.985286</td>
      <td>-0.036985</td>
      <td>0.364555</td>
      <td>1.064996</td>
      <td>-0.386215</td>
      <td>0.768072</td>
      <td>0.631217</td>
      <td>1.680692</td>
      <td>...</td>
      <td>1.086667</td>
      <td>-0.099602</td>
      <td>-0.899343</td>
      <td>1.150400</td>
      <td>1.645249</td>
      <td>0.846212</td>
      <td>-0.568730</td>
      <td>0.114108</td>
      <td>0.222135</td>
      <td>0.417394</td>
    </tr>
    <tr>
      <th>Dale v. Smith</th>
      <td>0.104029</td>
      <td>-0.085672</td>
      <td>-0.024778</td>
      <td>-0.115830</td>
      <td>-0.000990</td>
      <td>0.007011</td>
      <td>-0.031642</td>
      <td>0.228075</td>
      <td>0.088328</td>
      <td>0.192005</td>
      <td>...</td>
      <td>0.105260</td>
      <td>0.108159</td>
      <td>-0.299935</td>
      <td>-0.012761</td>
      <td>0.344503</td>
      <td>0.236111</td>
      <td>-0.192207</td>
      <td>0.076963</td>
      <td>0.408612</td>
      <td>-0.004024</td>
    </tr>
    <tr>
      <th>Tatem v. Gilpin</th>
      <td>-1.544591</td>
      <td>-0.367269</td>
      <td>1.843301</td>
      <td>-1.141894</td>
      <td>-0.528872</td>
      <td>-0.585710</td>
      <td>0.141591</td>
      <td>1.141459</td>
      <td>0.065310</td>
      <td>1.232744</td>
      <td>...</td>
      <td>-0.115278</td>
      <td>1.858365</td>
      <td>-0.161845</td>
      <td>1.349125</td>
      <td>1.140487</td>
      <td>0.163453</td>
      <td>-0.690625</td>
      <td>0.081092</td>
      <td>0.660049</td>
      <td>0.008251</td>
    </tr>
    <tr>
      <th>Woolaston v. Mendenhall</th>
      <td>-0.126265</td>
      <td>-0.205903</td>
      <td>-0.060064</td>
      <td>-0.218396</td>
      <td>0.023546</td>
      <td>-0.123984</td>
      <td>-0.042988</td>
      <td>0.546628</td>
      <td>-0.083534</td>
      <td>-0.001944</td>
      <td>...</td>
      <td>-0.042067</td>
      <td>0.105622</td>
      <td>0.020130</td>
      <td>0.304406</td>
      <td>0.800667</td>
      <td>0.720252</td>
      <td>-0.230814</td>
      <td>-0.157946</td>
      <td>0.386927</td>
      <td>-0.389636</td>
    </tr>
    <tr>
      <th>State v. Gilpin</th>
      <td>-1.142242</td>
      <td>0.235695</td>
      <td>-0.280827</td>
      <td>0.264692</td>
      <td>0.308955</td>
      <td>0.184637</td>
      <td>-0.154193</td>
      <td>1.030702</td>
      <td>0.115311</td>
      <td>0.147392</td>
      <td>...</td>
      <td>-0.204630</td>
      <td>1.787471</td>
      <td>-0.528877</td>
      <td>0.601344</td>
      <td>0.716936</td>
      <td>0.809408</td>
      <td>-0.202716</td>
      <td>0.238423</td>
      <td>0.876433</td>
      <td>0.049357</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 300 columns</p>
</div>




```python
# get all pair-wise document similarities
pairwise_sims = cosine_similarity(doc2vec_matrix)
pairwise_sims.shape
```




    (2361, 2361)




```python
d2v_similarity_matrix = pd.DataFrame(data = pairwise_sims, 
                                 columns = df['name_abbreviation'],
                                  index = df['name_abbreviation'])
d2v_similarity_matrix
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
      <th>name_abbreviation</th>
      <th>Dale v. Smith</th>
      <th>Dale v. Smith</th>
      <th>Tatem v. Gilpin</th>
      <th>Woolaston v. Mendenhall</th>
      <th>State v. Gilpin</th>
      <th>Clayton v. Mitchell</th>
      <th>Rodney v. Shankland</th>
      <th>Warner v. Allee</th>
      <th>Philip v. Wood</th>
      <th>Thompson v. Lynam</th>
      <th>...</th>
      <th>Slaughter v. Moore</th>
      <th>Walter v. Peninsula Cut Stone Co.</th>
      <th>Williamson v. McMonagle</th>
      <th>Emmons v. Curlett</th>
      <th>Jacobs v. Wilmington Trust Co.</th>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <th>Dayett v. Willitts</th>
      <th>In re McFarlin</th>
      <th>In re the Real Estate of Donaghy</th>
      <th>In re Tomlinson</th>
    </tr>
    <tr>
      <th>name_abbreviation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dale v. Smith</th>
      <td>1.000000</td>
      <td>0.545046</td>
      <td>0.361473</td>
      <td>0.398555</td>
      <td>0.213928</td>
      <td>0.303746</td>
      <td>0.398599</td>
      <td>0.236108</td>
      <td>0.457560</td>
      <td>0.404828</td>
      <td>...</td>
      <td>0.117428</td>
      <td>0.131287</td>
      <td>0.446586</td>
      <td>0.229344</td>
      <td>0.092380</td>
      <td>0.128654</td>
      <td>0.203455</td>
      <td>0.270541</td>
      <td>0.393987</td>
      <td>0.394396</td>
    </tr>
    <tr>
      <th>Dale v. Smith</th>
      <td>0.545046</td>
      <td>1.000000</td>
      <td>0.503557</td>
      <td>0.763449</td>
      <td>0.354098</td>
      <td>0.629107</td>
      <td>0.444420</td>
      <td>0.493306</td>
      <td>0.512136</td>
      <td>0.562194</td>
      <td>...</td>
      <td>0.351538</td>
      <td>0.418752</td>
      <td>0.557765</td>
      <td>0.402697</td>
      <td>0.390820</td>
      <td>0.410636</td>
      <td>0.450140</td>
      <td>0.383753</td>
      <td>0.540443</td>
      <td>0.514331</td>
    </tr>
    <tr>
      <th>Tatem v. Gilpin</th>
      <td>0.361473</td>
      <td>0.503557</td>
      <td>1.000000</td>
      <td>0.487007</td>
      <td>0.468193</td>
      <td>0.417716</td>
      <td>0.185449</td>
      <td>0.252743</td>
      <td>0.315075</td>
      <td>0.490102</td>
      <td>...</td>
      <td>0.209007</td>
      <td>0.257796</td>
      <td>0.570448</td>
      <td>0.268623</td>
      <td>0.186313</td>
      <td>0.296050</td>
      <td>0.233146</td>
      <td>0.165398</td>
      <td>0.254639</td>
      <td>0.370501</td>
    </tr>
    <tr>
      <th>Woolaston v. Mendenhall</th>
      <td>0.398555</td>
      <td>0.763449</td>
      <td>0.487007</td>
      <td>1.000000</td>
      <td>0.419931</td>
      <td>0.641821</td>
      <td>0.409955</td>
      <td>0.519390</td>
      <td>0.422086</td>
      <td>0.540030</td>
      <td>...</td>
      <td>0.372823</td>
      <td>0.517479</td>
      <td>0.492037</td>
      <td>0.471115</td>
      <td>0.352451</td>
      <td>0.484767</td>
      <td>0.429516</td>
      <td>0.402462</td>
      <td>0.473887</td>
      <td>0.501567</td>
    </tr>
    <tr>
      <th>State v. Gilpin</th>
      <td>0.213928</td>
      <td>0.354098</td>
      <td>0.468193</td>
      <td>0.419931</td>
      <td>1.000000</td>
      <td>0.471210</td>
      <td>0.172706</td>
      <td>0.364926</td>
      <td>0.444153</td>
      <td>0.535461</td>
      <td>...</td>
      <td>0.269021</td>
      <td>0.243047</td>
      <td>0.398589</td>
      <td>0.162260</td>
      <td>0.128595</td>
      <td>0.362399</td>
      <td>0.252113</td>
      <td>0.156447</td>
      <td>0.241554</td>
      <td>0.292832</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <td>0.128654</td>
      <td>0.410636</td>
      <td>0.296050</td>
      <td>0.484767</td>
      <td>0.362399</td>
      <td>0.438883</td>
      <td>0.096776</td>
      <td>0.304563</td>
      <td>0.181904</td>
      <td>0.321051</td>
      <td>...</td>
      <td>0.250482</td>
      <td>0.266473</td>
      <td>0.289525</td>
      <td>0.182224</td>
      <td>0.292872</td>
      <td>1.000000</td>
      <td>0.222573</td>
      <td>0.088780</td>
      <td>0.162217</td>
      <td>0.154853</td>
    </tr>
    <tr>
      <th>Dayett v. Willitts</th>
      <td>0.203455</td>
      <td>0.450140</td>
      <td>0.233146</td>
      <td>0.429516</td>
      <td>0.252113</td>
      <td>0.378787</td>
      <td>0.392632</td>
      <td>0.452116</td>
      <td>0.355713</td>
      <td>0.424138</td>
      <td>...</td>
      <td>0.136291</td>
      <td>0.477573</td>
      <td>0.202846</td>
      <td>0.415593</td>
      <td>0.271189</td>
      <td>0.222573</td>
      <td>1.000000</td>
      <td>0.199981</td>
      <td>0.355438</td>
      <td>0.411967</td>
    </tr>
    <tr>
      <th>In re McFarlin</th>
      <td>0.270541</td>
      <td>0.383753</td>
      <td>0.165398</td>
      <td>0.402462</td>
      <td>0.156447</td>
      <td>0.255215</td>
      <td>0.261905</td>
      <td>0.215344</td>
      <td>0.341861</td>
      <td>0.348523</td>
      <td>...</td>
      <td>-0.032894</td>
      <td>0.177219</td>
      <td>0.241342</td>
      <td>0.167533</td>
      <td>0.192999</td>
      <td>0.088780</td>
      <td>0.199981</td>
      <td>1.000000</td>
      <td>0.317843</td>
      <td>0.416644</td>
    </tr>
    <tr>
      <th>In re the Real Estate of Donaghy</th>
      <td>0.393987</td>
      <td>0.540443</td>
      <td>0.254639</td>
      <td>0.473887</td>
      <td>0.241554</td>
      <td>0.405805</td>
      <td>0.277376</td>
      <td>0.283749</td>
      <td>0.366058</td>
      <td>0.530065</td>
      <td>...</td>
      <td>0.139251</td>
      <td>0.308738</td>
      <td>0.399134</td>
      <td>0.378755</td>
      <td>0.149090</td>
      <td>0.162217</td>
      <td>0.355438</td>
      <td>0.317843</td>
      <td>1.000000</td>
      <td>0.452617</td>
    </tr>
    <tr>
      <th>In re Tomlinson</th>
      <td>0.394396</td>
      <td>0.514331</td>
      <td>0.370501</td>
      <td>0.501567</td>
      <td>0.292832</td>
      <td>0.450989</td>
      <td>0.304413</td>
      <td>0.327896</td>
      <td>0.477082</td>
      <td>0.510300</td>
      <td>...</td>
      <td>0.009066</td>
      <td>0.383705</td>
      <td>0.341658</td>
      <td>0.359733</td>
      <td>0.107120</td>
      <td>0.154853</td>
      <td>0.411967</td>
      <td>0.416644</td>
      <td>0.452617</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>2361 rows × 2361 columns</p>
</div>




```python
d2v_similarity_matrix.sort_values(by='Braasch v. Galdi Securities Corp.', 
                          ascending=False)
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
      <th>name_abbreviation</th>
      <th>Dale v. Smith</th>
      <th>Dale v. Smith</th>
      <th>Tatem v. Gilpin</th>
      <th>Woolaston v. Mendenhall</th>
      <th>State v. Gilpin</th>
      <th>Clayton v. Mitchell</th>
      <th>Rodney v. Shankland</th>
      <th>Warner v. Allee</th>
      <th>Philip v. Wood</th>
      <th>Thompson v. Lynam</th>
      <th>...</th>
      <th>Slaughter v. Moore</th>
      <th>Walter v. Peninsula Cut Stone Co.</th>
      <th>Williamson v. McMonagle</th>
      <th>Emmons v. Curlett</th>
      <th>Jacobs v. Wilmington Trust Co.</th>
      <th>Harned v. Beacon Hill Real Estate Co.</th>
      <th>Dayett v. Willitts</th>
      <th>In re McFarlin</th>
      <th>In re the Real Estate of Donaghy</th>
      <th>In re Tomlinson</th>
    </tr>
    <tr>
      <th>name_abbreviation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Braasch v. Galdi Securities Corp.</th>
      <td>0.126698</td>
      <td>0.374946</td>
      <td>0.204116</td>
      <td>0.416708</td>
      <td>0.229190</td>
      <td>0.342317</td>
      <td>0.235291</td>
      <td>0.311257</td>
      <td>0.177751</td>
      <td>0.375221</td>
      <td>...</td>
      <td>0.154482</td>
      <td>0.349800</td>
      <td>0.255369</td>
      <td>0.174987</td>
      <td>-0.034787</td>
      <td>0.407291</td>
      <td>0.221773</td>
      <td>0.258002</td>
      <td>0.286584</td>
      <td>0.262214</td>
    </tr>
    <tr>
      <th>Stauffer v. Standard Brands Inc.</th>
      <td>0.042307</td>
      <td>0.359034</td>
      <td>0.147802</td>
      <td>0.379506</td>
      <td>0.123488</td>
      <td>0.347834</td>
      <td>0.152061</td>
      <td>0.260334</td>
      <td>0.165832</td>
      <td>0.281101</td>
      <td>...</td>
      <td>0.271778</td>
      <td>0.281123</td>
      <td>0.199861</td>
      <td>0.176259</td>
      <td>0.137507</td>
      <td>0.425581</td>
      <td>0.159213</td>
      <td>0.175437</td>
      <td>0.217912</td>
      <td>0.168248</td>
    </tr>
    <tr>
      <th>Levine v. Milton</th>
      <td>0.254727</td>
      <td>0.664787</td>
      <td>0.369533</td>
      <td>0.693846</td>
      <td>0.445195</td>
      <td>0.625061</td>
      <td>0.358744</td>
      <td>0.470847</td>
      <td>0.367526</td>
      <td>0.564710</td>
      <td>...</td>
      <td>0.378976</td>
      <td>0.503244</td>
      <td>0.466977</td>
      <td>0.319072</td>
      <td>0.215430</td>
      <td>0.550745</td>
      <td>0.340251</td>
      <td>0.333348</td>
      <td>0.425895</td>
      <td>0.357382</td>
    </tr>
    <tr>
      <th>Buechner v. Farbenfabriken Bayer Aktiengesellschaft</th>
      <td>0.082356</td>
      <td>0.501326</td>
      <td>0.205655</td>
      <td>0.590319</td>
      <td>0.366286</td>
      <td>0.533764</td>
      <td>0.186514</td>
      <td>0.444444</td>
      <td>0.231917</td>
      <td>0.464863</td>
      <td>...</td>
      <td>0.238823</td>
      <td>0.490308</td>
      <td>0.364325</td>
      <td>0.345307</td>
      <td>0.210539</td>
      <td>0.528105</td>
      <td>0.384619</td>
      <td>0.300872</td>
      <td>0.383241</td>
      <td>0.323471</td>
    </tr>
    <tr>
      <th>Schenck v. Salt Dome Oil Corp.</th>
      <td>0.186274</td>
      <td>0.487017</td>
      <td>0.212852</td>
      <td>0.573438</td>
      <td>0.310863</td>
      <td>0.482060</td>
      <td>0.216566</td>
      <td>0.358883</td>
      <td>0.199740</td>
      <td>0.380531</td>
      <td>...</td>
      <td>0.294926</td>
      <td>0.344359</td>
      <td>0.281732</td>
      <td>0.239880</td>
      <td>0.205216</td>
      <td>0.461625</td>
      <td>0.318543</td>
      <td>0.224938</td>
      <td>0.305464</td>
      <td>0.323038</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Old Time Petroleum Co. v. Turcol</th>
      <td>0.467074</td>
      <td>0.387612</td>
      <td>0.280915</td>
      <td>0.301745</td>
      <td>0.276656</td>
      <td>0.256025</td>
      <td>0.270394</td>
      <td>0.268969</td>
      <td>0.517319</td>
      <td>0.510051</td>
      <td>...</td>
      <td>0.161065</td>
      <td>0.261006</td>
      <td>0.469050</td>
      <td>0.331259</td>
      <td>0.186972</td>
      <td>0.040198</td>
      <td>0.266956</td>
      <td>0.326546</td>
      <td>0.427378</td>
      <td>0.397184</td>
    </tr>
    <tr>
      <th>Consolidated Fisheries Co. v. Consolidated Solubles Co.</th>
      <td>0.414057</td>
      <td>0.422682</td>
      <td>0.332263</td>
      <td>0.333625</td>
      <td>0.232839</td>
      <td>0.325060</td>
      <td>0.278306</td>
      <td>0.266982</td>
      <td>0.487894</td>
      <td>0.333154</td>
      <td>...</td>
      <td>0.194753</td>
      <td>0.235759</td>
      <td>0.327178</td>
      <td>0.362049</td>
      <td>0.216814</td>
      <td>0.087555</td>
      <td>0.267221</td>
      <td>0.148930</td>
      <td>0.195682</td>
      <td>0.306065</td>
    </tr>
    <tr>
      <th>Girard Trust Co. v. Rector of St. Anne's Protestant Episcopal Church</th>
      <td>0.032571</td>
      <td>0.280378</td>
      <td>0.162042</td>
      <td>0.246606</td>
      <td>-0.033352</td>
      <td>0.289901</td>
      <td>-0.034912</td>
      <td>0.101616</td>
      <td>0.020605</td>
      <td>0.101999</td>
      <td>...</td>
      <td>0.351033</td>
      <td>0.193199</td>
      <td>0.354910</td>
      <td>0.154284</td>
      <td>0.452391</td>
      <td>0.158464</td>
      <td>0.112984</td>
      <td>-0.090175</td>
      <td>0.003556</td>
      <td>-0.030193</td>
    </tr>
    <tr>
      <th>In re the Trust Estate of Sellers</th>
      <td>0.079893</td>
      <td>0.268354</td>
      <td>0.105641</td>
      <td>0.249614</td>
      <td>0.094754</td>
      <td>0.290682</td>
      <td>0.076005</td>
      <td>0.229837</td>
      <td>0.189160</td>
      <td>0.267667</td>
      <td>...</td>
      <td>0.111193</td>
      <td>0.292293</td>
      <td>0.226292</td>
      <td>0.238959</td>
      <td>0.275948</td>
      <td>0.005422</td>
      <td>0.243504</td>
      <td>0.023477</td>
      <td>0.172381</td>
      <td>0.189151</td>
    </tr>
    <tr>
      <th>Executive Council of the Protestant Episcopal Church in the Diocese of Delaware, Inc. v. Moss</th>
      <td>0.249499</td>
      <td>0.355512</td>
      <td>0.278486</td>
      <td>0.357694</td>
      <td>-0.007354</td>
      <td>0.274766</td>
      <td>0.145744</td>
      <td>0.081801</td>
      <td>0.164160</td>
      <td>0.187015</td>
      <td>...</td>
      <td>0.225622</td>
      <td>0.143511</td>
      <td>0.300511</td>
      <td>0.053125</td>
      <td>0.500617</td>
      <td>0.249337</td>
      <td>0.220781</td>
      <td>0.155378</td>
      <td>0.198856</td>
      <td>0.209124</td>
    </tr>
  </tbody>
</table>
<p>2361 rows × 2361 columns</p>
</div>




```python
case_of_interest = 'Braasch v. Galdi Securities Corp.'
source_idx = get_index(case_of_interest)

print(f"CASE OF INTEREST: {case_of_interest}")
if isinstance(source_idx, (int, np.integer)):
    print(df.at[source_idx, 'text'][:1500] + "...") 
print("\n")

# Add your Doc2Vec matrix to the list
matrices = [
    ("CountVectorizer", cv_cos_sim),
    ("TF-IDF", tfidf_cos_sim),
    ("Doc2Vec", d2v_similarity_matrix) # Added d2v here
]

for label, sim_df in matrices:
    # We use the case_of_interest to pull the relevant column/row
    # sort_values(ascending=False) puts the 1.0 (self-match) at index 0
    # .index[1] gets the next best match
    sorted_series = sim_df[case_of_interest].sort_values(ascending=False)
    
    most_similar_name = sorted_series.index[1]
    score = sorted_series.values[1]
    
    similar_idx = get_index(most_similar_name)
    
    print(f"Representation: {label}")
    print(f"MOST SIMILAR CASE: {most_similar_name}")
    print(f"SIMILARITY SCORE: {score:.4f}")
    print("-" * 30)
    
    if isinstance(similar_idx, (int, np.integer)):
        text_snippet = df.at[similar_idx, 'text'][:1000] + "..."
        print(text_snippet)
    print("\n")
```

    CASE OF INTEREST: Braasch v. Galdi Securities Corp.
    Short, Vice Chancellor:
    This case is before the court on defendants’ motion to dismiss for failure to state a claim on which relief can be granted.
    Plaintiffs are the owners of 5400 shares of the common stock of American Sumatra Tobacco Corporation (American Sumatra), a Delaware corporation. They here sue (1) individually on their own behalf; (2) representatively on behalf of all other stockholders of the corporation similarly situated, including those who sold their shares to the defendant N. V. Deli Maatschappij (Deli), a corporation of the Kingdom of the Netherlands, pursuant to an offer to buy made to all common stockholders; and (3) derivatively on behalf of American Sumatra.
    On June 28, 1960, Deli was the owner of more than fifty per cent of the common stock of American Sumatra. On that date it made an offer to all other stockholders to buy 202,338 shares of the common stock of American Sumatra at $17 per share. The offer resulted in Deli acquiring in excess of 200,000 additional shares of American Sumatra and increasing its stock ownership to more than ninety per cent of the outstanding shares of the company. On October 21, 1960 Deli organized, as its wholly owned subsidiary, Tobacco Holdings, Inc., a Delaware corporation, and thereupon transferred to Tobacco Holdings, Inc. all its shares of American Sumatra. In November, 1960 American Sumatra was merged into Tobacco Holdings, Inc. pursuant to the provisions of 8 Del.C. § 253. Immediately thereafter, the name of Tobacc...
    
    
    Representation: CountVectorizer
    MOST SIMILAR CASE: Abelow v. Symonds
    SIMILARITY SCORE: 0.6984
    ------------------------------
    Marvel, Vice Chancellor:
    This action by holders of common stock of Midstates Oil Corporation as originally designed sought an order restraining the proposed sale by that corporation of its assets and properties to the defendant, Middle States Petroleum Corporation, the owner of 95.93% of the common stock of Midstates. The original complaint was filed on December 29, 1958, the eve of a stockholders’ meeting called to approve such proposed sale of assets and consequent liquidation of Midstates under a plan which provided for the payment of $1,125 for each share of such stock to be surrendered under the plan. Also named as defendants in the action were present and former officers and directors of Midstates and Middle States, and Tennessee Gas Transmission Company, a corporation allegedly in control of Middle States which, according to the complaint, had through an exercise of such control brought about an exchange of stock whereby Tennessee Gas had become the holder of 92% of the outstand...
    
    
    Representation: TF-IDF
    MOST SIMILAR CASE: Abelow v. Symonds
    SIMILARITY SCORE: 0.6651
    ------------------------------
    Marvel, Vice Chancellor:
    This action by holders of common stock of Midstates Oil Corporation as originally designed sought an order restraining the proposed sale by that corporation of its assets and properties to the defendant, Middle States Petroleum Corporation, the owner of 95.93% of the common stock of Midstates. The original complaint was filed on December 29, 1958, the eve of a stockholders’ meeting called to approve such proposed sale of assets and consequent liquidation of Midstates under a plan which provided for the payment of $1,125 for each share of such stock to be surrendered under the plan. Also named as defendants in the action were present and former officers and directors of Midstates and Middle States, and Tennessee Gas Transmission Company, a corporation allegedly in control of Middle States which, according to the complaint, had through an exercise of such control brought about an exchange of stock whereby Tennessee Gas had become the holder of 92% of the outstand...
    
    
    Representation: Doc2Vec
    MOST SIMILAR CASE: Stauffer v. Standard Brands Inc.
    SIMILARITY SCORE: 0.6734
    ------------------------------
    Short, Vice Chancellor:
    Plaintiffs bring this action on behalf of themselves and all other stockholders of Planters Nut and Chocolate Company, a Delaware corporation [Planters of Delaware], similarly situated, to set aside the merger of Planters of Delaware into Standard Brands Incorporated, a Delaware corporation, under § 253 of the Delaware Corporation Law, or, in the alternative, for damages. Defendants have moved to dismiss the complaint for failure to state a claim on which relief can be granted.
    Prior to June, 1960 Planters Nut and Chocolate Company was a Pennsylvania corporation [Planters of Pennsylvania] with 229,667 shares of common stock outstanding, of which 114,345 shares were owned by five trustees of three trusts known as the “Obici Trusts.” In June, 1960 defendant Standard Brands Incorporated [Standard Brands] made an offer to the trustees to purchase their holdings of the common stock of Planters of Pennsylvania for a price of $105 per share. Two of the trustees agreed ...
    
    



```python
# save doc2vec
# d2v_model.save('d2v-vectors.pkl')
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
