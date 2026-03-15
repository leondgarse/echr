# Lab 4 - Corpus-Level Statistical Perspectives - Distributional NLP Without Machine Learning


In this lab, we will be looking further at pre-processing, tokenizing text, etc. and some more classical statistical corpus-level approaches.

Although these approaches are not at the cutting edge, they are can be very powerful and are useful developing a distributional thinking about text.

Because of the shifterator library, we will be using python version 3.9.

To use python version 3.9, make a new environment in anaconda navigator.



```python
!python --version
```

    Python 3.10.19



```python
!pip install nltk
!pip install gensim
!pip install -U scikit-learn
!pip install pandas
!pip install matplotlib
```

    Requirement already satisfied: nltk in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (3.9.2)
    Requirement already satisfied: click in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from nltk) (8.3.0)
    Requirement already satisfied: joblib in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from nltk) (1.5.2)
    Requirement already satisfied: regex>=2021.8.3 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from nltk) (2025.11.3)
    Requirement already satisfied: tqdm in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from nltk) (4.67.1)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Requirement already satisfied: gensim in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (4.4.0)
    Requirement already satisfied: numpy>=1.18.5 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from gensim) (2.2.6)
    Requirement already satisfied: scipy>=1.7.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from gensim) (1.15.3)
    Requirement already satisfied: smart_open>=1.8.1 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from gensim) (7.5.0)
    Requirement already satisfied: wrapt in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from smart_open>=1.8.1->gensim) (2.1.1)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Requirement already satisfied: scikit-learn in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (1.7.2)
    Requirement already satisfied: numpy>=1.22.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from scikit-learn) (2.2.6)
    Requirement already satisfied: scipy>=1.8.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from scikit-learn) (1.15.3)
    Requirement already satisfied: joblib>=1.2.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from scikit-learn) (1.5.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from scikit-learn) (3.6.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Requirement already satisfied: pandas in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (2.3.3)
    Requirement already satisfied: numpy>=1.22.4 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from pandas) (2.2.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from pandas) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from pandas) (2025.2)
    Requirement already satisfied: six>=1.5 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Requirement already satisfied: matplotlib in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (3.10.8)
    Requirement already satisfied: contourpy>=1.0.1 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (4.61.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (1.4.9)
    Requirement already satisfied: numpy>=1.23 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (2.2.6)
    Requirement already satisfied: packaging>=20.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (25.0)
    Requirement already satisfied: pillow>=8 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (12.0.0)
    Requirement already satisfied: pyparsing>=3 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (3.2.5)
    Requirement already satisfied: python-dateutil>=2.7 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: six>=1.5 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


# Text pre-processing, tokenization, featurization

# Tokenization 

Last time we introduced "tokenization."

As you might've noticed, the text data we are dealing with is mostly **strings**. There are a couple of problems with this:

* As you can imagine, a computer is not really able to "understand" what a "word" (or token). This is because in a "string", a whitespace is just another substring. To a computer, all letters and spaces are meaningless - it's just information to store.
* Thus, in order to ensure that we can parse the string, we must first identify the "words" (or tokens) in it. This is called [**tokenization**](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)
* In simplistic terms, tokenization is when we're trying to put some units of text (whether words or sentences) as items in a list.
* Furthermore, tokenization is usually an important step __prior__ to text analysis that we can do. 
* This goes back to the data structure distinction (between strings and lists for example) that we covered in previous labs. Some libraries can work with strings, but others require tokenization (a list of words) as input. When we think about it, this is just another way of and __inputting__ text into some function. 

To understand tokenization better, let's try a simple example


```python
sample_string = "The quick brown fox jumps over the lazy dog."
sample_string
```




    'The quick brown fox jumps over the lazy dog.'



 Now let's lowercase this string.


```python
text_lower = sample_string.lower() # lowercase 
text_lower
```




    'the quick brown fox jumps over the lazy dog.'



The simplest heuristic we can apply to tokenization is to use **"whitespace tokenizsation"** - meaning that every time a computer sees a whitespace, it will split the string. Let's try that:


```python
# Tokens
tokens = text_lower.split() # splits a string on white space
print(tokens)
```

    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog.']


One of the problems with this simplistic approach is that punctuations can be preserved as parts of tokens. Note the last word - "dog." 

Thus, the word is not "dog" but "dog." - even though to humans this might not seem like a big difference, but to a computer these two words are completely different. 


```python
tokens[-1] # the last element of the list
```




    'dog.'



**We can try to remove punctuation entirely.** But it's important to note that we will be losing some important information - and ideally we want to preserve as much as possible (but it also depends!)



There are many ways of removing punctuation. For example, we can use a Regular Expression tokenizer which matches a pattern which captures **only words** - click on this [regex pattern](https://regexr.com/6sfat). Note that because we match only words (**\w+**), the dot is not captured by the pattern.


```python
import nltk
nltk.download('punkt')

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

tokenizer.tokenize(text_lower)
```

    [nltk_data] Downloading package punkt to /home/leondgarse/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']



We can also use libraries for this - for instance, gensim. But the key point is there are many ways and libraries that can be used to tokenize text.


```python
from gensim.parsing.preprocessing import strip_punctuation

no_punctuation = strip_punctuation(text_lower)
no_punctuation
```




    'the quick brown fox jumps over the lazy dog '



As you can see, the string doesn't have a dot. Now we can again use the simple "split" syntax to split on whitespace.


```python
no_punctuation.split()
```




    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']



What about numbers? Let's change the string a little bit.


```python
text_with_numbers = "these 2 quick brown foxes jump over those 20 lazy dogs"
```


```python
tokens = text_with_numbers.split()
print(tokens)
```

    ['these', '2', 'quick', 'brown', 'foxes', 'jump', 'over', 'those', '20', 'lazy', 'dogs']


Below we'll use a [list comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) which gives a convenient way of creating a list out of an existing list.


```python
# Numbers
# remove numbers (keep if not a digit)
# Here - we're using a "list comprehension" which is basically a for loop that creates a new list
no_numbers = [t for t in tokens if not t.isdigit()]
print(no_numbers )
```

    ['these', 'quick', 'brown', 'foxes', 'jump', 'over', 'those', 'lazy', 'dogs']


Below, is the exact same code but we do the same thing with a for loop:



```python
no_numbers = []

for t in tokens:
    if not t.isdigit():
        no_numbers.append(t)
```


```python
no_numbers
```




    ['these', 'quick', 'brown', 'foxes', 'jump', 'over', 'those', 'lazy', 'dogs']




```python
# keep if not a digit, else replace with "#"
norm_numbers = [t if not t.isdigit() else '#' 
                for t in tokens ]
print(norm_numbers)
```

    ['these', '#', 'quick', 'brown', 'foxes', 'jump', 'over', 'those', '#', 'lazy', 'dogs']


## Stopwords

Another key concept that comes up when we deal with pre-processing is the so-called "stopwords" removal - words like "the", "and" etc. A word like "the" for example has no fundamental semantic meaning other than being a grammatical definite article.

But note - **just because we remove the word *the*, doesn't mean it's not informative.** In fact, many of these stop words are incredibly important, but for the purposes of simple counting and calculations, they can be removed, at least, in theory. 


```python
# Stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/leondgarse/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
stoplist = stopwords.words('english') 
print ("stop words:", stoplist)
```

    stop words: ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"]



```python
# keep the words if not a stopword
nostop = [t for t in tokens if t not in stoplist] # list comprehension
print(nostop)
```

    ['2', 'quick', 'brown', 'foxes', 'jump', '20', 'lazy', 'dogs']


There are different packages with different stopword lists. 

Here's an example from **scikit-learn**



```python
# scikit-learn stopwords
# depending on sklearn version, for sklearn==0.24.1, stop_words are here
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as stop_words
print(sorted(list(stop_words))[:200]) #first 200 stopwords

```

    ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over']


**Gensim** stopwords removing function:


```python
# gensim stopwords
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

remove_stopwords("Better late than never, but better never late.")
```




    'Better late never, better late.'



**Spacy** stopwords


```python
# In case the below code doesn't work. 
```


```python
!pip install -q spacy en_core_web_sm
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
!python -m spacy download en_core_web_sm
```

    Collecting en-core-web-sm==3.8.0
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl (12.8 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.8/12.8 MB[0m [31m6.2 MB/s[0m  [33m0:00:02[0ma [36m0:00:01[0mm eta [36m0:00:01[0m
    [?25h
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    [38;5;2m✔ Download and installation successful[0m
    You can now load the package via spacy.load('en_core_web_sm')



```python
import spacy                  ## install in Anaconda navigator - Because it won't work via "PIP install" in Python 3.9
import en_core_web_sm

nlp = en_core_web_sm.load()

```


```python
print(sorted(list(nlp.Defaults.stop_words))[:30])
```

    ["'d", "'ll", "'m", "'re", "'s", "'ve", 'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amount', 'an', 'and', 'another', 'any']


As we will see later, for the more CS-minded folks, stopword removal can also be thought of as __"dimensionality reduction"__ - ie, removing columns - more on this in the subsequent labs. 

### Noun Chunk phrases in `spacy`

Recall that we studied how to find phrases last lab with `phrasemachine` and `spacy`. 

Phrases can be an important. Just as in the previous lab, we can use spacy to extract basic "noun phrases" - i.e.  phrases that emerge from the text based on their use, but in a simpler syntax.

Spacy's [noun chunks](https://spacy.io/usage/linguistic-features#noun-chunks) are another way of extracting simple prhases. According to Spacy's documentation "You can think of noun chunks as a noun plus the words describing the noun – for example, “the lavish green grass” or “the world’s largest tech fund”."


```python
# Text to analyze
text = "The quick brown fox jumps over the lazy dog."

# Process the text
doc = nlp(text)

# Extract noun phrases
noun_phrases = [chunk.text for chunk in doc.noun_chunks]
print(noun_phrases)
```

    ['The quick brown fox', 'the lazy dog']


## Stemming

Stemming is reducing a word to its stem or root. For example, according to the [SnowballStemmer documentation](https://pypi.org/project/snowballstemmer/), "the English stemmer maps "connection", "connections", "connective", "connected", and "connecting" to the stem **connect**."


```python
## recall our token list
print(tokens)
```

    ['these', '2', 'quick', 'brown', 'foxes', 'jump', 'over', 'those', '20', 'lazy', 'dogs']



```python
# Stemming
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english') # snowball stemmer, english
```


```python
# remake list of tokens, replace with stemmed versions
tokens_stemmed = [stemmer.stem(t) for t in tokens]
print("Unstemmed tokens: \n",tokens)
print("Stemmed tokens: \n", tokens_stemmed)
```

    Unstemmed tokens: 
     ['these', '2', 'quick', 'brown', 'foxes', 'jump', 'over', 'those', '20', 'lazy', 'dogs']
    Stemmed tokens: 
     ['these', '2', 'quick', 'brown', 'fox', 'jump', 'over', 'those', '20', 'lazi', 'dog']


## Lemmatizing and Part of Speech (POS) Tagging

Lemmatizing is closely related to stemming, with the difference being that we reduce the word to its "lemma" rather than its "stem." A lemma is a more fundamental base form of a word. The difference is subtle, but a good example from wikipedia is:

* [Spacy](https://spacy.io/usage/spacy-101#features) gives the following examples: Assigning the base forms of words. For example, the lemma of **“are”** is **“be”**, and the lemma of **“rats”** is **“rat”.**
* This link is missed by stemming - just as with finding phrases, lemmatizing requires knowledge about the __Part of Speech__ tags of the word and the words around the word (whether a word is surrounded by Nouns or Verbs), and is thus a bit more intricate than stemming. 
* Therefore, lemmatization is related to stemming in that they are both methods for **normalizing text** - ie making the text more "standard", where it doesn't matter that you use "are" or "is" - both are reduced to "be."
* Depending on the context, one can prefer to lemmatize or stem their text (but usually, lemmatization is preferred - but again, it depends)
* These days, for the purposes of LLMs most of the text is kept in its original form because of the increase in computational power. However, for the purposes of this class, stemming/lemmatizing is still important as we won't be working with huge datasets.





```python
text_lower = 'these 2 brown foxes are better than 20 dogs'
```


```python
doc = nlp(text_lower)

for token in doc:
    print(token, 
          token.pos_, 
          token.lemma_)
```

    these DET these
    2 NUM 2
    brown ADJ brown
    foxes NOUN fox
    are AUX be
    better ADJ well
    than ADP than
    20 NUM 20
    dogs NOUN dog


Note: Keep in mind that POS tagging itself is not ideal, as the example below shows. The lemma of "better" should be "good", but because it's recorded as an adjective by the POS tagger, it becomes "well" 

## Sentence tokenization


Tokenizing sentences is just like tokenizing words - the difference being is that each item in a list is going to be a sentence rather than a token. This is useful - sometimes, for your research purposes, the __unit of analysis__ might be sentences and not individual words.


Let's modify our running example a little bit by adding a fun quote from David Foster Wallace 


```python
text = "The quick brown fox jumps over the lazy dog. And yet: why not be someone who stays up all night torturing himself mentally over the question of whether or not there's a dog?"
print(text)
```

    The quick brown fox jumps over the lazy dog. And yet: why not be someone who stays up all night torturing himself mentally over the question of whether or not there's a dog?


Let's try nltk's sentence tokenizer


```python
nltk.download('punkt_tab')
from nltk import sent_tokenize
```

    [nltk_data] Downloading package punkt_tab to
    [nltk_data]     /home/leondgarse/nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!



```python
sentences = sent_tokenize(text) ### sentence tokenization
sentences
```




    ['The quick brown fox jumps over the lazy dog.',
     "And yet: why not be someone who stays up all night torturing himself mentally over the question of whether or not there's a dog?"]



Now let's try spacy's sentence tokenization


```python
doc = nlp(text)
sentences = list(doc.sents)
sentences
```




    [The quick brown fox jumps over the lazy dog.,
     And yet: why not be someone who stays up all night torturing himself mentally over the question of whether or not there's a dog?]



Although the outputs are the same, what's going on behind the scenes is not. 
You should note that the `spacy` sentence tokenization is based on a nlp model (which we call using `nlp()` syntax - and this model takes into account more information that `nltk`'s sentence tokenizer.

* Thus, `spacy` be good if you want high quality sentences
* But since it's a bit more complex, it's computationally more expensive, and thus could take longer than `nltk`

# Zipf's law - Statistical regularities in language

One of the most fascinating and robust empirical regularities in human language is that word frequencies are highly uneven. A small number of words occur extremely often, while the vast majority of words appear rarely. 

This pattern is known as a [**Zipfian distribution**](https://en.wikipedia.org/wiki/Zipf%27s_law) or Zipf-Mandelbrot's law, after the linguist George Kingsley Zipf who discovered this phenomenon in 1932 (but there are earlier discoverers as well). The famous mathematician Mandelbrot generalized this law. 

Zipf’s law states that if words are ranked by frequency the frequency of a word is approximately inversely proportional to its rank. 

In practice, this means:
- the most frequent word occurs about approximately twice as often as the second most frequent,
- three times as often as the third, and so on.

What makes Zipf’s law remarkable and rather hypnotizing is that it is not language-specific, genre-specific, or author-specific. In lingustic data, it appears across:

- different languages,

- different historical periods,

- different authors and styles,

- spoken and written text.


[Zipf-Mandelbrot's law](https://en.wikipedia.org/wiki/Zipf%E2%80%93Mandelbrot_law) (which is an example of a [power law distribution](https://en.wikipedia.org/wiki/Power_law)), surprisingly, appears in many other domains, including natural phenomena. Even in a text with random characters, it holds. It also appeared when people studied the [size of cities in US](https://blogs.cornell.edu/info2040/2016/11/13/zipfs-law-for-cities-a-simple-explanation-for-urban-populations/) but it seems to not apply to [Chinese cities](https://mathoverflow.net/a/39232). Consider the related idea of a Pareto distribution.

Zipf's law is found in things like [income distirubution of Japanese companies](https://www.sciencedirect.com/science/article/abs/pii/S0378437199000862?via%3Dihub) to [citations of academic articles](https://content.wolfram.com/sites/13/2018/02/11-6-4.pdf) - so it seems to be a part of some kind of a larger  phenomenon. 


```python
import glob
import os
import pandas as pd

def load_corpus(base_path, author):
    rows = []
    for filepath in glob.glob(os.path.join(base_path, "*.txt")):
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        rows.append({
            "filename": os.path.basename(filepath),
            "authorship": author,
            "text": text
        })
    return rows
```


```python
data = []
data = data + load_corpus("data/plato_works", "Plato")
data = data + load_corpus("data/aristotle", "Aristotle")
```


```python
df_plato = pd.DataFrame(data)
```


```python
df_plato
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
      <th>filename</th>
      <th>authorship</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apology.txt</td>
      <td>Plato</td>
      <td>Apology\n\nApology\nBy Plato\nCommentary:\nQui...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>charmides.txt</td>
      <td>Plato</td>
      <td>Charmides, or Temperance\n\nCharmides, or Temp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cratylus.txt</td>
      <td>Plato</td>
      <td>Cratylus\n\nCratylus\nBy Plato\nCommentary:\nA...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>critias.txt</td>
      <td>Plato</td>
      <td>Critias\n\nCritias\nBy Plato\nCommentary:\nMan...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>crito.txt</td>
      <td>Plato</td>
      <td>Crito\n\nCrito\nBy Plato\nCommentary:\nMany co...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>euthydemus.txt</td>
      <td>Plato</td>
      <td>Euthydemus\n\nEuthydemus\nBy Plato\nCommentary...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>euthyfro.txt</td>
      <td>Plato</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Plato/eu...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>gorgias.txt</td>
      <td>Plato</td>
      <td>Gorgias\n\nGorgias\nBy Plato\nCommentary:\nMan...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ion.txt</td>
      <td>Plato</td>
      <td>Ion\n\nIon\nBy Plato\nCommentary:\nSeveral com...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>laches.txt</td>
      <td>Plato</td>
      <td>Laches, or Courage\n\nLaches, or Courage\nBy P...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>laws.txt</td>
      <td>Plato</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>lysis.txt</td>
      <td>Plato</td>
      <td>Lysis, or Friendship\n\nLysis, or Friendship\n...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>meno.txt</td>
      <td>Plato</td>
      <td>Meno\n\nMeno\nBy Plato\nCommentary:\nMany comm...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>parmenides.txt</td>
      <td>Plato</td>
      <td>Parmenides\n\nParmenides\nBy Plato\nCommentary...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>phaedo.txt</td>
      <td>Plato</td>
      <td>Phaedo\n\nPhaedo\nBy Plato\nCommentary:\nSever...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>phaedrus.txt</td>
      <td>Plato</td>
      <td>Phaedrus\n\nPhaedrus\nBy Plato\nCommentary:\nM...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>philebus.txt</td>
      <td>Plato</td>
      <td>Philebus\n\nPhilebus\nBy Plato\nCommentary:\nA...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>protagoras.txt</td>
      <td>Plato</td>
      <td>Protagoras\n\nProtagoras\nBy Plato\nCommentary...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>republic.txt</td>
      <td>Plato</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>seventh_letter.txt</td>
      <td>Plato</td>
      <td>The Seventh Letter\nBy Plato\n\n\nTranslated b...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>sophist.txt</td>
      <td>Plato</td>
      <td>Sophist\n\nSophist\nBy Plato\nCommentary:\nA f...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>stateman.txt</td>
      <td>Plato</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Plato/st...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>symposium.txt</td>
      <td>Plato</td>
      <td>Symposium\n\nSymposium\nBy Plato\nCommentary:\...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>theatetus.txt</td>
      <td>Plato</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Plato/th...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>timaeus.txt</td>
      <td>Plato</td>
      <td>Timaeus\n\nTimaeus\nBy Plato\nCommentary:\nSev...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>athenian_const.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>categories.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>gait_anim.txt</td>
      <td>Aristotle</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Aristotl...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>gener_corr.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>heavens.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>history_anim.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>interpretation.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>longev_short.txt</td>
      <td>Aristotle</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Aristotl...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>memory.txt</td>
      <td>Aristotle</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Aristotl...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>metaphysics.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>meteorology.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>nicomachaen.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>on_dreams.txt</td>
      <td>Aristotle</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Aristotl...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>parts_animals.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>physics.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>poetics.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>politics.txt</td>
      <td>Aristotle</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Aristotl...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>posterior.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>prior.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>prophesying.txt</td>
      <td>Aristotle</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Aristotl...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>rhetoric.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>sense.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>sleep.txt</td>
      <td>Aristotle</td>
      <td>\n&lt;BASE HREF="http://classics.mit.edu/Aristotl...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>sophist_refut.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>soul.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>topics.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>youth_old.txt</td>
      <td>Aristotle</td>
      <td>Provided by The Internet Classics Archive.\nSe...</td>
    </tr>
  </tbody>
</table>
</div>




```python
import re
from nltk.tokenize import word_tokenize

def tokenize(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if re.fullmatch(r"[a-z]+", t)]
    return tokens

df_plato["tokens"] = df_plato["text"].apply(tokenize)
```


```python
df_plato.head()
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
      <th>filename</th>
      <th>authorship</th>
      <th>text</th>
      <th>tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apology.txt</td>
      <td>Plato</td>
      <td>Apology\n\nApology\nBy Plato\nCommentary:\nQui...</td>
      <td>[apology, apology, by, plato, commentary, quit...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>charmides.txt</td>
      <td>Plato</td>
      <td>Charmides, or Temperance\n\nCharmides, or Temp...</td>
      <td>[charmides, or, temperance, charmides, or, tem...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cratylus.txt</td>
      <td>Plato</td>
      <td>Cratylus\n\nCratylus\nBy Plato\nCommentary:\nA...</td>
      <td>[cratylus, cratylus, by, plato, commentary, a,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>critias.txt</td>
      <td>Plato</td>
      <td>Critias\n\nCritias\nBy Plato\nCommentary:\nMan...</td>
      <td>[critias, critias, by, plato, commentary, many...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>crito.txt</td>
      <td>Plato</td>
      <td>Crito\n\nCrito\nBy Plato\nCommentary:\nMany co...</td>
      <td>[crito, crito, by, plato, commentary, many, co...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_plato["n_tokens"] = df_plato["tokens"].apply(len) ## word count
df_plato.head()
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
      <th>filename</th>
      <th>authorship</th>
      <th>text</th>
      <th>tokens</th>
      <th>n_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apology.txt</td>
      <td>Plato</td>
      <td>Apology\n\nApology\nBy Plato\nCommentary:\nQui...</td>
      <td>[apology, apology, by, plato, commentary, quit...</td>
      <td>11433</td>
    </tr>
    <tr>
      <th>1</th>
      <td>charmides.txt</td>
      <td>Plato</td>
      <td>Charmides, or Temperance\n\nCharmides, or Temp...</td>
      <td>[charmides, or, temperance, charmides, or, tem...</td>
      <td>10625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cratylus.txt</td>
      <td>Plato</td>
      <td>Cratylus\n\nCratylus\nBy Plato\nCommentary:\nA...</td>
      <td>[cratylus, cratylus, by, plato, commentary, a,...</td>
      <td>12417</td>
    </tr>
    <tr>
      <th>3</th>
      <td>critias.txt</td>
      <td>Plato</td>
      <td>Critias\n\nCritias\nBy Plato\nCommentary:\nMan...</td>
      <td>[critias, critias, by, plato, commentary, many...</td>
      <td>6750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>crito.txt</td>
      <td>Plato</td>
      <td>Crito\n\nCrito\nBy Plato\nCommentary:\nMany co...</td>
      <td>[crito, crito, by, plato, commentary, many, co...</td>
      <td>5351</td>
    </tr>
  </tbody>
</table>
</div>




```python
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

all_tokens = [t for tokens in df_plato["tokens"] for t in tokens]
freqs = Counter(all_tokens)
```


```python
zipf_df = (
    pd.DataFrame(freqs.items(), columns=["word", "freq"])
      .sort_values("freq", ascending=False)
      .reset_index(drop=True)
)

zipf_df["rank"] = zipf_df.index + 1
```


```python
zipf_df
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
      <th>word</th>
      <th>freq</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>the</td>
      <td>107952</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>of</td>
      <td>60275</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>and</td>
      <td>58739</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>is</td>
      <td>45707</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>41439</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22715</th>
      <td>oddest</td>
      <td>1</td>
      <td>22716</td>
    </tr>
    <tr>
      <th>22716</th>
      <td>tuck</td>
      <td>1</td>
      <td>22717</td>
    </tr>
    <tr>
      <th>22717</th>
      <td>splayed</td>
      <td>1</td>
      <td>22718</td>
    </tr>
    <tr>
      <th>22718</th>
      <td>sprawls</td>
      <td>1</td>
      <td>22719</td>
    </tr>
    <tr>
      <th>22719</th>
      <td>turtles</td>
      <td>1</td>
      <td>22720</td>
    </tr>
  </tbody>
</table>
<p>22720 rows × 3 columns</p>
</div>




```python
plt.figure(figsize=(6,4))

plt.scatter(
    np.log(zipf_df["rank"]),
    np.log(zipf_df["freq"]),
    s=10
)

plt.xlabel("log(rank)")
plt.ylabel("log(frequency)")
plt.title("Zipfian Distribution: Plato + Aristotle")
plt.show()
```

Let's see the authors' seperate distributiosn.


```python
author_tokens = (
    df_plato.groupby("authorship")["tokens"]
      .apply(lambda x: [t for tokens in x for t in tokens])
)
```


```python
rows = []

for author, tokens in author_tokens.items():
    freqs = Counter(tokens)
    temp = (
        pd.DataFrame(freqs.items(), columns=["word", "freq"])
          .sort_values("freq", ascending=False)
          .reset_index(drop=True)
    ) 
    temp["rank"] = temp.index + 1
    temp["authorship"] = author
    rows.append(temp)
```


```python
zipf_df = pd.concat(rows, ignore_index=True)
zipf_df
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
      <th>word</th>
      <th>freq</th>
      <th>rank</th>
      <th>authorship</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>the</td>
      <td>73923</td>
      <td>1</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>of</td>
      <td>37689</td>
      <td>2</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>is</td>
      <td>34275</td>
      <td>3</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>and</td>
      <td>31620</td>
      <td>4</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>25252</td>
      <td>5</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32327</th>
      <td>interrupting</td>
      <td>1</td>
      <td>14487</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>32328</th>
      <td>ateires</td>
      <td>1</td>
      <td>14488</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>32329</th>
      <td>cruelty</td>
      <td>1</td>
      <td>14489</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>32330</th>
      <td>chrysippus</td>
      <td>1</td>
      <td>14490</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>32331</th>
      <td>colourless</td>
      <td>1</td>
      <td>14491</td>
      <td>Plato</td>
    </tr>
  </tbody>
</table>
<p>32332 rows × 4 columns</p>
</div>




```python
plt.figure(figsize=(6,4))

for author, subdf in zipf_df.groupby("authorship"):
    plt.scatter(
        np.log(subdf["rank"]),
        np.log(subdf["freq"]),
        s=10,
        label=author
    )

plt.xlabel("log(rank)")
plt.ylabel("log(freq)")
plt.title("Zipfian Distributions by Author")
plt.legend()
plt.show()
```

Words that are used only once in a corpus are known as the [Hapax Legomenon](https://en.wikipedia.org/wiki/Hapax_legomenon) - and we can see that there are quite a lot of them in the corpus we study. 


```python
hapax_df = (
    zipf_df
    .query("freq == 1")
    .reset_index(drop=True)
)
```


```python
hapax_df
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
      <th>word</th>
      <th>freq</th>
      <th>rank</th>
      <th>authorship</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>culture</td>
      <td>1</td>
      <td>11824</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arterial</td>
      <td>1</td>
      <td>11825</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mercly</td>
      <td>1</td>
      <td>11826</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dilates</td>
      <td>1</td>
      <td>11827</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>suppuration</td>
      <td>1</td>
      <td>11828</td>
      <td>Aristotle</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11106</th>
      <td>interrupting</td>
      <td>1</td>
      <td>14487</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>11107</th>
      <td>ateires</td>
      <td>1</td>
      <td>14488</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>11108</th>
      <td>cruelty</td>
      <td>1</td>
      <td>14489</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>11109</th>
      <td>chrysippus</td>
      <td>1</td>
      <td>14490</td>
      <td>Plato</td>
    </tr>
    <tr>
      <th>11110</th>
      <td>colourless</td>
      <td>1</td>
      <td>14491</td>
      <td>Plato</td>
    </tr>
  </tbody>
</table>
<p>11111 rows × 4 columns</p>
</div>




```python
hapax_df.groupby("authorship").size()
```




    authorship
    Aristotle    6018
    Plato        5093
    dtype: int64



Why is Zipf's law important? It is a constraint that any language model and any large enough corpus **must** obey. 

As we will see next week, rare vocabluary in machine learning contexts tends to be ignored or subdivided into smaller words - so these words are seen as from a statistical lens as outliers that do not occur often. However, for corpus linguistics and just human comprehension, rare words tend to be pretty important.

For Natural Language Processing and Large Language Models in particular - Zipf's law poses another challenge in that frequent words appear much more often in the training data, while rare words like the "hapaxes" appear much rarer. To quote a [recent paper on Zipf's law and LLMs](https://arxiv.org/pdf/2511.17575):

- "short, frequent tokens are easily retrieved from context;
- long or rare tokens may fall outside the context window and effectively behave as unseen events"

We'll explore more of this later.

Note that, if you lemmatize/stem words, you'll get a somewhat different distribution, but roughly the same. 

For example, a word like "muddy" will be included under the word "mud" - thus counting it as 2 occurances. However, this also results in information loss because muddy is not mud.

# Comparing texts - some exploratory data analysis

## Shifterator

[Shifterator](https://shifterator.readthedocs.io/en/latest/cookbook/getting_started.html#case-study) is a  library which helps us intuitively understand and visualize the classical question of "comparing texts" - in the words of the creators "word shift graphs are interpretable horizontal bar charts for visualizing how any two texts compare according to a given measure."  

* Firstly, there are a number of different __measures__ that are discussed in the paper. From relative frequencies of words to "Jensen-Shannon divergence" - which is actually pretty commonly used in NLP. For the purposes of the class, it's not important to know the __math__ behind these measures - what's important is that some measures have certain benefits over others - and thus, visualizations will come out different. Essentially, all they do is show difference between texts. 
* The data input that it expects is a frequency count of words in a given text - and the `clean_text` function below does just that.




```python
!pip install shifterator
import shifterator as sh
```

    Requirement already satisfied: shifterator in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (0.3.0)
    Requirement already satisfied: matplotlib in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from shifterator) (3.10.8)
    Requirement already satisfied: numpy in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from shifterator) (2.2.6)
    Requirement already satisfied: contourpy>=1.0.1 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (4.61.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (1.4.9)
    Requirement already satisfied: packaging>=20.0 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (25.0)
    Requirement already satisfied: pillow>=8 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (12.0.0)
    Requirement already satisfied: pyparsing>=3 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (3.2.5)
    Requirement already satisfied: python-dateutil>=2.7 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from matplotlib->shifterator) (2.9.0.post0)
    Requirement already satisfied: six>=1.5 in /media/leondgarse/DATA/virtualenvs/workon310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib->shifterator) (1.17.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m25.3[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
from pathlib import Path

# Define the paths to your files
denning_path = Path('data/cases/miller_v_jackson/denning_minority.txt')
lane_path = Path('data/cases/miller_v_jackson/lane_majority.txt')
```


```python
denning_file = open(denning_path, 'r')  # Open the file
denning_content = denning_file.read()  # Read the content

print(denning_content)  # Print the content
denning_file.close()  # You can close the file to free up resources (it's not that big of a file though)
```


```python
lane_file = open(lane_path, 'r')  # Open the file
lane_content = lane_file.read()  # Read the content

print(lane_content)  # Print the content
lane_file.close()  # You can close the file to free up resources (it's not that big of a file though)
```


```python
from collections import Counter
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

denning_content = strip_punctuation(remove_stopwords(denning_content.lower()))
lane_content = strip_punctuation(remove_stopwords(lane_content.lower()))
```


```python
lane_content
```

Let's now clean the text and create a counter object (frequency of words), which is required to work with shifterator


```python
denning_counter = Counter(denning_content.split())
lane_counter = Counter(lane_content.split())
```

Now that we have our data in the appropriate format for the shifterator, we can try to examine the different measures shifterator provides. 

* The [shifterator cookbook](https://shifterator.readthedocs.io/en/latest/cookbook/index.html) goes over these in great detail. 
* Keep in mind that once we have the data, we don't need to make it again. 
* Also, even though I removed the stopwords, technically, this is not necessary as some measures (like the shannon entropy shift) can safely ignore stopwords. But I did that just in case.


Let's start with the __proportion frequency shift__ 


```python
proportion_shift = sh.ProportionShift(type2freq_1 = denning_counter,
                                      type2freq_2 = lane_counter)
try:
    proportion_shift.get_shift_graph(system_names = ['Denning', 'Lane'],
                                 title='Proportion Shift of Opinions ')
except AttributeError:  ## to supprress the attribute error 
    pass
```


```python
import collections
import collections.abc

# Manually add the reference back where the library expects it
collections.Mapping = collections.abc.Mapping

entropy_shift = sh.EntropyShift(type2freq_1 = denning_counter,
                                type2freq_2 = lane_counter,
                                base = 2)
try:
    entropy_shift.get_shift_graph(system_names = ['Denning', 'Lane'],
                                 title='Proportion Shift of Opinions ')
except AttributeError:  ## to supprress the attribute error 
    pass

```


```python
jsd_shift = sh.JSDivergenceShift(type2freq_1 = denning_counter,
                                 type2freq_2 = lane_counter,
                                 weight_1=0.5,
                                 weight_2=0.5,
                                 base=2,
                                 alpha=1)
try:
    jsd_shift.get_shift_graph(); 
except AttributeError:  ## to supprress the attribute error 
    pass
```

Do you see any differences between these measures? What words are emphasized?

In order to answer this, you have to read the brief explanation that is provided in the [shifterator cookbook](https://shifterator.readthedocs.io/en/latest/cookbook/frequency_shifts.html#jensen-shannon-divergence-shifts)


```python

```

## Scattertext


We are going to use [scattertext](https://github.com/JasonKessler/scattertext) to visualize text comparisons. 

Note: scattertext is more taxing than a simple measure like "Jensen Shannon Divergence" - so use it on small/medium corpuses.


```python
!pip install scattertext
!pip install plotly
import scattertext as st
from pathlib import Path
import pandas as pd
```

To work with scattertext we'll need two things:

* A column that labels our data into two distinct classes - in this case `Party` (the judge) - because we care about the distinction between Denning (1) vs Lane (0)
* Spacy's NLP model


```python
# Define the paths to your files
denning_path = Path('data/cases/miller_v_jackson/denning_minority.txt')
lane_path = Path('data/cases/miller_v_jackson/lane_majority.txt')

denning_content = denning_path.read_text()
lane_content =  lane_path.read_text()
```


```python
judge_df = pd.DataFrame({
    'party': ['Denning', 'Lane'],
    'text': [denning_content, lane_content]
})
```


```python
judge_df
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
      <th>party</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Denning</td>
      <td>In summertime village cricket is the delight o...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lane</td>
      <td>Since about 1905 cricket has been played on a ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
corpus = st.CorpusFromPandas(judge_df,
                             category_col='party',
                             text_col='text',
                             nlp=st.whitespace_nlp_with_sentences).build()
```


```python
term_freq_df = corpus.get_term_freq_df()
term_freq_df['Denning score'] = corpus.get_scaled_f_scores('Denning')
```


```python
print(list(term_freq_df.sort_values(by='Denning score', ascending=False).index[:20]))
```

    ['cricket club', 'the cricket', 'cricket ground', 'club', 'he', 'could', 'his', 'an injunction', 'but', 'ball', 'page', 'very', 'right', 'has', 'cricket', 'ground', 'if', 'at page', 'i', 'case']



```python
from IPython.core.display import display, HTML
from IPython.display import IFrame
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))
%matplotlib inline
```

    /tmp/ipykernel_105389/2738240820.py:1: DeprecationWarning: Importing display from IPython.core.display is deprecated since IPython 7.14, please import from IPython.display
      from IPython.core.display import display, HTML
    /tmp/ipykernel_105389/2738240820.py:3: DeprecationWarning: Importing display from IPython.core.display is deprecated since IPython 7.14, please import from IPython.display
      from IPython.core.display import display, HTML



<style>.container { width:98% !important; }</style>



```python
import plotly
html = st.produce_scattertext_explorer(corpus,
          category='Denning',
          category_name='Denning',
          not_category_name='Lane',
          minimum_term_frequency = 3, 
          pmi_threshold_coefficient = 3,
          transform=st.Scalers.dense_rank, 
          width_in_pixels=800)

```


```python
file_name = 'Denning vs Lane.html'         
with open(file_name, 'wb') as fn:
    fn.write(html.encode('utf-8'))              

display(IFrame(file_name, width=1000, height=650))
```

Produce HTML file - note that the parameters set here can determine how the visualization looks like - in particular "minimum_term_frequency" and "pmi_threshold_coef"


```python
# Use raw string for Windows path
tweets_path = Path(r"data/tweets.csv")

# Read CSV into pandas DataFrame
tweets_df = pd.read_csv(tweets_path)

```


```python
tweets_df
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
      <th>id</th>
      <th>text</th>
      <th>isRetweet</th>
      <th>isDeleted</th>
      <th>device</th>
      <th>favorites</th>
      <th>retweets</th>
      <th>date</th>
      <th>isFlagged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98454970654916608</td>
      <td>Republicans and Democrats have both created ou...</td>
      <td>f</td>
      <td>f</td>
      <td>TweetDeck</td>
      <td>49</td>
      <td>255</td>
      <td>2011-08-02 18:07:48</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1234653427789070336</td>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>73748</td>
      <td>17404</td>
      <td>2020-03-03 01:34:50</td>
      <td>f</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1218010753434820614</td>
      <td>RT @CBS_Herridge: READ: Letter to surveillance...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>7396</td>
      <td>2020-01-17 03:22:47</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1304875170860015617</td>
      <td>The Unsolicited Mail In Ballot Scam is a major...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>80527</td>
      <td>23502</td>
      <td>2020-09-12 20:10:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1218159531554897920</td>
      <td>RT @MZHemingway: Very friendly telling of even...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>9081</td>
      <td>2020-01-17 13:13:59</td>
      <td>f</td>
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
    </tr>
    <tr>
      <th>56566</th>
      <td>1319485303363571714</td>
      <td>RT @RandPaul: I don’t know why @JoeBiden think...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>20683</td>
      <td>2020-10-23 03:46:25</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56567</th>
      <td>1319484210101379072</td>
      <td>RT @EliseStefanik: President @realDonaldTrump ...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>9869</td>
      <td>2020-10-23 03:42:05</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56568</th>
      <td>1319444420861829121</td>
      <td>RT @TeamTrump: LIVE: Presidential Debate #Deba...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>8197</td>
      <td>2020-10-23 01:03:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56569</th>
      <td>1319384118849949702</td>
      <td>Just signed an order to support the workers of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>176289</td>
      <td>36001</td>
      <td>2020-10-22 21:04:21</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56570</th>
      <td>1319345719829008387</td>
      <td>Suburban women want Safety &amp;amp; Security. Joe...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>95169</td>
      <td>19545</td>
      <td>2020-10-22 18:31:46</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>56571 rows × 9 columns</p>
</div>




```python
# Assume tweets_df is already loaded
tweets_df['popularity'] = tweets_df['retweets'].apply(lambda x: 'Popular' if x >= 10000 else 'LessPopular')

# Quick check
tweets_df[['text', 'retweets', 'popularity']]
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
      <th>text</th>
      <th>retweets</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republicans and Democrats have both created ou...</td>
      <td>255</td>
      <td>LessPopular</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>17404</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RT @CBS_Herridge: READ: Letter to surveillance...</td>
      <td>7396</td>
      <td>LessPopular</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Unsolicited Mail In Ballot Scam is a major...</td>
      <td>23502</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RT @MZHemingway: Very friendly telling of even...</td>
      <td>9081</td>
      <td>LessPopular</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56566</th>
      <td>RT @RandPaul: I don’t know why @JoeBiden think...</td>
      <td>20683</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>56567</th>
      <td>RT @EliseStefanik: President @realDonaldTrump ...</td>
      <td>9869</td>
      <td>LessPopular</td>
    </tr>
    <tr>
      <th>56568</th>
      <td>RT @TeamTrump: LIVE: Presidential Debate #Deba...</td>
      <td>8197</td>
      <td>LessPopular</td>
    </tr>
    <tr>
      <th>56569</th>
      <td>Just signed an order to support the workers of...</td>
      <td>36001</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>56570</th>
      <td>Suburban women want Safety &amp;amp; Security. Joe...</td>
      <td>19545</td>
      <td>Popular</td>
    </tr>
  </tbody>
</table>
<p>56571 rows × 3 columns</p>
</div>




```python
percentage = tweets_df['popularity'].value_counts(normalize=True) * 100
percentage
```




    popularity
    LessPopular    67.614149
    Popular        32.385851
    Name: proportion, dtype: float64




```python
tweets_sample = tweets_df.sample(n=10_000, random_state=42)
```


```python
corpus = st.CorpusFromPandas(
            tweets_sample,
            category_col='popularity',
            text_col='text',
            nlp=st.whitespace_nlp_with_sentences
            ).build()
```


```python
html = st.produce_scattertext_explorer(corpus,
    category='Popular',
    category_name='Popular Tweets',
    not_category_name='Less Popular Tweets',
    minimum_term_frequency=5,
    pmi_threshold_coefficient=3,
    transform=st.Scalers.dense_rank,
    width_in_pixels=900
)

display(HTML(html))
```


```python
with open('foo.html', 'w') as ff:
    ff.write(html)
```

Now try this with Oakley decision in the cases folder.


```python
# In case someone is interested in another Fighting words implementation 
#!git clone https://github.com/jmhessel/FightingWords.git ## Monroe, Quinn et al fightingwords implementation in python by jmhessel
#!https://gist.github.com/xandaschofield/3c4070b2f232b185ce6a09e47b4e7473 ## fighting words visualization implementation by Xanda Schofield 

## for recent fighting words implementation - see: https://github.com/AidaCPL/Fighting_Words_News/blob/main/code/FightingWords_Code_Documentation.ipynb
```
