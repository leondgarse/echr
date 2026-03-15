# Lab 2 

**Introduction to Jupyter Notebooks, Data Structures, and Pandas**

The basic strategies and tools for data analysis covered in this notebook will be the foundations of this class. It will cover an overview of our software and some programming concepts.



# **Our Computing Environment, Jupyter notebooks** 
This webpage is called a "Jupyter notebook" (we'll call it "notebook" for short). A notebook is a place to **write programs** and **view their results and output**. 

## Text cells
In a notebook, each rectangle containing text or code is called a *cell*.

Text cells (like this one) can be edited by double-clicking on them. They're written in a simple format called [Markdown](http://daringfireball.net/projects/markdown/syntax) to add formatting and section headings. 

After you edit a text cell, click the "run cell" button at the top that looks like ▶| to confirm any changes. (Try not to delete the instructions of the lab.) 

Tip: You can also use a shortcut of **`Shift + Return (enter)`** to "run" this cell.

**Understanding Check 1** This paragraph is in its own text cell.  
Try editing it so that this sentence is the last sentence in the paragraph, and then click the "run cell" ▶| button or "shift+enter". This sentence, for example, should be deleted.  So should this one.

## Code cells
Other cells contain code in the Python 3 language. Running a code cell will execute all of the code it contains.

To run the code in a code cell, first click on that cell to activate it.  It'll be highlighted with a little green or blue rectangle.  Next, either press ▶| or again, hold down the `shift` key and press `return` or `enter`.

The fundamental building block of Python code is an expression. Cells can contain multiple lines with multiple expressions. 

When you run a cell, the lines of code are executed in the order in which they appear. Every `print` expression prints a line. Run the next cell and notice the order of the output.

Note: Notice the change from "Markdown" to "Code" on the top.


```python
print("First this line is printed,")
print("and then this one.")
print("Hello World")
print("\N{WAVING HAND SIGN}, \N{EARTH GLOBE AMERICAS}!")
```

    First this line is printed,
    and then this one.
    Hello World
    👋, 🌎!


If a restart happens, you should **rerun** any cells with imports, variables, and loaded data.

## Writing Jupyter notebooks
You can use Jupyter notebooks for your own projects or documents.  When you make your own notebook, you'll need to create your own cells for text and code.

To add a cell, click the **`+`** button in the menu bar.  It'll start out as a text cell.  You can change it to a code cell by clicking inside it so it's highlighted, clicking the drop-down box next to the restart (⟳) button in the menu bar, and choosing "Code".

Tip: as a shortcut, you can also select the cell (single click or click on the left of it) and press **`A`** to add a cell above or **`B`** to add a cell below. 

## Errors
Python is a language, and like natural human languages, it has rules.  It differs from natural language in two important ways:

1. The rules are **simple**.  You can learn most of them in a few weeks and gain reasonable proficiency with the language in a semester.
2. The rules are **rigid**.  If you're proficient in a natural language, you can understand a non-proficient speaker, glossing over small mistakes.  A computer running Python code is not smart enough to do that.

Whenever you write code, you'll make mistakes.  When you run a code cell that has errors, Python will sometimes produce error messages to tell you what you did wrong.

**Errors are okay**; even experienced programmers make many errors.  When you make an error, you just have to find the source of the problem, fix it, and move on.

We have made an error in the next cell.  Run it and see what happens.


```python
print("This line is missing something."
```


      Cell In[5], line 1
        print("This line is missing something."
                                               ^
    SyntaxError: incomplete input



The last line of the error output attempts to tell you what went wrong.  The **syntax** of a language is its structure, and this **`SyntaxError`** tells you that you have created an illegal structure.  **"`EOF`"** means "end of file," so the message is saying Python expected you to write something more (in this case, a right parenthesis) before finishing the cell.

There's a lot of terminology in programming languages, but you don't need to know it all in order to program effectively. If you see a cryptic message like this, you can often get by without deciphering it.  

At this day an age, it is also just possible to ask google and/or ChatGPT or whatever AI you are using to fix the error.

**Understanding Check** Try to fix the code above so that you can run the cell and see the intended message instead of an error.



# **Python basics -  Introduction to Programming Concepts and Data** 
Before getting into the more advanced analysis techniques that will be required in this course, we need to cover a few of the foundational elements of programming in Python.

## Expressions
The departure point for all programming is the concept of the **expression**. 

An expression is a combination of variables, operators, and other Python elements that the language interprets and acts upon. Expressions act as a set of instructions to be fed through the interpreter, with the goal of generating specific outcomes. See below for some examples of basic expressions.


```python
# Examples of expressions:

#addition
print(2 + 2)

#string concatenation 
print('me' + ' and I')

#you can print a number with a string if you cast it 
print("me" + str(2))

#exponents
print(12 ** 2)

```

    4
    me and I
    me2
    144


You will notice that only the last line in a cell gets printed out. If you want to see the values of previous expressions, you need to call **`print`** on that expression. Try adding **`print`** statements to some of the above expressions to get them to display.

##  Data Types

In Python, all things have a type. In the above example, you saw saw __*integers*__ (positive and negative whole numbers) and __*strings*__ (sequences of characters, often thought of as words or sentences). 

We denote strings by surrounding the desired value with quotes: 
* For example, "AI" and "2026" are strings, while `bears` and `2020` (both without quotes) are not strings (`bears` without quotes would be interpreted as a variable). 

In addition to strings and integers, you'll also be using decimal numbers in Python, which are called __*floats*__ (positive and negative decimal numbers). 

You'll also often run into __*booleans*__. They can take on one of two values: `True` or `False`. Booleans are often used to check conditions; for example, we might have a list of dogs, and we want to sort them into small dogs and large dogs. One way we could accomplish this is to say either `True` or `False` for each dog after seeing if the dog weighs more than 15 pounds. 

We'll soon be going over additional data types. Below is a table that summarizes the information in this section:

|Variable Type|Definition|Examples|
|-|-|-|
|Integer|Positive and negative whole numbers|`42`, `-10`, `0`|
|Float|Positive and negative decimal numbers|`73.9`, `2.4`, `0.0`|
|String|Sequence of characters|`"just some text"`, `"variables"`|
|Boolean|True or false value|`True`, `False`|


## Variables
In the example below, __`a`__ and __`b`__ are Python objects known as __variables__. 

We are giving an object (in this case, an `integer` and a `float`, two Python data types) a name that we can store for later use. To use that value, we can simply type the name that we stored the value as. 

Variables are stored within the notebook's environment, meaning stored variable values carry over from cell to cell.


```python
# assign values to "a" and "b"
a = 4
b = 10/5

# Notice that "a" retains its value.
print(a)
```

    4



```python
# add variables "a" and "b"
a + b
```




    6.0



## Strings
In Python, a string is a sequence of characters. Strings can be created by enclosing characters inside a single quote or double-quotes. 


```python
#All 4 strings are the same 
String = "Hello World" 
String = 'Hello World' 
String = """Hello World""" 
String = "Hello" + " " + "World"
print(String)
```

    Hello World



```python
first_letter = String[0] # Indexing in python start at 0 
print(first_letter)

last_letter = String[-1] # Indexing into the last character 
print(last_letter)
```

    H
    d



```python
## empty string
string = "" 
string
```




    ''



Note: You **cannot** do the similar operations on strings as you can on integers. This is because they are a different **data type**


```python
four = 4
three = 3
four + three
```




    7




```python
four = "4"
three = "3"
four + three
```




    '43'



As you can see, "adding" strings concatenates them.


```python
four = 4
three = "3"
four + three
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[14], line 3
          1 four = 4
          2 three = "3"
    ----> 3 four + three


    TypeError: unsupported operand type(s) for +: 'int' and 'str'


Why didn't it work? Let's look at the **types** of data we are dealing with:


```python
type(four)
```




    int




```python
type(three)
```




    str



## Lists
The next topic is particularly useful in the kind of data manipulation that you will see throughout this class. 

The following few cells will introduce the concept of __`lists`__ (and their counterpart, __`numpy arrays`__). Read through the following cell to understand the basic structure of a list. 

* A list is an __ordered collection of objects__. They allow us to store and access groups of variables and other objects for easy access and analysis. Check out this [documentation](https://docs.python.org/3/tutorial/datastructures.html) for an in-depth look at the capabilities of lists.

To **initialize a list**, you use **square brackets**. Putting objects separated by commas in between the brackets will add them to the list. 


```python
# an empty list
lst = []
print(lst)
```

    []



```python
# reassigning our empty list to a new list
lst = [1, 3, 6, 'lists', 'are' 'fun', 4]
print(lst)
```

    [1, 3, 6, 'lists', 'arefun', 4]



```python
#lists in python are zero-indexed so the indices for lst are 0,1,2,3,4,5 and 6
example = lst[2]
print(example)
```

    6



```python
#list slicing: This line will store the first (inclusive) through fourth (exclusive) elements of lst as a new list 
#called lst_2:
lst_2 = lst[1:4]
lst_2
```




    [3, 6, 'lists']



Lists can also be operated on with a few built-in analysis functions. These include __`min`__ and __`max`__, among others. 

Lists can also be __concatenated__ together. 

Find some examples below.


```python
# A list containing six integers.
a_list = [1, 6, 4, 8, 13, 2]
```


```python
# Another list containing six integers.
b_list = [4, 5, 2, 14, 9, 11]
```


```python
print('Max of a_list:', max(a_list))
print('Min of b_list:', min(a_list))
```

    Max of a_list: 13
    Min of b_list: 1



```python
# Concatenate a_list and b_list:
c_list = a_list + b_list
print('Concatenated:', c_list)
```

    Concatenated: [1, 6, 4, 8, 13, 2, 4, 5, 2, 14, 9, 11]


## Looping
[__Loops__](https://docs.python.org/3/tutorial/controlflow.html) are often useful in manipulating, iterating over, or transforming large lists and arrays. 

The first type we will discuss is the __`for loop`__. For loops are helpful in **traversing a list** and **performing an action on each element**.

For example, the following code moves through every element in `example_array`, adds it to the previous element in `example_array`, and copies this sum to a new array. 

It's important to note that `"i"` or `"element"` is an **arbitrary** variable name used to represent whichever index value the loop is currently operating on. We can change the variable name to whatever we want and achieve the same result, as long as we stay consistent.


```python
# simplest for loop 

for i in [1, 2, 3]:  # how many times we perform an action (in this case 3) - note that X is arbitrary 
    print(i)         # action - print
```

    1
    2
    3



```python
for i in range(5): # we can use range to generate a range of numbers from 0 to 4
    print(i)
```

    0
    1
    2
    3
    4


For loops work over strings too


```python
string = "Goodbye World"
string
```




    'Goodbye World'




```python
for letter in string: 
    print(letter)    
```

    G
    o
    o
    d
    b
    y
    e
     
    W
    o
    r
    l
    d


Recall the c_list we created


```python
print(c_list)
```

    [1, 6, 4, 8, 13, 2, 4, 5, 2, 14, 9, 11]



```python
new_list = []             # initialize an empty list 

for element in c_list:    # for every *element* in a list - ie how many times we perform the action below
    new_element = element + 5    # action to perform - add 5
    new_list.append(new_element) # append this "new element" to "new list"

print(new_list)
```

    [6, 11, 9, 13, 18, 7, 9, 10, 7, 19, 14, 16]



```python
# we can also use range - which is less pythonic, but is sometimes necessary 
# iterate using list *indices* rather than elements themselves
for i in range(len(c_list)):
    c_list[i] = c_list[i] + 5

c_list
```




    [6, 11, 9, 13, 18, 7, 9, 10, 7, 19, 14, 16]



## Other types of loops - while loop
The __while loop__ repeatedly performs operations until a conditional is no longer satisfied. A conditional is a [boolean expression](https://en.wikipedia.org/wiki/Boolean_expression), that is an expression that evaluates to `True` or `False`. 

What makes a while loop different from a for loop is that a for loop ends after a fixed number of iterations, whereas a while loop ends when a True or False condition is met. **Many times, for loops are sufficient for most tasks, at least in my experience.** (probably 99% of the time)

In the below example, an array of integers 0 to 9 is generated. When the program enters the while loop on the subsequent line, it notices that the maximum value of the array is less than 50. Because of this, it adds 1 to the fifth element, as instructed. Once the instructions embedded in the loop are complete, the program refers back to the conditional. Again, the maximum value is less than 50. This process repeats until the the fifth element, now the maximum value of the array, is equal to 50, at which point the conditional is no longer true and the loop breaks.


```python
i = 0

while i < 5:          # set condition 
    print(i)          # do something 
    i += 1            # add 1 to every element -  # identical to i = i + 1 
```

    0
    1
    2
    3
    4



```python
# A simple list
numbers = [10, 20, 30]

i = 0 # Start index at 0

while i < len(numbers):      # Loop while index is within the list length
    print(numbers[i] + 1)     # Print the current element + 1 
    i += 1                   # Move to the next index
```

    11
    21
    31


Let us create a new list called "do_people_like_newman". Add to every string in the `seinfeld` list below the string "dislikes Newman". Store your new strings in the new "newman" list


```python
seinfeld = ['Jerry', 'Elaine', 'Kramer', 'Costanza']
seinfeld
```




    ['Jerry', 'Elaine', 'Kramer', 'Costanza']




```python
do_people_like_newman = []

for character in seinfeld: 
    character = character + " dislikes Newman"
    do_people_like_newman.append(character)
    

```


```python
do_people_like_newman
```




    ['Jerry dislikes Newman',
     'Elaine dislikes Newman',
     'Kramer dislikes Newman',
     'Costanza dislikes Newman']



## Dictionaries
The [__`dictionary`__](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) data structure is a collection of **"key-value"** pairs. Sometimes dictionaries are referred to as "maps" or "associative arrays". 


```
Dictionary = { key1 : value1 , 
               key2 : value2 , 
               key3 : value3 }
```

You might notice that unlike lists that use `[]`, dictionaries are initialized with a `{}` - also known as **squigly brackets** - and inside these brackets we have the  **key : value** pair. Each key-value pair is separated by a comma. Unlike lists, dictionaries are indexable by their "keys." 


For example if we wanted to create dictionary called "states" where:
* the __key__ is the __name of the state__ 
* the __value__ is the __state abbreviation__.


we would use the following code:


```python
states = {"California" : 'CA',
               "Idaho" : 'ID',
              "Nevada" : 'NV'}

print(states)
```

    {'California': 'CA', 'Idaho': 'ID', 'Nevada': 'NV'}


We can **access** the abbreviations (values) by indexing into the dictionary with brackets and the key value.

Note that in a list, we primarily index based on numbers. Here, we can use the key - which is a string. Thus, dictionaries can be more intuitive.

For example, if you want to return **`VALUE`** associated with **`KEY`**, you would do the following:

```
example_dict[KEY]
```

This would return **`VALUE`**.

How would you return the abbreviation for California using the states dictionary above? Assign `result` to this expression.


```python
# Using the states dictionary above, assign result to 'CA' by replacing the ellipses
result = states["California"]
result
```




    'CA'



Just like an actual dictionary, each "key" can store  multiple objects  - which is what makes dictionaries very useful. Again, unlike a list, we don't need to know the position of an element in a list - we can just "call" it via "keys". 

               

In the example below, the word park **park** is used as a key to store a **list** of its definitions.


```python
dictionary = {'parity': 'the quality or state of being equal or equivalent',
              'park' : ["a large public green area in a town, used for recreation" , 
                         "bring (a vehicle that one is driving) to a halt and leave it temporarily"]
              }
             
```

How would you return the second definition of the word "park" from the dictionary above?


```python
## Write your code below

```

## Functions
Functions are useful when you want to repeat a series of steps on multiple different objects, but don't want to type out the steps over and over again. Many functions are built into Python already; for example, you've already made use of `len()` to retrieve the number of elements in a list (or a lenght of a string). You can also write your own functions, and at this point you already have the skills to do so.


Functions generally take a set of __parameters__ (also called inputs), which define the objects they will use when they are run. For example, the `len()` function takes a list or array as its parameter, and returns the length of that list.


Let's make our first function **"take_five"** which will subtract 5 from a number. 

It will have only one input, which is the number we want to take 5 from.


```python
def take_five(number):
    output = number - 5
    return output
```


```python
take_five(100)
```




    95



Let's look at a more complicated function that takes two parameters, compares them somehow, and then returns a boolean value (`True` or `False`) depending on the comparison. The `is_multiple` function below takes as parameters an integer `m` and an integer `n`, checks if `m` is a multiple of `n`, and returns `True` if it is. Otherwise, it returns `False`. 

`if` statements, just like `while` loops, are dependent on boolean expressions. If the conditional is `True`, then the following indented code block will be executed. If the conditional evaluates to `False`, then the code block will be skipped over. Read more about `if` statements [here](https://docs.python.org/3/tutorial/controlflow.html).

Below, we will use the modulus operator `%`. 
* Typing `x % y` will return the remainder when `y` is divided by `x`. 
* Therefore, (`x % y == 0`) will return `True` when `y` divides `x` with remainder `0`


```python
print(10 % 3)   # 10 divided by 3 is 3 remainder 1 - 3 * 3 + 1 = 10

print(12 % 3)   # 12 divided by 3 is 4 remainder 0

```

    1
    0


We can set the condition to be set to "==0" which will give True if the number IS divisilble and false if not.


```python
10 % 3 == 0  # is 10 divisible by 3 - (with remainder 0) = FALSE

```




    False




```python
12 % 3 == 0  # is 12 divisible by 3 - (with remainder 0) = TRUE
```




    True



Final step:

`if / else` lets a program choose between two paths:

- if → do this when the condition is true

- else → do this when the condition is false


```python
def is_multiple(m, n):
    if (m % n == 0): 
        return True
    else:
        return False
```


```python
is_multiple(12, 4)
```




    True




```python
is_multiple(12, 7)
```




    False



## Review
To sum up:

* Variables - which are "names" used for storing objects in python
* Basic data structures - `strings`, `lists`, `integers`, `dictionaries`, etc
* Data can be manipulated at scale using loops
* There are also functions, which can be called. They usually require parameters as inputs. 


# **Introduction to Pandas**

For this course, we'll be working extensively with the [__`Pandas`__](https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html) package. This is an incredibly powerful tool that is used for examining, creating, and manipulating **Tabular Data** - think excel spreadsheets with "rows" and "columns."

Let's import pandas below (again, don't forget that documentations for packages can be important if you want to understand what's going on behind the scenes- simply Googling can be helpful):


```python
import pandas as pd
```

## Creating a simple Pandas Dataframe


Let's make our own DataFrame from scratch without having to import data from another file. 

Let's say we have two arrays (or lists), one with a list of fruits, and another with a list of their prices. Then, we can create a new `DataFrame` with each of these arrays as columns by setting the argument `data` in `pd.DataFrame()` to a **dictionary** of these columns:



```python
fruit_names = ['Apple', 'Orange', 'Banana']
fruit_prices = [1, 0.75, 0.5]

fruit_table = pd.DataFrame(data = {
     "Fruit"    : fruit_names,
     "Price ($)": fruit_prices})

fruit_table
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Banana</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, the pandas **`DataFrame`** method uses a **dictionary** of pairs of column labels and its corresponding arrays (lists), and creates a new DataFrame with each array as a column of the DataFrame. 

Let us select a column


```python
fruit_table["Fruit"] # series
```




    0     Apple
    1    Orange
    2    Banana
    Name: Fruit, dtype: object




```python
fruit_table[["Fruit"]] # pandas style column
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
      <th>Fruit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Banana</td>
    </tr>
  </tbody>
</table>
</div>



`.iloc` — index-based (positions in the data)


```python
# Select first row (position 0)
fruit_table.iloc[0]
```




    Fruit        Apple
    Price ($)      1.0
    Name: 0, dtype: object




```python
# Select first row, second column
fruit_table.iloc[0, 1]   
```




    1.0




```python
# Select first two rows
fruit_table.iloc[0:2]
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select rows 1-2 and columns 0–1
fruit_table.iloc[1:3, 0:2]
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Banana</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>



`.loc` — label-based (name of the row/column)

Uses labels (row names and column names)


```python
# Select row with label 0 # notice that the label (name) of the row is "0"
fruit_table.loc[0]
```




    Fruit        Apple
    Price ($)      1.0
    Name: 0, dtype: object




```python
fruit_table.loc[[1]]
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select rows 0 to 1 (inclusive!)
fruit_table.loc[0:1]
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
fruit_table.loc[:, "Fruit"]
```




    0     Apple
    1    Orange
    2    Banana
    Name: Fruit, dtype: object



`iloc` → "give me row at the index 0"

`loc`  → "give me the row named 0"


```python
df = fruit_table.set_index("Fruit") # set index to be fruits

df
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
      <th>Price ($)</th>
    </tr>
    <tr>
      <th>Fruit</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Apple</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Orange</th>
      <td>0.75</td>
    </tr>
    <tr>
      <th>Banana</th>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc["Apple"]     # works (label)
#df.iloc[1]          # works (position)

#df.loc[1]         # error (no label 1)
```




    Price ($)    1.0
    Name: Apple, dtype: float64





Finally, to create a new dataframe (with no columns or rows), we simply write:


```python
empty_table = pd.DataFrame()
empty_table
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



We typically start off with empty tables when we need to add rows inside for loops, which we'll see later.


```python

```

## Importing data from file using Pandas

We will use the "california_housing_train" data, which contains the median house prices for California Districts derived from the 1990 census.


To create this table, we will draw the data from the path `data/`, stored in a file called `california_housing_train.csv`. 

In general, to import data from a `.csv` file, we write **`pd.read_csv("file_path_&_name")`.** Information in `.csv`'s are separated by commas, and are what are typically used with the `pandas` package. 


```python
file_path = "data/california_housing_train.csv"
```


```python
california_housing = pd.read_csv(file_path)

```

Now that we have loaded in our DataFrames we want see our new table. Here are a few useful methods to see our table. 

* We can use the **`head()`** function to see the first 5 rows of our table. Alternatively we can use the **`tail()`** to see the last 5 rows of our table. 
* The **shape** method returns a tuple where the first item is the number of rows and the second is the number columns in the table. 

```
df.head()
df.tail()
df.shape
```

* Now that we know what are table looks like and the shape of it, we can use the **`describe()`** function to see basic summary statistics from our table. 

```
df.describe()
```


```python
# Show first 5 rows of our data
california_housing.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-114.31</td>
      <td>34.19</td>
      <td>15.0</td>
      <td>5612.0</td>
      <td>1283.0</td>
      <td>1015.0</td>
      <td>472.0</td>
      <td>1.4936</td>
      <td>66900.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-114.47</td>
      <td>34.40</td>
      <td>19.0</td>
      <td>7650.0</td>
      <td>1901.0</td>
      <td>1129.0</td>
      <td>463.0</td>
      <td>1.8200</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-114.56</td>
      <td>33.69</td>
      <td>17.0</td>
      <td>720.0</td>
      <td>174.0</td>
      <td>333.0</td>
      <td>117.0</td>
      <td>1.6509</td>
      <td>85700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-114.57</td>
      <td>33.64</td>
      <td>14.0</td>
      <td>1501.0</td>
      <td>337.0</td>
      <td>515.0</td>
      <td>226.0</td>
      <td>3.1917</td>
      <td>73400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-114.57</td>
      <td>33.57</td>
      <td>20.0</td>
      <td>1454.0</td>
      <td>326.0</td>
      <td>624.0</td>
      <td>262.0</td>
      <td>1.9250</td>
      <td>65500.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show last 5 rows of our data
california_housing.tail()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16995</th>
      <td>-124.26</td>
      <td>40.58</td>
      <td>52.0</td>
      <td>2217.0</td>
      <td>394.0</td>
      <td>907.0</td>
      <td>369.0</td>
      <td>2.3571</td>
      <td>111400.0</td>
    </tr>
    <tr>
      <th>16996</th>
      <td>-124.27</td>
      <td>40.69</td>
      <td>36.0</td>
      <td>2349.0</td>
      <td>528.0</td>
      <td>1194.0</td>
      <td>465.0</td>
      <td>2.5179</td>
      <td>79000.0</td>
    </tr>
    <tr>
      <th>16997</th>
      <td>-124.30</td>
      <td>41.84</td>
      <td>17.0</td>
      <td>2677.0</td>
      <td>531.0</td>
      <td>1244.0</td>
      <td>456.0</td>
      <td>3.0313</td>
      <td>103600.0</td>
    </tr>
    <tr>
      <th>16998</th>
      <td>-124.30</td>
      <td>41.80</td>
      <td>19.0</td>
      <td>2672.0</td>
      <td>552.0</td>
      <td>1298.0</td>
      <td>478.0</td>
      <td>1.9797</td>
      <td>85800.0</td>
    </tr>
    <tr>
      <th>16999</th>
      <td>-124.35</td>
      <td>40.54</td>
      <td>52.0</td>
      <td>1820.0</td>
      <td>300.0</td>
      <td>806.0</td>
      <td>270.0</td>
      <td>3.0147</td>
      <td>94600.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
california_housing.shape
```




    (17000, 9)




```python
rows = california_housing.shape[0] 
columns = california_housing.shape[1]
print("The number of rows is: " + str(rows), 
      "The number of columns is: " + str(columns))
```

    The number of rows is: 17000 The number of columns is: 9



```python
california_housing.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>17000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.562108</td>
      <td>35.625225</td>
      <td>28.589353</td>
      <td>2643.664412</td>
      <td>539.410824</td>
      <td>1429.573941</td>
      <td>501.221941</td>
      <td>3.883578</td>
      <td>207300.912353</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.005166</td>
      <td>2.137340</td>
      <td>12.586937</td>
      <td>2179.947071</td>
      <td>421.499452</td>
      <td>1147.852959</td>
      <td>384.520841</td>
      <td>1.908157</td>
      <td>115983.764387</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.790000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1462.000000</td>
      <td>297.000000</td>
      <td>790.000000</td>
      <td>282.000000</td>
      <td>2.566375</td>
      <td>119400.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.250000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>434.000000</td>
      <td>1167.000000</td>
      <td>409.000000</td>
      <td>3.544600</td>
      <td>180400.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.000000</td>
      <td>37.720000</td>
      <td>37.000000</td>
      <td>3151.250000</td>
      <td>648.250000</td>
      <td>1721.000000</td>
      <td>605.250000</td>
      <td>4.767000</td>
      <td>265000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>37937.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Renaming columns


```python
new_names = {
    "longitude": "LONG",
    "latitude": "LAT"}
```


```python
california = california_housing.rename(index = str, 
                                       columns = new_names)
california.head()
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-114.31</td>
      <td>34.19</td>
      <td>15.0</td>
      <td>5612.0</td>
      <td>1283.0</td>
      <td>1015.0</td>
      <td>472.0</td>
      <td>1.4936</td>
      <td>66900.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-114.47</td>
      <td>34.40</td>
      <td>19.0</td>
      <td>7650.0</td>
      <td>1901.0</td>
      <td>1129.0</td>
      <td>463.0</td>
      <td>1.8200</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-114.56</td>
      <td>33.69</td>
      <td>17.0</td>
      <td>720.0</td>
      <td>174.0</td>
      <td>333.0</td>
      <td>117.0</td>
      <td>1.6509</td>
      <td>85700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-114.57</td>
      <td>33.64</td>
      <td>14.0</td>
      <td>1501.0</td>
      <td>337.0</td>
      <td>515.0</td>
      <td>226.0</td>
      <td>3.1917</td>
      <td>73400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-114.57</td>
      <td>33.57</td>
      <td>20.0</td>
      <td>1454.0</td>
      <td>326.0</td>
      <td>624.0</td>
      <td>262.0</td>
      <td>1.9250</td>
      <td>65500.0</td>
    </tr>
  </tbody>
</table>
</div>



## Accessing Values

Often, it is useful to access only the rows, columns, or values related to our analysis. We'll look at several ways to cut down our table into smaller, more digestible parts.

Let's say we wanted to grab only the first _three_ rows of this DataFrame. We can do this by using the **`loc`** function; it takes in a list or range of numbers, and creates a new DataFrame with rows from the original DataFrame whose indices are given in the array or range. Remember that in Python, indices start at 0! Below are a few examples:


```python
california.iloc[[1, 3, 5]] # Takes rows with indices 1, 3, and 5 (the 2nd, 4th, and 6th rows)
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-114.47</td>
      <td>34.40</td>
      <td>19.0</td>
      <td>7650.0</td>
      <td>1901.0</td>
      <td>1129.0</td>
      <td>463.0</td>
      <td>1.8200</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-114.57</td>
      <td>33.64</td>
      <td>14.0</td>
      <td>1501.0</td>
      <td>337.0</td>
      <td>515.0</td>
      <td>226.0</td>
      <td>3.1917</td>
      <td>73400.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-114.58</td>
      <td>33.63</td>
      <td>29.0</td>
      <td>1387.0</td>
      <td>236.0</td>
      <td>671.0</td>
      <td>239.0</td>
      <td>3.3438</td>
      <td>74000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
california.iloc[[7]] # Takes the row with index 7 (8th row)
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>-114.59</td>
      <td>34.83</td>
      <td>41.0</td>
      <td>812.0</td>
      <td>168.0</td>
      <td>375.0</td>
      <td>158.0</td>
      <td>1.7083</td>
      <td>48500.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
california.iloc[range(6)] # Takes the row with indices 0, 1, ... 6
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-114.31</td>
      <td>34.19</td>
      <td>15.0</td>
      <td>5612.0</td>
      <td>1283.0</td>
      <td>1015.0</td>
      <td>472.0</td>
      <td>1.4936</td>
      <td>66900.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-114.47</td>
      <td>34.40</td>
      <td>19.0</td>
      <td>7650.0</td>
      <td>1901.0</td>
      <td>1129.0</td>
      <td>463.0</td>
      <td>1.8200</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-114.56</td>
      <td>33.69</td>
      <td>17.0</td>
      <td>720.0</td>
      <td>174.0</td>
      <td>333.0</td>
      <td>117.0</td>
      <td>1.6509</td>
      <td>85700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-114.57</td>
      <td>33.64</td>
      <td>14.0</td>
      <td>1501.0</td>
      <td>337.0</td>
      <td>515.0</td>
      <td>226.0</td>
      <td>3.1917</td>
      <td>73400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-114.57</td>
      <td>33.57</td>
      <td>20.0</td>
      <td>1454.0</td>
      <td>326.0</td>
      <td>624.0</td>
      <td>262.0</td>
      <td>1.9250</td>
      <td>65500.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-114.58</td>
      <td>33.63</td>
      <td>29.0</td>
      <td>1387.0</td>
      <td>236.0</td>
      <td>671.0</td>
      <td>239.0</td>
      <td>3.3438</td>
      <td>74000.0</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, we can also choose to display certain columns of the DataFrame. There are two methods to accomplish this, and both methods take in lists of either column indices or column labels:
- Insert the names of the columns as a list in the DataFrame
- The **`drop`** method creates a new DataFrame with all columns _except_ those indicated by the parameters (i.e. the parameters are dropped).

Some examples:


```python
california.loc[:, ["housing_median_age", "total_rooms"]].head() # Selects only "housing_median_age" and "total_rooms" columns
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
      <th>housing_median_age</th>
      <th>total_rooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.0</td>
      <td>5612.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.0</td>
      <td>7650.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.0</td>
      <td>720.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.0</td>
      <td>1501.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.0</td>
      <td>1454.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
california[['median_income','population']] # select only these two columns
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
      <th>median_income</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.4936</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.8200</td>
      <td>1129.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.6509</td>
      <td>333.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.1917</td>
      <td>515.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.9250</td>
      <td>624.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16995</th>
      <td>2.3571</td>
      <td>907.0</td>
    </tr>
    <tr>
      <th>16996</th>
      <td>2.5179</td>
      <td>1194.0</td>
    </tr>
    <tr>
      <th>16997</th>
      <td>3.0313</td>
      <td>1244.0</td>
    </tr>
    <tr>
      <th>16998</th>
      <td>1.9797</td>
      <td>1298.0</td>
    </tr>
    <tr>
      <th>16999</th>
      <td>3.0147</td>
      <td>806.0</td>
    </tr>
  </tbody>
</table>
<p>17000 rows × 2 columns</p>
</div>




```python
california.drop(california.columns[[0, 1]], axis=1).head() # Drops the columns with indices 0 and 1
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
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.0</td>
      <td>5612.0</td>
      <td>1283.0</td>
      <td>1015.0</td>
      <td>472.0</td>
      <td>1.4936</td>
      <td>66900.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.0</td>
      <td>7650.0</td>
      <td>1901.0</td>
      <td>1129.0</td>
      <td>463.0</td>
      <td>1.8200</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.0</td>
      <td>720.0</td>
      <td>174.0</td>
      <td>333.0</td>
      <td>117.0</td>
      <td>1.6509</td>
      <td>85700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.0</td>
      <td>1501.0</td>
      <td>337.0</td>
      <td>515.0</td>
      <td>226.0</td>
      <td>3.1917</td>
      <td>73400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.0</td>
      <td>1454.0</td>
      <td>326.0</td>
      <td>624.0</td>
      <td>262.0</td>
      <td>1.9250</td>
      <td>65500.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
california.iloc[[1,2,3,5], [3,5]] # Select only columns with indices 1 and 5, 
                                   # then only the rows with indices 1, 2, 3, 5
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
      <th>total_rooms</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>7650.0</td>
      <td>1129.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>720.0</td>
      <td>333.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1501.0</td>
      <td>515.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1387.0</td>
      <td>671.0</td>
    </tr>
  </tbody>
</table>
</div>



If you want to select more than one column at a time and/or a certain number of rows you can use 


```
df.loc(: , ['column_name' , 'column_name']) 
```

where the first argument is the index you want and the second argument is the list of columns you want. 

The " : " after **`.loc(`**  is shorthand for **all**. This example gives you all the rows for the two columns. 


To make sure wee understand the `loc`, `iloc`, and `drop` functions, let's try selecting the columns "households" to "median_house_value" with only the first 3 rows:


```python
california.iloc[1:4, 6:10] # first 3 rows = 1:4, columns 4 to 10

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
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>463.0</td>
      <td>1.8200</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>117.0</td>
      <td>1.6509</td>
      <td>85700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>226.0</td>
      <td>3.1917</td>
      <td>73400.0</td>
    </tr>
  </tbody>
</table>
</div>



Finally, the `loc` function in the DataFrame can be modified so instead of only choosing certain rows or columns you can give conditions for the selected columns or rows:
- A column label
- A condition that each row should match

In other words, we call the select rows as so: `DataFrame_name.loc[DataFrame_name["column_name'] filter]`.


Here are some examples of selection:

The variable `median_house_value` indicates median house value in an area. The below query will find all rows (areas) of the house value is exactly 90000


```python
california.loc[california["median_house_value"] == 90000]
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>165</th>
      <td>-116.23</td>
      <td>33.71</td>
      <td>17.0</td>
      <td>4874.0</td>
      <td>1349.0</td>
      <td>5032.0</td>
      <td>1243.0</td>
      <td>2.4440</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>294</th>
      <td>-116.79</td>
      <td>33.99</td>
      <td>16.0</td>
      <td>319.0</td>
      <td>68.0</td>
      <td>212.0</td>
      <td>67.0</td>
      <td>1.4688</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>1712</th>
      <td>-117.25</td>
      <td>34.13</td>
      <td>33.0</td>
      <td>2898.0</td>
      <td>503.0</td>
      <td>1374.0</td>
      <td>487.0</td>
      <td>3.6856</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>-117.29</td>
      <td>34.14</td>
      <td>39.0</td>
      <td>1989.0</td>
      <td>401.0</td>
      <td>805.0</td>
      <td>341.0</td>
      <td>2.4250</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>6218</th>
      <td>-118.24</td>
      <td>33.94</td>
      <td>42.0</td>
      <td>380.0</td>
      <td>106.0</td>
      <td>411.0</td>
      <td>100.0</td>
      <td>0.9705</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>6316</th>
      <td>-118.25</td>
      <td>33.93</td>
      <td>38.0</td>
      <td>180.0</td>
      <td>43.0</td>
      <td>246.0</td>
      <td>56.0</td>
      <td>2.8500</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>8733</th>
      <td>-118.59</td>
      <td>35.72</td>
      <td>28.0</td>
      <td>1491.0</td>
      <td>408.0</td>
      <td>98.0</td>
      <td>48.0</td>
      <td>1.4205</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>9548</th>
      <td>-119.32</td>
      <td>36.25</td>
      <td>21.0</td>
      <td>1231.0</td>
      <td>204.0</td>
      <td>609.0</td>
      <td>206.0</td>
      <td>2.8365</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>10217</th>
      <td>-119.89</td>
      <td>36.64</td>
      <td>34.0</td>
      <td>1422.0</td>
      <td>237.0</td>
      <td>716.0</td>
      <td>222.0</td>
      <td>2.9750</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>14625</th>
      <td>-122.17</td>
      <td>37.75</td>
      <td>48.0</td>
      <td>1751.0</td>
      <td>390.0</td>
      <td>935.0</td>
      <td>349.0</td>
      <td>1.4375</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>14904</th>
      <td>-122.23</td>
      <td>38.09</td>
      <td>26.0</td>
      <td>4397.0</td>
      <td>997.0</td>
      <td>2539.0</td>
      <td>965.0</td>
      <td>2.4875</td>
      <td>90000.0</td>
    </tr>
    <tr>
      <th>16954</th>
      <td>-124.15</td>
      <td>40.78</td>
      <td>36.0</td>
      <td>2112.0</td>
      <td>374.0</td>
      <td>829.0</td>
      <td>368.0</td>
      <td>3.3984</td>
      <td>90000.0</td>
    </tr>
  </tbody>
</table>
</div>



The variable `population` corresponds to population in an area. With the following where statement, we'll find the variables where the population is between 1 and 100 (ie sparsely populated areas).


```python
df_population = california.loc[california["population"].isin(range(1, 100))]
df_population
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>-114.65</td>
      <td>32.79</td>
      <td>21.0</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>64.0</td>
      <td>27.0</td>
      <td>0.8571</td>
      <td>25000.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-114.67</td>
      <td>33.92</td>
      <td>17.0</td>
      <td>97.0</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>15.0</td>
      <td>1.2656</td>
      <td>27500.0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>-115.80</td>
      <td>33.26</td>
      <td>2.0</td>
      <td>96.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>16.0</td>
      <td>5.3374</td>
      <td>47500.0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>-115.88</td>
      <td>32.93</td>
      <td>15.0</td>
      <td>208.0</td>
      <td>49.0</td>
      <td>51.0</td>
      <td>20.0</td>
      <td>4.0208</td>
      <td>32500.0</td>
    </tr>
    <tr>
      <th>120</th>
      <td>-115.94</td>
      <td>33.38</td>
      <td>5.0</td>
      <td>186.0</td>
      <td>43.0</td>
      <td>41.0</td>
      <td>21.0</td>
      <td>2.7000</td>
      <td>58800.0</td>
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
      <th>16643</th>
      <td>-122.74</td>
      <td>39.71</td>
      <td>16.0</td>
      <td>255.0</td>
      <td>73.0</td>
      <td>85.0</td>
      <td>38.0</td>
      <td>1.6607</td>
      <td>14999.0</td>
    </tr>
    <tr>
      <th>16733</th>
      <td>-122.89</td>
      <td>39.42</td>
      <td>16.0</td>
      <td>411.0</td>
      <td>114.0</td>
      <td>26.0</td>
      <td>19.0</td>
      <td>0.4999</td>
      <td>73500.0</td>
    </tr>
    <tr>
      <th>16743</th>
      <td>-122.91</td>
      <td>39.18</td>
      <td>43.0</td>
      <td>89.0</td>
      <td>18.0</td>
      <td>86.0</td>
      <td>27.0</td>
      <td>2.0208</td>
      <td>72500.0</td>
    </tr>
    <tr>
      <th>16801</th>
      <td>-123.17</td>
      <td>40.31</td>
      <td>36.0</td>
      <td>98.0</td>
      <td>28.0</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>0.5360</td>
      <td>14999.0</td>
    </tr>
    <tr>
      <th>16851</th>
      <td>-123.43</td>
      <td>40.22</td>
      <td>20.0</td>
      <td>133.0</td>
      <td>35.0</td>
      <td>87.0</td>
      <td>37.0</td>
      <td>3.6250</td>
      <td>67500.0</td>
    </tr>
  </tbody>
</table>
<p>178 rows × 9 columns</p>
</div>



## Sorting

It can be very useful to sort our DataFrames according to some column. The `sort` function does exactly that; it takes the column that you want to sort by. By default, the `sort_values` function sorts the table in _ascending_ order of the data in the column indicated; however, you can change this by setting the optional parameter `ascending=False`.

Recall that we created a `df_population` dataset above which contained areas with small population. Let's sort the values by the `median_house_value` column to see sparsely populated areas with expensive houses.   


```python
df_population.sort_values(by=['median_house_value'], 
                          ascending = False) # Sort table by value of property taken in ascending order
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6380</th>
      <td>-118.26</td>
      <td>34.05</td>
      <td>52.0</td>
      <td>58.0</td>
      <td>52.0</td>
      <td>41.0</td>
      <td>27.0</td>
      <td>4.0972</td>
      <td>500001.0</td>
    </tr>
    <tr>
      <th>8170</th>
      <td>-118.43</td>
      <td>33.97</td>
      <td>16.0</td>
      <td>70.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>7.7197</td>
      <td>500001.0</td>
    </tr>
    <tr>
      <th>10339</th>
      <td>-120.10</td>
      <td>38.91</td>
      <td>33.0</td>
      <td>1561.0</td>
      <td>282.0</td>
      <td>30.0</td>
      <td>11.0</td>
      <td>1.8750</td>
      <td>500001.0</td>
    </tr>
    <tr>
      <th>1559</th>
      <td>-117.22</td>
      <td>33.87</td>
      <td>16.0</td>
      <td>56.0</td>
      <td>7.0</td>
      <td>39.0</td>
      <td>14.0</td>
      <td>2.6250</td>
      <td>500001.0</td>
    </tr>
    <tr>
      <th>862</th>
      <td>-117.08</td>
      <td>34.08</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>11.0</td>
      <td>39.0</td>
      <td>14.0</td>
      <td>3.0625</td>
      <td>500001.0</td>
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
      <th>116</th>
      <td>-115.88</td>
      <td>32.93</td>
      <td>15.0</td>
      <td>208.0</td>
      <td>49.0</td>
      <td>51.0</td>
      <td>20.0</td>
      <td>4.0208</td>
      <td>32500.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-114.67</td>
      <td>33.92</td>
      <td>17.0</td>
      <td>97.0</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>15.0</td>
      <td>1.2656</td>
      <td>27500.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-114.65</td>
      <td>32.79</td>
      <td>21.0</td>
      <td>44.0</td>
      <td>33.0</td>
      <td>64.0</td>
      <td>27.0</td>
      <td>0.8571</td>
      <td>25000.0</td>
    </tr>
    <tr>
      <th>16643</th>
      <td>-122.74</td>
      <td>39.71</td>
      <td>16.0</td>
      <td>255.0</td>
      <td>73.0</td>
      <td>85.0</td>
      <td>38.0</td>
      <td>1.6607</td>
      <td>14999.0</td>
    </tr>
    <tr>
      <th>16801</th>
      <td>-123.17</td>
      <td>40.31</td>
      <td>36.0</td>
      <td>98.0</td>
      <td>28.0</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>0.5360</td>
      <td>14999.0</td>
    </tr>
  </tbody>
</table>
<p>178 rows × 9 columns</p>
</div>



Example: Only keep rows where population is above the average


```python
above_average = california[california["population"] > (california["population"]).mean()]
above_average.head()
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
      <th>LONG</th>
      <th>LAT</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>-114.58</td>
      <td>33.61</td>
      <td>25.0</td>
      <td>2907.0</td>
      <td>680.0</td>
      <td>1841.0</td>
      <td>633.0</td>
      <td>2.6768</td>
      <td>82400.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-114.59</td>
      <td>33.61</td>
      <td>34.0</td>
      <td>4789.0</td>
      <td>1175.0</td>
      <td>3134.0</td>
      <td>1056.0</td>
      <td>2.1782</td>
      <td>58400.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-114.60</td>
      <td>33.62</td>
      <td>16.0</td>
      <td>3741.0</td>
      <td>801.0</td>
      <td>2434.0</td>
      <td>824.0</td>
      <td>2.6797</td>
      <td>86500.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-115.22</td>
      <td>33.54</td>
      <td>18.0</td>
      <td>1706.0</td>
      <td>397.0</td>
      <td>3424.0</td>
      <td>283.0</td>
      <td>1.6250</td>
      <td>53500.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>-115.48</td>
      <td>32.68</td>
      <td>15.0</td>
      <td>3414.0</td>
      <td>666.0</td>
      <td>2097.0</td>
      <td>622.0</td>
      <td>2.3319</td>
      <td>91200.0</td>
    </tr>
  </tbody>
</table>
</div>



## Removing NAs and Duplicates from the DataFrame
Next we will cover dropping unwanted values and duplicate rows using **`dropna()`** and **`drop_duplicates()`** respectively. Both of these functions return a new DataFrame without changing the original by default. 

In order to store the new table you will have to **assign it to a variable**. This is generally the default behavior for most Pandas functions

Let's go back to our small fruit tables dataframe:


```python
fruit_table
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Banana</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
```


```python
fruit_table.loc[4] = np.nan ## insert NA
fruit_table.loc[5] = fruit_table.loc[0] ## insert a duplicate
```


```python
fruit_table
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Banana</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_fruit_df = fruit_table.drop_duplicates()
new_fruit_df
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Banana</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_fruit_df = fruit_table.dropna()
new_fruit_df
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Banana</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Apple</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



### Manipulating data in columns
One way to manipulate DataFrame tables is to use **`df['column name']`** to return one column of the table. 

In the example below we look at the **Fruit** column and use **`value_counts()`** to see how many times each unique value appears. Similarly, you can also apply other functions to columns. 

Adding new columns also uses this **`["column name"]`** syntax. You can specify **`df["column name"]`** and set it equal to the data you want to add. For example if you wanted to add a column of names with all upper case letters.
```
df["upper_case_names"] = df["names"].str.upper() 
```


```python
fruit_counts = new_fruit_df["Fruit"].value_counts()
print("Value counts:")
print(fruit_counts)
```

    Value counts:
    Fruit
    Apple     2
    Orange    1
    Banana    1
    Name: count, dtype: int64



```python
## this might give a warning - but is more intuitive
new_fruit_df['Fruit'] = new_fruit_df['Fruit'].str.upper()
```

    /tmp/ipykernel_105385/1683265854.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_fruit_df['Fruit'] = new_fruit_df['Fruit'].str.upper()



```python
new_fruit_df
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
      <th>Fruit</th>
      <th>Price ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>APPLE</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORANGE</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BANANA</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>APPLE</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```

## Histograms

Histograms are a nifty way to display quantitative information. 

* The x-axis (horizontal axis) is typically a quantitative variable of interest 
* The y-axis (vertical axis) is generally a frequency. 

Plot a histogram of the "median_house_value", and then **experiment** with the **bins** parameter. 

**What do you see? Explain how the "bins" parameter influences the "hist" function**. What happens as you increase the number of bins, say, to 100?



```python
california.hist("median_house_value", 
                grid = True,
                bins = 10) 
```


```python
california.hist("median_house_value", 
                grid = False, 
                bins = range(0,500001,10000))
```

To draw a line on the mean, we can import  [__matplotlib__](https://matplotlib.org/stable/gallery/index.html) - which is a useful package for plotting.

If you're feeling ambitious, try creating a graph that we haven't described here by looking at the [matplotlib](https://matplotlib.org/stable/gallery/index.html) documentation. Point is - there are billion ways of visualizing data - but all of them follow a similar pattern - namely, you need the data to represent (in this case, "median_house_value" column), and certain plot-specific parameters, like "bins".


```python
import matplotlib.pyplot as plt
```


```python
just_median_house_prices = california["median_house_value"]
just_median_house_prices.hist(bins = range(0,500001,10000))
plt.axvline(x = just_median_house_prices.median(), 
            color="red", 
            label = "median")
plt.legend()
```

Try plotting the mean instead of the median. What is the value of the mean? **Why is it different?**

## Scatter Plots
Scatter plots are generally used to relate two variables to one another. They can be useful when trying to infer relationships between variables, visualize simple regressions, and get a general sense of the "spread" of your data.

* Again, the x-axis (horizontal) will be our "median_income"
* The y-axis (vertical) will be the house price 


```python
# Median house value vs Median income. Do you spot a relationship?
california.plot.scatter(x = "median_income", 
                        y = "median_house_value")
```

In addition to "matplotlib", which we used above to draw the "median" line, we can also use [__"seaborn"__](https://seaborn.pydata.org/introduction.html). 

In my view, seaborn has generally **better looking plots**, and the "syntax" for creating plots is very simple. 



```python
import seaborn as sns
```


```python
sns.lmplot(x = "median_income", 
           y = "median_house_value", 
           data = california,
           fit_reg = True,
           line_kws={'color': 'red'},
           height = 10,
           aspect = 1) ## note - there is no width parameter - according to the documentation
                          ## Aspect = "aspect * height" gives the width of each facet in inches.
```

**Bonus**: If we are interested in making a "causal" statement, we would need to get the parameters of this line. 

To get the  "parameters" of the red line (linear regression), we can import scipy's stats functionality and use the following code.


```python
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(california['median_income'],
                                                               california['median_house_value'])
```


```python
slope
```




    42054.07487405612



## Importing *text* data from file using Pandas

We will now use the Trump Tweets data from Kaggle: https://www.kaggle.com/datasets/codebreaker619/donald-trump-tweets-dataset/code

We'll do the same thing as above, just with textual data. 





```python
tweets = pd.read_csv("data/tweets.csv")

```


```python
# Show first 5 rows of our data
tweets.head()
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
  </tbody>
</table>
</div>




```python
# Show last 5 rows of our data
tweets.tail()
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
</div>




```python
tweets.shape
```




    (56571, 9)




```python
rows = tweets.shape[0] 
columns = tweets.shape[1]
print("The number of rows is: " + str(rows), 
      "The number of columns is: " + str(columns))
```

    The number of rows is: 56571 The number of columns is: 9


### Renaming columns


```python
new_name = {
    "text": "text_of_tweet",
    "isRetweet" : "is_Retweet",
    "isDeleted" : "is_Deleted"}
```


```python
tweets = tweets.rename(index = str, columns = new_name)
tweets.head()
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
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
  </tbody>
</table>
</div>



## Accessing Values


```python
tweets.iloc[[1, 3, 5]] # Takes rows with indices 1, 3, and 5 (the 2nd, 4th, and 6th rows)
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
      <th>device</th>
      <th>favorites</th>
      <th>retweets</th>
      <th>date</th>
      <th>isFlagged</th>
    </tr>
  </thead>
  <tbody>
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
      <th>5</th>
      <td>1217962723234983937</td>
      <td>RT @WhiteHouse: President @realDonaldTrump ann...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>25048</td>
      <td>2020-01-17 00:11:56</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets.iloc[[7]] # Takes the row with index 7 (8th row)
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
      <th>device</th>
      <th>favorites</th>
      <th>retweets</th>
      <th>date</th>
      <th>isFlagged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>1319501865625784320</td>
      <td>https://t.co/4qwCKQOiOw</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>130822</td>
      <td>19127</td>
      <td>2020-10-23 04:52:14</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets.iloc[1:7] # Takes the row with indices 0, 1, ... 6
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
      <th>device</th>
      <th>favorites</th>
      <th>retweets</th>
      <th>date</th>
      <th>isFlagged</th>
    </tr>
  </thead>
  <tbody>
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
      <th>5</th>
      <td>1217962723234983937</td>
      <td>RT @WhiteHouse: President @realDonaldTrump ann...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>25048</td>
      <td>2020-01-17 00:11:56</td>
      <td>f</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1223640662689689602</td>
      <td>Getting a little exercise this morning! https:...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>285863</td>
      <td>30209</td>
      <td>2020-02-01 16:14:02</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets.loc[:, ["text_of_tweet", "device"]].head() # Selects only "housing_median_age" and "total_rooms" columns
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
      <th>text_of_tweet</th>
      <th>device</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Republicans and Democrats have both created ou...</td>
      <td>TweetDeck</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>Twitter for iPhone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RT @CBS_Herridge: READ: Letter to surveillance...</td>
      <td>Twitter for iPhone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Unsolicited Mail In Ballot Scam is a major...</td>
      <td>Twitter for iPhone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RT @MZHemingway: Very friendly telling of even...</td>
      <td>Twitter for iPhone</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets.drop(tweets.columns[[0, 1]], axis=1).head() # Drops the columns with indices 0 and 1
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
      <th>is_Retweet</th>
      <th>is_Deleted</th>
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
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>9081</td>
      <td>2020-01-17 13:13:59</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets.iloc[[1,5], [3,5,6]] # Select only rows with indices 1 and 5, 
                                   # then only the columns with indices 3,5
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
      <th>is_Deleted</th>
      <th>favorites</th>
      <th>retweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>73748</td>
      <td>17404</td>
    </tr>
    <tr>
      <th>5</th>
      <td>f</td>
      <td>0</td>
      <td>25048</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets.iloc[0:2]["text_of_tweet"]
```




    0    Republicans and Democrats have both created ou...
    1    I was thrilled to be back in the Great city of...
    Name: text_of_tweet, dtype: object




```python
for tweet in tweets["text_of_tweet"][:10]: # we can add enumerate to make better sense of the data
    print(tweet)
```

    Republicans and Democrats have both created our economic problems.
    I was thrilled to be back in the Great city of Charlotte, North Carolina with thousands of hardworking American Patriots who love our Country, cherish our values, respect our laws, and always put AMERICA FIRST! Thank you for a wonderful evening!! #KAG2020 https://t.co/dNJZfRsl9y
    RT @CBS_Herridge: READ: Letter to surveillance court obtained by CBS News questions where there will be further disciplinary action and cho…
    The Unsolicited Mail In Ballot Scam is a major threat to our Democracy, &amp; the Democrats know it. Almost all recent elections using this system, even though much smaller &amp;  with far fewer Ballots to count, have ended up being a disaster. Large numbers of missing Ballots &amp; Fraud!
    RT @MZHemingway: Very friendly telling of events here about Comey's apparent leaking to compliant media. If you read those articles and tho…
    RT @WhiteHouse: President @realDonaldTrump announced historic steps to protect the Constitutional right to pray in public schools! https://…
    Getting a little exercise this morning! https://t.co/fyAAcbhbgk
    https://t.co/4qwCKQOiOw
    https://t.co/VlEu8yyovv
    https://t.co/z5CRqHO8vg



```python
tweets.loc[tweets["retweets"] >= 90000]
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
      <th>device</th>
      <th>favorites</th>
      <th>retweets</th>
      <th>date</th>
      <th>isFlagged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>1325884977112883200</td>
      <td>The threshold identification of Ballots is tur...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>493076</td>
      <td>100609</td>
      <td>2020-11-09 19:36:26</td>
      <td>f</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1325896369534607360</td>
      <td>Georgia will be a big presidential win, as it ...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>637719</td>
      <td>94570</td>
      <td>2020-11-09 20:21:43</td>
      <td>f</td>
    </tr>
    <tr>
      <th>119</th>
      <td>1346488314157797389</td>
      <td>The Vice President has the power to reject fra...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>429984</td>
      <td>90069</td>
      <td>2021-01-05 16:06:45</td>
      <td>f</td>
    </tr>
    <tr>
      <th>159</th>
      <td>1326158564449275910</td>
      <td>WE ARE MAKING BIG PROGRESS. RESULTS START TO C...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>638933</td>
      <td>106449</td>
      <td>2020-11-10 13:43:35</td>
      <td>f</td>
    </tr>
    <tr>
      <th>162</th>
      <td>1331987171700510720</td>
      <td>Just saw the vote tabulations. There is NO WAY...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>496524</td>
      <td>98943</td>
      <td>2020-11-26 15:44:23</td>
      <td>t</td>
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
      <th>53534</th>
      <td>1152307567634391041</td>
      <td>Just spoke to @KanyeWest about his friend A$AP...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>679974</td>
      <td>179571</td>
      <td>2019-07-19 20:01:47</td>
      <td>f</td>
    </tr>
    <tr>
      <th>54842</th>
      <td>1127182661213138945</td>
      <td>RT @DonaldJTrumpJr: Very proud to have a Presi...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>100193</td>
      <td>2019-05-11 12:04:23</td>
      <td>f</td>
    </tr>
    <tr>
      <th>55325</th>
      <td>1118876219381026818</td>
      <td>https://t.co/222atp7wuB</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>353652</td>
      <td>97754</td>
      <td>2019-04-18 13:57:33</td>
      <td>f</td>
    </tr>
    <tr>
      <th>55404</th>
      <td>1116817144006750209</td>
      <td>WE WILL NEVER FORGET! https://t.co/VxrGFRFeJM</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter Media Studio</td>
      <td>238845</td>
      <td>93797</td>
      <td>2019-04-12 21:35:31</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56550</th>
      <td>1213316629666435072</td>
      <td>RT @realDonaldTrump: https://t.co/VXeKiVzpTf</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>153868</td>
      <td>2020-01-04 04:30:01</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>149 rows × 9 columns</p>
</div>



Question: Find all tweets that were deleted

## Sorting





```python
tweets.sort_values(by=['retweets'], 
                          ascending = False) # Sort table by retweets in ascending order
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
      <th>device</th>
      <th>favorites</th>
      <th>retweets</th>
      <th>date</th>
      <th>isFlagged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11759</th>
      <td>1311892190680014849</td>
      <td>Tonight, @FLOTUS and I tested positive for COV...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>1869706</td>
      <td>408866</td>
      <td>2020-10-02 04:54:06</td>
      <td>f</td>
    </tr>
    <tr>
      <th>35620</th>
      <td>881503147168071680</td>
      <td>#FraudNewsCNN #FNN https://t.co/WYUnHjjUjg</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>503063</td>
      <td>293109</td>
      <td>2017-07-02 13:21:42</td>
      <td>f</td>
    </tr>
    <tr>
      <th>39347</th>
      <td>795954831718498305</td>
      <td>TODAY WE MAKE AMERICA GREAT AGAIN!</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for Android</td>
      <td>498035</td>
      <td>281289</td>
      <td>2016-11-08 11:43:14</td>
      <td>f</td>
    </tr>
    <tr>
      <th>29598</th>
      <td>474134260149157888</td>
      <td>Are you allowed to impeach a president for gro...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for Android</td>
      <td>231077</td>
      <td>237674</td>
      <td>2014-06-04 10:23:11</td>
      <td>f</td>
    </tr>
    <tr>
      <th>9080</th>
      <td>1267637602724839424</td>
      <td>RT @SpaceX: Liftoff! https://t.co/DRBfdUM7JA</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>235250</td>
      <td>2020-06-02 02:02:10</td>
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
      <th>25181</th>
      <td>296278036582629376</td>
      <td>@SandyInu We'll see.</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter Web Client</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-29 15:26:03</td>
      <td>f</td>
    </tr>
    <tr>
      <th>25537</th>
      <td>289481566999158784</td>
      <td>@LaDaleBuggs  Thank you.</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter Web Client</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-10 21:19:18</td>
      <td>f</td>
    </tr>
    <tr>
      <th>25182</th>
      <td>296277147373412354</td>
      <td>@Cletendre21 @LetendreLarry But not hard enoug...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter Web Client</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-29 15:22:31</td>
      <td>f</td>
    </tr>
    <tr>
      <th>25536</th>
      <td>289481811623571456</td>
      <td>@MStuart1970 @foxandfriends  True!</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter Web Client</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-10 21:20:16</td>
      <td>f</td>
    </tr>
    <tr>
      <th>25316</th>
      <td>293815803613155328</td>
      <td>@adkradio Thanks and good luck.</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter Web Client</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-22 20:22:01</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>56571 rows × 9 columns</p>
</div>




```python
above_average = tweets[tweets["retweets"] > (tweets["retweets"].mean())]
above_average.head()
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
      <th>device</th>
      <th>favorites</th>
      <th>retweets</th>
      <th>date</th>
      <th>isFlagged</th>
    </tr>
  </thead>
  <tbody>
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
      <th>5</th>
      <td>1217962723234983937</td>
      <td>RT @WhiteHouse: President @realDonaldTrump ann...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>25048</td>
      <td>2020-01-17 00:11:56</td>
      <td>f</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1223640662689689602</td>
      <td>Getting a little exercise this morning! https:...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>285863</td>
      <td>30209</td>
      <td>2020-02-01 16:14:02</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(above_average)
```




    20397



## Removing NAs and Duplicates from the DataFrame



```python
tweets.isna().sum()
```




    id               0
    text_of_tweet    0
    is_Retweet       0
    is_Deleted       0
    device           0
    favorites        0
    retweets         0
    date             0
    isFlagged        0
    dtype: int64




```python
# Create a copy of the data (so we don't lose the original tweets data)
tweets_na = tweets.copy()   # create a copy
```


```python
import numpy as np
```


```python
tweets_na.iloc[4] = np.nan ## insert NA ## notice that if you use loc instead of iloc, it will create a new row 
tweets_na.iloc[2] = tweets.iloc[1] ## insert a duplicate
```


```python
tweets_na
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
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
      <td>9.845497e+16</td>
      <td>Republicans and Democrats have both created ou...</td>
      <td>f</td>
      <td>f</td>
      <td>TweetDeck</td>
      <td>49.0</td>
      <td>255.0</td>
      <td>2011-08-02 18:07:48</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.234653e+18</td>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>73748.0</td>
      <td>17404.0</td>
      <td>2020-03-03 01:34:50</td>
      <td>f</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.234653e+18</td>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>73748.0</td>
      <td>17404.0</td>
      <td>2020-03-03 01:34:50</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.304875e+18</td>
      <td>The Unsolicited Mail In Ballot Scam is a major...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>80527.0</td>
      <td>23502.0</td>
      <td>2020-09-12 20:10:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>1.319485e+18</td>
      <td>RT @RandPaul: I don’t know why @JoeBiden think...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>20683.0</td>
      <td>2020-10-23 03:46:25</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56567</th>
      <td>1.319484e+18</td>
      <td>RT @EliseStefanik: President @realDonaldTrump ...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>9869.0</td>
      <td>2020-10-23 03:42:05</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56568</th>
      <td>1.319444e+18</td>
      <td>RT @TeamTrump: LIVE: Presidential Debate #Deba...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>8197.0</td>
      <td>2020-10-23 01:03:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56569</th>
      <td>1.319384e+18</td>
      <td>Just signed an order to support the workers of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>176289.0</td>
      <td>36001.0</td>
      <td>2020-10-22 21:04:21</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56570</th>
      <td>1.319346e+18</td>
      <td>Suburban women want Safety &amp;amp; Security. Joe...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>95169.0</td>
      <td>19545.0</td>
      <td>2020-10-22 18:31:46</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>56571 rows × 9 columns</p>
</div>




```python
new_df = tweets_na.drop_duplicates()
new_df
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
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
      <td>9.845497e+16</td>
      <td>Republicans and Democrats have both created ou...</td>
      <td>f</td>
      <td>f</td>
      <td>TweetDeck</td>
      <td>49.0</td>
      <td>255.0</td>
      <td>2011-08-02 18:07:48</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.234653e+18</td>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>73748.0</td>
      <td>17404.0</td>
      <td>2020-03-03 01:34:50</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.304875e+18</td>
      <td>The Unsolicited Mail In Ballot Scam is a major...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>80527.0</td>
      <td>23502.0</td>
      <td>2020-09-12 20:10:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.217963e+18</td>
      <td>RT @WhiteHouse: President @realDonaldTrump ann...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>25048.0</td>
      <td>2020-01-17 00:11:56</td>
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
      <td>1.319485e+18</td>
      <td>RT @RandPaul: I don’t know why @JoeBiden think...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>20683.0</td>
      <td>2020-10-23 03:46:25</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56567</th>
      <td>1.319484e+18</td>
      <td>RT @EliseStefanik: President @realDonaldTrump ...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>9869.0</td>
      <td>2020-10-23 03:42:05</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56568</th>
      <td>1.319444e+18</td>
      <td>RT @TeamTrump: LIVE: Presidential Debate #Deba...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>8197.0</td>
      <td>2020-10-23 01:03:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56569</th>
      <td>1.319384e+18</td>
      <td>Just signed an order to support the workers of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>176289.0</td>
      <td>36001.0</td>
      <td>2020-10-22 21:04:21</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56570</th>
      <td>1.319346e+18</td>
      <td>Suburban women want Safety &amp;amp; Security. Joe...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>95169.0</td>
      <td>19545.0</td>
      <td>2020-10-22 18:31:46</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>56570 rows × 9 columns</p>
</div>




```python
new_df = tweets_na.dropna()
new_df
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
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
      <td>9.845497e+16</td>
      <td>Republicans and Democrats have both created ou...</td>
      <td>f</td>
      <td>f</td>
      <td>TweetDeck</td>
      <td>49.0</td>
      <td>255.0</td>
      <td>2011-08-02 18:07:48</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.234653e+18</td>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>73748.0</td>
      <td>17404.0</td>
      <td>2020-03-03 01:34:50</td>
      <td>f</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.234653e+18</td>
      <td>I was thrilled to be back in the Great city of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>73748.0</td>
      <td>17404.0</td>
      <td>2020-03-03 01:34:50</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.304875e+18</td>
      <td>The Unsolicited Mail In Ballot Scam is a major...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>80527.0</td>
      <td>23502.0</td>
      <td>2020-09-12 20:10:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.217963e+18</td>
      <td>RT @WhiteHouse: President @realDonaldTrump ann...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>25048.0</td>
      <td>2020-01-17 00:11:56</td>
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
      <td>1.319485e+18</td>
      <td>RT @RandPaul: I don’t know why @JoeBiden think...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>20683.0</td>
      <td>2020-10-23 03:46:25</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56567</th>
      <td>1.319484e+18</td>
      <td>RT @EliseStefanik: President @realDonaldTrump ...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>9869.0</td>
      <td>2020-10-23 03:42:05</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56568</th>
      <td>1.319444e+18</td>
      <td>RT @TeamTrump: LIVE: Presidential Debate #Deba...</td>
      <td>t</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>0.0</td>
      <td>8197.0</td>
      <td>2020-10-23 01:03:58</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56569</th>
      <td>1.319384e+18</td>
      <td>Just signed an order to support the workers of...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>176289.0</td>
      <td>36001.0</td>
      <td>2020-10-22 21:04:21</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56570</th>
      <td>1.319346e+18</td>
      <td>Suburban women want Safety &amp;amp; Security. Joe...</td>
      <td>f</td>
      <td>f</td>
      <td>Twitter for iPhone</td>
      <td>95169.0</td>
      <td>19545.0</td>
      <td>2020-10-22 18:31:46</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
<p>56570 rows × 9 columns</p>
</div>



### Manipulating data in columns



```python
## this might give a warning - but is more intuitive
tweets['is_Retweet'] = tweets['is_Retweet'].str.upper()
```


```python
tweets
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
      <th>text_of_tweet</th>
      <th>is_Retweet</th>
      <th>is_Deleted</th>
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
      <td>F</td>
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
      <td>F</td>
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
      <td>T</td>
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
      <td>F</td>
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
      <td>T</td>
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
      <td>T</td>
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
      <td>T</td>
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
      <td>T</td>
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
      <td>F</td>
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
      <td>F</td>
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



Try uppercasing and lowercasing the text_of_tweet column


```python

```

# Simple Visualizations

## Histograms


```python
# Create a new column with the length of each tweet
tweets["tweet_length"] = tweets["text_of_tweet"].str.len()
```


```python
tweets.hist("tweet_length", 
                grid = True,
                bins = 100)
```

We can use matplotlib


```python
import matplotlib.pyplot as plt

# Plot histogram
tweets["tweet_length"].hist(bins=100, figsize=(10,5))

plt.title("Histogram of Tweet Lengths")
plt.xlabel("Number of characters")
plt.ylabel("Number of tweets")
plt.show()
```


```python
just_median_retweets = tweets["tweet_length"]

tweets["tweet_length"].hist(bins=100, 
                            figsize=(10,5))

plt.axvline(x = just_median_retweets.median(), 
            color="red", 
            label = "median")
plt.legend()
```


```python
just_median_retweets.median()
```




    132.0




Can we plot mean vs median?


```python

```

## Scatter Plots



```python
# Median house value vs Median income. Do you spot a relationship?
tweets.plot.scatter(x = "tweet_length", 
                        y = "retweets")
```


```python
import seaborn as sns
```


```python
sns.lmplot(x = "tweet_length", 
           y = "retweets", 
           data = tweets,
           fit_reg = True,
           line_kws={'color': 'red'},
           height = 5,
           aspect = 1) ## note - there is no width parameter - according to the documentation
                          ## Aspect = "aspect * height" gives the width of each facet in inches.
```

Let us make a causal interpretation of this.


```python
from scipy.stats import linregress

# Independent variable (x) and dependent (y)
x = tweets["tweet_length"]
y = tweets["retweets"]

# Fit linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

print(f"Linear regression equation: retweets = {slope:.2f} * tweet_length + {intercept:.2f}")
print(f"P-value: {p_value:.10f}")
```

    Linear regression equation: retweets = 47.98 * tweet_length + 2487.82
    P-value: 0.0000000000



```python
slope

```




    47.98059940583321




```python
intercept
```




    2487.8173046572592



What do these numbers mean?

As you can see, we used pandas to draw plots, access and subset the data, etc. Thinking about data in terms of rows and columns is very natural so pandas is indeed **a very convenient tool for understanding data.** Thus, getting a hand of pandas is important.

As a summary, here are some useful functions that you can refer to when using pandas.
    
|Name|Example|Purpose|
|-|-|-|
|`DataFrame`|`DataFrame()`|Create an empty DataFrame, usually to extend with data|
|`pd.read_csv`|`pandas.read_table("my_data.csv")`|Create a DataFrame from a data file|
|`pd.DataFrame({})`|`df = pandas.DataFrame({"N": np.arange(5), "2*N": np.arange(0, 10, 2)})`|Create a copy of a DataFrame with specified columns|
|`loc`|`df.loc[df["N"] > 10]`|Create a copy of a DataFrame with only the rows that match some *predicate*|
|`loc`|`df.loc["N"]`|Create a copy of a DataFrame with only specified column names|
|`(subsetting)`|`df[["N"]]`|Another way to create a copy of a DataFrame with only specified column names|
|`iloc`|`df.iloc(np.arange(0, 6, 2))`|Create a copy of the DataFrame with only the rows whose indices are in the given array|
|`sort`|`df.sort(["N"])`|Create a copy of a DataFrame sorted by the values in a column|
|`index`|`len(df.index)`|Compute the number of rows in a DataFrame|
|`columns`|`len(tbl.columns)`|Compute the number of columns in a DataFrame|
|`drop`|`df.drop(columns=["2*N"])`|Create a copy of a DataFrame without some of the columns|


# Opening multiple .txt files and creating a pandas dataframe 

In your projects, in addition to curated datasets with rows and columns (like the tweet data set above), you might also have a number of pdf and txt files that you might want to explore. Although this might be challenging, it's actually quite easy to deal with if you understand that you just need to make your files into a txt file and store them in a directory.

Generally: One should always aim to convert an unknown problem (how to work with txt/pdf files) into things that you know or roughly understand (make a pandas dataframe out of txt/pdf files). This is also important when you "vibe code" - ie ask, AI to help you. 

Thus, we must find a way to put all the text files into a pandas dataframe. One way of doing this is using python libraries, a particularly useful one that I have been using all the time is called `glob`


```python
import glob

# Path to the folder containing the txt files
folder_path = "data/plato_works"

# Use glob to find all .txt files
txt_files = glob.glob(f"{folder_path}/*.txt")
```


```python
txt_files
```




    ['data/plato_works/apology.txt',
     'data/plato_works/charmides.txt',
     'data/plato_works/cratylus.txt',
     'data/plato_works/critias.txt',
     'data/plato_works/crito.txt',
     'data/plato_works/euthydemus.txt',
     'data/plato_works/euthyfro.txt',
     'data/plato_works/gorgias.txt',
     'data/plato_works/ion.txt',
     'data/plato_works/laches.txt',
     'data/plato_works/laws.txt',
     'data/plato_works/lysis.txt',
     'data/plato_works/meno.txt',
     'data/plato_works/parmenides.txt',
     'data/plato_works/phaedo.txt',
     'data/plato_works/phaedrus.txt',
     'data/plato_works/philebus.txt',
     'data/plato_works/protagoras.txt',
     'data/plato_works/republic.txt',
     'data/plato_works/seventh_letter.txt',
     'data/plato_works/sophist.txt',
     'data/plato_works/stateman.txt',
     'data/plato_works/symposium.txt',
     'data/plato_works/theatetus.txt',
     'data/plato_works/timaeus.txt']




```python
# List to store data
data = []

# Loop through each file
for file in txt_files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    filename = file.split("/")[-1]  # get just the filename - not the other C\\users etc 
    data.append({"filename": filename, 
                 "text": text})

# Create pandas DataFrame
plato = pd.DataFrame(data)
```


```python
plato.head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apology.txt</td>
      <td>Apology\n\nApology\nBy Plato\nCommentary:\nQui...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>charmides.txt</td>
      <td>Charmides, or Temperance\n\nCharmides, or Temp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cratylus.txt</td>
      <td>Cratylus\n\nCratylus\nBy Plato\nCommentary:\nA...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>critias.txt</td>
      <td>Critias\n\nCritias\nBy Plato\nCommentary:\nMan...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>crito.txt</td>
      <td>Crito\n\nCrito\nBy Plato\nCommentary:\nMany co...</td>
    </tr>
  </tbody>
</table>
</div>




```python
apology_text = plato[plato['filename'] == 'apology.txt']['text'].values[0]
```


```python
print(apology_text)
```

    Apology
    
    Apology
    By Plato
    Commentary:
    Quite a few comments have been posted about
    Apology
    .
    Download:
    A 58k
    text-only version is
    .
    Apology
    By Plato
    Translated by Benjamin Jowett
    Socrates' Defense
    How you have felt, O men of Athens, at hearing the speeches of
    my accusers, I cannot tell; but I know that their persuasive words almost
    made me forget who I was - such was the effect of them; and yet they have
    hardly spoken a word of truth. But many as their falsehoods were, there
    was one of them which quite amazed me; - I mean when they told you to be
    upon your guard, and not to let yourselves be deceived by the force of
    my eloquence. They ought to have been ashamed of saying this, because they
    were sure to be detected as soon as I opened my lips and displayed my deficiency;
    they certainly did appear to be most shameless in saying this, unless by
    the force of eloquence they mean the force of truth; for then I do indeed
    admit that I am eloquent. But in how different a way from theirs!  Well,
    as I was saying, they have hardly uttered a word, or not more than a word,
    of truth; but you shall hear from me the whole truth: not, however, delivered
    after their manner, in a set oration duly ornamented with words and phrases.
    No indeed!  but I shall use the words and arguments which occur to me at
    the moment; for I am certain that this is right, and that at my time of
    life I ought not to be appearing before you, O men of Athens, in the character
    of a juvenile orator - let no one expect this of me. And I must beg of
    you to grant me one favor, which is this - If you hear me using the same
    words in my defence which I have been in the habit of using, and which
    most of you may have heard in the agora, and at the tables of the money-changers,
    or anywhere else, I would ask you not to be surprised at this, and not
    to interrupt me.  For I am more than seventy years of age, and this is
    the first time that I have ever appeared in a court of law, and I am quite
    a stranger to the ways of the place; and therefore I would have you regard
    me as if I were really a stranger, whom you would excuse if he spoke in
    his native tongue, and after the fashion of his country; - that I think
    is not an unfair request. Never mind the manner, which may or may not be
    good; but think only of the justice of my cause, and give heed to that:
    let the judge decide justly and the speaker speak truly.
    And first, I have to reply to the older charges and to my first
    accusers, and then I will go to the later ones. For I have had many accusers,
    who accused me of old, and their false charges have continued during many
    years; and I am more afraid of them than of Anytus and his associates,
    who are dangerous, too, in their own way. But far more dangerous are these,
    who began when you were children, and took possession of your minds with
    their falsehoods, telling of one Socrates, a wise man, who speculated about
    the heaven above, and searched into the earth beneath, and made the worse
    appear the better cause. These are the accusers whom I dread; for they
    are the circulators of this rumor, and their hearers are too apt to fancy
    that speculators of this sort do not believe in the gods. And they are
    many, and their charges against me are of ancient date, and they made them
    in days when you were impressible - in childhood, or perhaps in youth -
    and the cause when heard went by default, for there was none to answer.
    And, hardest of all, their names I do not know and cannot tell; unless
    in the chance of a comic poet. But the main body of these slanderers who
    from envy and malice have wrought upon you - and there are some of them
    who are convinced themselves, and impart their convictions to others -
    all these, I say, are most difficult to deal with; for I cannot have them
    up here, and examine them, and therefore I must simply fight with shadows
    in my own defence, and examine when there is no one who answers. I will
    ask you then to assume with me, as I was saying, that my opponents are
    of two kinds - one recent, the other ancient; and I hope that you will
    see the propriety of my answering the latter first, for these accusations
    you heard long before the others, and much oftener.
    Well, then, I will make my defence, and I will endeavor in the
    short time which is allowed to do away with this evil opinion of me which
    you have held for such a long time; and I hope I may succeed, if this be
    well for you and me, and that my words may find favor with you. But I know
    that to accomplish this is not easy - I quite see the nature of the task.
    Let the event be as God wills: in obedience to the law I make my
    defence.
    I will begin at the beginning, and ask what the accusation is which
    has given rise to this slander of me, and which has encouraged Meletus
    to proceed against me. What do the slanderers say? They shall be my prosecutors,
    and I will sum up their words in an affidavit. "Socrates is an evil-doer,
    and a curious person, who searches into things under the earth and in heaven,
    and he makes the worse appear the better cause; and he teaches the aforesaid
    doctrines to others." That is the nature of the accusation, and that is
    what you have seen yourselves in the comedy of Aristophanes; who has introduced
    a man whom he calls Socrates, going about and saying that he can walk in
    the air, and talking a deal of nonsense concerning matters of which I do
    not pretend to know either much or little - not that I mean to say anything
    disparaging of anyone who is a student of natural philosophy. I should
    be very sorry if Meletus could lay that to my charge. But the simple truth
    is, O Athenians, that I have nothing to do with these studies. Very many
    of those here present are witnesses to the truth of this, and to them I
    appeal. Speak then, you who have heard me, and tell your neighbors whether
    any of you have ever known me hold forth in few words or in many upon matters
    of this sort. ... You hear their answer. And from what they say of this
    you will be able to judge of the truth of the rest.
    As little foundation is there for the report that I am a teacher,
    and take money; that is no more true than the other. Although, if a man
    is able to teach, I honor him for being paid. There is Gorgias of Leontium,
    and Prodicus of Ceos, and Hippias of Elis, who go the round of the cities,
    and are able to persuade the young men to leave their own citizens, by
    whom they might be taught for nothing, and come to them, whom they not
    only pay, but are thankful if they may be allowed to pay them. There is
    actually a Parian philosopher residing in Athens, of whom I have heard;
    and I came to hear of him in this way: - I met a man who has spent a world
    of money on the Sophists, Callias the son of Hipponicus, and knowing that
    he had sons, I asked him: "Callias," I said, "if your two sons were foals
    or calves, there would be no difficulty in finding someone to put over
    them; we should hire a trainer of horses or a farmer probably who would
    improve and perfect them in their own proper virtue and excellence; but
    as they are human beings, whom are you thinking of placing over them? Is
    there anyone who understands human and political virtue? You must have
    thought about this as you have sons; is there anyone?" "There is," he said.
    "Who is he?" said I, "and of what country? and what does he charge?" "Evenus
    the Parian," he replied; "he is the man, and his charge is five minae."
    Happy is Evenus, I said to myself, if he really has this wisdom, and teaches
    at such a modest charge. Had I the same, I should have been very proud
    and conceited; but the truth is that I have no knowledge of the
    kind.
    I dare say, Athenians, that someone among you will reply, "Why
    is this, Socrates, and what is the origin of these accusations of you:
    for there must have been something strange which you have been doing? All
    this great fame and talk about you would never have arisen if you had been
    like other men: tell us, then, why this is, as we should be sorry to judge
    hastily of you." Now I regard this as a fair challenge, and I will endeavor
    to explain to you the origin of this name of "wise," and of this evil fame.
    Please to attend then. And although some of you may think I am joking,
    I declare that I will tell you the entire truth. Men of Athens, this reputation
    of mine has come of a certain sort of wisdom which I possess. If you ask
    me what kind of wisdom, I reply, such wisdom as is attainable by man, for
    to that extent I am inclined to believe that I am wise; whereas the persons
    of whom I was speaking have a superhuman wisdom, which I may fail to describe,
    because I have it not myself; and he who says that I have, speaks falsely,
    and is taking away my character. And here, O men of Athens, I must beg
    you not to interrupt me, even if I seem to say something extravagant. For
    the word which I will speak is not mine. I will refer you to a witness
    who is worthy of credit, and will tell you about my wisdom - whether I
    have any, and of what sort - and that witness shall be the god of Delphi.
    You must have known Chaerephon; he was early a friend of mine, and also
    a friend of yours, for he shared in the exile of the people, and returned
    with you. Well, Chaerephon, as you know, was very impetuous in all his
    doings, and he went to Delphi and boldly asked the oracle to tell him whether
    - as I was saying, I must beg you not to interrupt - he asked the oracle
    to tell him whether there was anyone wiser than I was, and the Pythian
    prophetess answered that there was no man wiser. Chaerephon is dead himself,
    but his brother, who is in court, will confirm the truth of this
    story.
    Why do I mention this? Because I am going to explain to you why
    I have such an evil name. When I heard the answer, I said to myself, What
    can the god mean? and what is the interpretation of this riddle? for I
    know that I have no wisdom, small or great. What can he mean when he says
    that I am the wisest of men? And yet he is a god and cannot lie; that would
    be against his nature. After a long consideration, I at last thought of
    a method of trying the question. I reflected that if I could only find
    a man wiser than myself, then I might go to the god with a refutation in
    my hand. I should say to him, "Here is a man who is wiser than I am; but
    you said that I was the wisest." Accordingly I went to one who had the
    reputation of wisdom, and observed to him - his name I need not mention;
    he was a politician whom I selected for examination - and the result was
    as follows: When I began to talk with him, I could not help thinking that
    he was not really wise, although he was thought wise by many, and wiser
    still by himself; and I went and tried to explain to him that he thought
    himself wise, but was not really wise; and the consequence was that he
    hated me, and his enmity was shared by several who were present and heard
    me. So I left him, saying to myself, as I went away: Well, although I do
    not suppose that either of us knows anything really beautiful and good,
    I am better off than he is - for he knows nothing, and thinks that he knows.
    I neither know nor think that I know. In this latter particular, then,
    I seem to have slightly the advantage of him. Then I went to another, who
    had still higher philosophical pretensions, and my conclusion was exactly
    the same. I made another enemy of him, and of many others besides
    him.
    After this I went to one man after another, being not unconscious
    of the enmity which I provoked, and I lamented and feared this: but necessity
    was laid upon me - the word of God, I thought, ought to be considered first.
    And I said to myself, Go I must to all who appear to know, and find out
    the meaning of the oracle. And I swear to you, Athenians, by the dog I
    swear! - for I must tell you the truth - the result of my mission was just
    this: I found that the men most in repute were all but the most foolish;
    and that some inferior men were really wiser and better. I will tell you
    the tale of my wanderings and of the "Herculean" labors, as I may call
    them, which I endured only to find at last the oracle irrefutable. When
    I left the politicians, I went to the poets; tragic, dithyrambic, and all
    sorts. And there, I said to myself, you will be detected; now you will
    find out that you are more ignorant than they are. Accordingly, I took
    them some of the most elaborate passages in their own writings, and asked
    what was the meaning of them - thinking that they would teach me something.
    Will you believe me? I am almost ashamed to speak of this, but still I
    must say that there is hardly a person present who would not have talked
    better about their poetry than they did themselves. That showed me in an
    instant that not by wisdom do poets write poetry, but by a sort of genius
    and inspiration; they are like diviners or soothsayers who also say many
    fine things, but do not understand the meaning of them. And the poets appeared
    to me to be much in the same case; and I further observed that upon the
    strength of their poetry they believed themselves to be the wisest of men
    in other things in which they were not wise. So I departed, conceiving
    myself to be superior to them for the same reason that I was superior to
    the politicians.
    At last I went to the artisans, for I was conscious that I knew
    nothing at all, as I may say, and I was sure that they knew many fine things;
    and in this I was not mistaken, for they did know many things of which
    I was ignorant, and in this they certainly were wiser than I was. But I
    observed that even the good artisans fell into the same error as the poets;
    because they were good workmen they thought that they also knew all sorts
    of high matters, and this defect in them overshadowed their wisdom - therefore
    I asked myself on behalf of the oracle, whether I would like to be as I
    was, neither having their knowledge nor their ignorance, or like them in
    both; and I made answer to myself and the oracle that I was better off
    as I was.
    This investigation has led to my having many enemies of the worst
    and most dangerous kind, and has given occasion also to many calumnies,
    and I am called wise, for my hearers always imagine that I myself possess
    the wisdom which I find wanting in others: but the truth is, O men of Athens,
    that God only is wise; and in this oracle he means to say that the wisdom
    of men is little or nothing; he is not speaking of Socrates, he is only
    using my name as an illustration, as if he said, He, O men, is the wisest,
    who, like Socrates, knows that his wisdom is in truth worth nothing. And
    so I go my way, obedient to the god, and make inquisition into the wisdom
    of anyone, whether citizen or stranger, who appears to be wise; and if
    he is not wise, then in vindication of the oracle I show him that he is
    not wise; and this occupation quite absorbs me, and I have no time to give
    either to any public matter of interest or to any concern of my own, but
    I am in utter poverty by reason of my devotion to the
    god.
    There is another thing: - young men of the richer classes, who
    have not much to do, come about me of their own accord; they like to hear
    the pretenders examined, and they often imitate me, and examine others
    themselves; there are plenty of persons, as they soon enough discover,
    who think that they know something, but really know little or nothing:
    and then those who are examined by them instead of being angry with themselves
    are angry with me: This confounded Socrates, they say; this villainous
    misleader of youth! - and then if somebody asks them, Why, what evil does
    he practise or teach? they do not know, and cannot tell; but in order that
    they may not appear to be at a loss, they repeat the ready-made charges
    which are used against all philosophers about teaching things up in the
    clouds and under the earth, and having no gods, and making the worse appear
    the better cause; for they do not like to confess that their pretence of
    knowledge has been detected - which is the truth: and as they are numerous
    and ambitious and energetic, and are all in battle array and have persuasive
    tongues, they have filled your ears with their loud and inveterate calumnies.
    And this is the reason why my three accusers, Meletus and Anytus and Lycon,
    have set upon me; Meletus, who has a quarrel with me on behalf of the poets;
    Anytus, on behalf of the craftsmen; Lycon, on behalf of the rhetoricians:
    and as I said at the beginning, I cannot expect to get rid of this mass
    of calumny all in a moment. And this, O men of Athens, is the truth and
    the whole truth; I have concealed nothing, I have dissembled nothing. And
    yet I know that this plainness of speech makes them hate me, and what is
    their hatred but a proof that I am speaking the truth? - this is the occasion
    and reason of their slander of me, as you will find out either in this
    or in any future inquiry.
    I have said enough in my defence against the first class of my
    accusers; I turn to the second class, who are headed by Meletus, that good
    and patriotic man, as he calls himself. And now I will try to defend myself
    against them: these new accusers must also have their affidavit read. What
    do they say? Something of this sort: - That Socrates is a doer of evil,
    and corrupter of the youth, and he does not believe in the gods of the
    state, and has other new divinities of his own. That is the sort of charge;
    and now let us examine the particular counts. He says that I am a doer
    of evil, who corrupt the youth; but I say, O men of Athens, that Meletus
    is a doer of evil, and the evil is that he makes a joke of a serious matter,
    and is too ready at bringing other men to trial from a pretended zeal and
    interest about matters in which he really never had the smallest interest.
    And the truth of this I will endeavor to prove.
    Come hither, Meletus, and let me ask a question of you. You think
    a great deal about the improvement of youth?
    Yes, I do.
    Tell the judges, then, who is their improver; for you must know,
    as you have taken the pains to discover their corrupter, and are citing
    and accusing me before them. Speak, then, and tell the judges who their
    improver is. Observe, Meletus, that you are silent, and have nothing to
    say. But is not this rather disgraceful, and a very considerable proof
    of what I was saying, that you have no interest in the matter? Speak up,
    friend, and tell us who their improver is.
    The laws.
    But that, my good sir, is not my meaning. I want to know who the
    person is, who, in the first place, knows the laws.
    The judges, Socrates, who are present in court.
    What do you mean to say, Meletus, that they are able to instruct
    and improve youth?
    Certainly they are.
    What, all of them, or some only and not others?
    All of them.
    By the goddess Here, that is good news! There are plenty of improvers,
    then. And what do you say of the audience, - do they improve
    them?
    Yes, they do.
    And the senators?
    Yes, the senators improve them.
    But perhaps the members of the citizen assembly corrupt them? -
    or do they too improve them?
    They improve them.
    Then every Athenian improves and elevates them; all with the exception
    of myself; and I alone am their corrupter? Is that what you
    affirm?
    That is what I stoutly affirm.
    I am very unfortunate if that is true. But suppose I ask you a
    question: Would you say that this also holds true in the case of horses?
    Does one man do them harm and all the world good? Is not the exact opposite
    of this true? One man is able to do them good, or at least not many; -
    the trainer of horses, that is to say, does them good, and others who have
    to do with them rather injure them? Is not that true, Meletus, of horses,
    or any other animals? Yes, certainly. Whether you and Anytus say yes or
    no, that is no matter. Happy indeed would be the condition of youth if
    they had one corrupter only, and all the rest of the world were their improvers.
    And you, Meletus, have sufficiently shown that you never had a thought
    about the young: your carelessness is seen in your not caring about matters
    spoken of in this very indictment.
    And now, Meletus, I must ask you another question: Which is better,
    to live among bad citizens, or among good ones? Answer, friend, I say;
    for that is a question which may be easily answered. Do not the good do
    their neighbors good, and the bad do them evil?
    Certainly.
    And is there anyone who would rather be injured than benefited
    by those who live with him? Answer, my good friend; the law requires you
    to answer - does anyone like to be injured?
    Certainly not.
    And when you accuse me of corrupting and deteriorating the youth,
    do you allege that I corrupt them intentionally or unintentionally?
    Intentionally, I say.
    But you have just admitted that the good do their neighbors good,
    and the evil do them evil. Now is that a truth which your superior wisdom
    has recognized thus early in life, and am I, at my age, in such darkness
    and ignorance as not to know that if a man with whom I have to live is
    corrupted by me, I am very likely to be harmed by him, and yet I corrupt
    him, and intentionally, too; - that is what you are saying, and of that
    you will never persuade me or any other human being. But either I do not
    corrupt them, or I corrupt them unintentionally, so that on either view
    of the case you lie. If my offence is unintentional, the law has no cognizance
    of unintentional offences: you ought to have taken me privately, and warned
    and admonished me; for if I had been better advised, I should have left
    off doing what I only did unintentionally - no doubt I should; whereas
    you hated to converse with me or teach me, but you indicted me in this
    court, which is a place not of instruction, but of punishment.
    I have shown, Athenians, as I was saying, that Meletus has no care
    at all, great or small, about the matter. But still I should like to know,
    Meletus, in what I am affirmed to corrupt the young. I suppose you mean,
    as I infer from your indictment, that I teach them not to acknowledge the
    gods which the state acknowledges, but some other new divinities or spiritual
    agencies in their stead. These are the lessons which corrupt the youth,
    as you say.
    Yes, that I say emphatically.
    Then, by the gods, Meletus, of whom we are speaking, tell me and
    the court, in somewhat plainer terms, what you mean! for I do not as yet
    understand whether you affirm that I teach others to acknowledge some gods,
    and therefore do believe in gods and am not an entire atheist - this you
    do not lay to my charge; but only that they are not the same gods which
    the city recognizes - the charge is that they are different gods. Or, do
    you mean to say that I am an atheist simply, and a teacher of
    atheism?
    I mean the latter - that you are a complete
    atheist.
    That is an extraordinary statement, Meletus. Why do you say that?
    Do you mean that I do not believe in the godhead of the sun or moon, which
    is the common creed of all men?
    I assure you, judges, that he does not believe in them; for he
    says that the sun is stone, and the moon earth.
    Friend Meletus, you think that you are accusing Anaxagoras; and
    you have but a bad opinion of the judges, if you fancy them ignorant to
    such a degree as not to know that those doctrines are found in the books
    of Anaxagoras the Clazomenian, who is full of them. And these are the doctrines
    which the youth are said to learn of Socrates, when there are not unfrequently
    exhibitions of them at the theatre (price of admission one drachma at the
    most); and they might cheaply purchase them, and laugh at Socrates if he
    pretends to father such eccentricities. And so, Meletus, you really think
    that I do not believe in any god?
    I swear by Zeus that you believe absolutely in none at
    all.
    You are a liar, Meletus, not believed even by yourself. For I cannot
    help thinking, O men of Athens, that Meletus is reckless and impudent,
    and that he has written this indictment in a spirit of mere wantonness
    and youthful bravado. Has he not compounded a riddle, thinking to try me?
    He said to himself: - I shall see whether this wise Socrates will discover
    my ingenious contradiction, or whether I shall be able to deceive him and
    the rest of them. For he certainly does appear to me to contradict himself
    in the indictment as much as if he said that Socrates is guilty of not
    believing in the gods, and yet of believing in them - but this surely is
    a piece of fun.
    I should like you, O men of Athens, to join me in examining what
    I conceive to be his inconsistency; and do you, Meletus, answer. And I
    must remind you that you are not to interrupt me if I speak in my accustomed
    manner.
    Did ever man, Meletus, believe in the existence of human things,
    and not of human beings? ... I wish, men of Athens, that he would answer,
    and not be always trying to get up an interruption. Did ever any man believe
    in horsemanship, and not in horses? or in flute-playing, and not in flute-players?
    No, my friend; I will answer to you and to the court, as you refuse to
    answer for yourself. There is no man who ever did. But now please to answer
    the next question: Can a man believe in spiritual and divine agencies,
    and not in spirits or demigods?
    He cannot.
    I am glad that I have extracted that answer, by the assistance
    of the court; nevertheless you swear in the indictment that I teach and
    believe in divine or spiritual agencies (new or old, no matter for that);
    at any rate, I believe in spiritual agencies, as you say and swear in the
    affidavit; but if I believe in divine beings, I must believe in spirits
    or demigods; - is not that true? Yes, that is true, for I may assume that
    your silence gives assent to that. Now what are spirits or demigods? are
    they not either gods or the sons of gods? Is that true?
    Yes, that is true.
    But this is just the ingenious riddle of which I was speaking:
    the demigods or spirits are gods, and you say first that I don't believe
    in gods, and then again that I do believe in gods; that is, if I believe
    in demigods. For if the demigods are the illegitimate sons of gods, whether
    by the Nymphs or by any other mothers, as is thought, that, as all men
    will allow, necessarily implies the existence of their parents. You might
    as well affirm the existence of mules, and deny that of horses and asses.
    Such nonsense, Meletus, could only have been intended by you as a trial
    of me. You have put this into the indictment because you had nothing real
    of which to accuse me. But no one who has a particle of understanding will
    ever be convinced by you that the same man can believe in divine and superhuman
    things, and yet not believe that there are gods and demigods and
    heroes.
    I have said enough in answer to the charge of Meletus: any elaborate
    defence is unnecessary; but as I was saying before, I certainly have many
    enemies, and this is what will be my destruction if I am destroyed; of
    that I am certain; - not Meletus, nor yet Anytus, but the envy and detraction
    of the world, which has been the death of many good men, and will probably
    be the death of many more; there is no danger of my being the last of
    them.
    Someone will say: And are you not ashamed, Socrates, of a course
    of life which is likely to bring you to an untimely end? To him I may fairly
    answer: There you are mistaken: a man who is good for anything ought not
    to calculate the chance of living or dying; he ought only to consider whether
    in doing anything he is doing right or wrong - acting the part of a good
    man or of a bad. Whereas, according to your view, the heroes who fell at
    Troy were not good for much, and the son of Thetis above all, who altogether
    despised danger in comparison with disgrace; and when his goddess mother
    said to him, in his eagerness to slay Hector, that if he avenged his companion
    Patroclus, and slew Hector, he would die himself - "Fate," as she said,
    "waits upon you next after Hector"; he, hearing this, utterly despised
    danger and death, and instead of fearing them, feared rather to live in
    dishonor, and not to avenge his friend.  "Let me die next," he replies,
    "and be avenged of my enemy, rather than abide here by the beaked ships,
    a scorn and a burden of the earth." Had Achilles any thought of death and
    danger? For wherever a man's place is, whether the place which he has chosen
    or that in which he has been placed by a commander, there he ought to remain
    in the hour of danger; he should not think of death or of anything, but
    of disgrace. And this, O men of Athens, is a true saying.
    Strange, indeed, would be my conduct, O men of Athens, if I who,
    when I was ordered by the generals whom you chose to command me at Potidaea
    and Amphipolis and Delium, remained where they placed me, like any other
    man, facing death; if, I say, now, when, as I conceive and imagine, God
    orders me to fulfil the philosopher's mission of searching into myself
    and other men, I were to desert my post through fear of death, or any other
    fear; that would indeed be strange, and I might justly be arraigned in
    court for denying the existence of the gods, if I disobeyed the oracle
    because I was afraid of death: then I should be fancying that I was wise
    when I was not wise. For this fear of death is indeed the pretence of wisdom,
    and not real wisdom, being the appearance of knowing the unknown; since
    no one knows whether death, which they in their fear apprehend to be the
    greatest evil, may not be the greatest good. Is there not here conceit
    of knowledge, which is a disgraceful sort of ignorance? And this is the
    point in which, as I think, I am superior to men in general, and in which
    I might perhaps fancy myself wiser than other men, - that whereas I know
    but little of the world below, I do not suppose that I know: but I do know
    that injustice and disobedience to a better, whether God or man, is evil
    and dishonorable, and I will never fear or avoid a possible good rather
    than a certain evil. And therefore if you let me go now, and reject the
    counsels of Anytus, who said that if I were not put to death I ought not
    to have been prosecuted, and that if I escape now, your sons will all be
    utterly ruined by listening to my words - if you say to me, Socrates, this
    time we will not mind Anytus, and will let you off, but upon one condition,
    that are to inquire and speculate in this way any more, and that if you
    are caught doing this again you shall die; - if this was the condition
    on which you let me go, I should reply: Men of Athens, I honor and love
    you; but I shall obey God rather than you, and while I have life and strength
    I shall never cease from the practice and teaching of philosophy, exhorting
    anyone whom I meet after my manner, and convincing him, saying: O my friend,
    why do you who are a citizen of the great and mighty and wise city of Athens,
    care so much about laying up the greatest amount of money and honor and
    reputation, and so little about wisdom and truth and the greatest improvement
    of the soul, which you never regard or heed at all? Are you not ashamed
    of this? And if the person with whom I am arguing says: Yes, but I do care;
    I do not depart or let him go at once; I interrogate and examine and cross-examine
    him, and if I think that he has no virtue, but only says that he has, I
    reproach him with undervaluing the greater, and overvaluing the less. And
    this I should say to everyone whom I meet, young and old, citizen and alien,
    but especially to the citizens, inasmuch as they are my brethren. For this
    is the command of God, as I would have you know; and I believe that to
    this day no greater good has ever happened in the state than my service
    to the God. For I do nothing but go about persuading you all, old and young
    alike, not to take thought for your persons and your properties, but first
    and chiefly to care about the greatest improvement of the soul. I tell
    you that virtue is not given by money, but that from virtue come money
    and every other good of man, public as well as private. This is my teaching,
    and if this is the doctrine which corrupts the youth, my influence is ruinous
    indeed. But if anyone says that this is not my teaching, he is speaking
    an untruth. Wherefore, O men of Athens, I say to you, do as Anytus bids
    or not as Anytus bids, and either acquit me or not; but whatever you do,
    know that I shall never alter my ways, not even if I have to die many
    times.
    Men of Athens, do not interrupt, but hear me; there was an agreement
    between us that you should hear me out. And I think that what I am going
    to say will do you good: for I have something more to say, at which you
    may be inclined to cry out; but I beg that you will not do this. I would
    have you know that, if you kill such a one as I am, you will injure yourselves
    more than you will injure me. Meletus and Anytus will not injure me: they
    cannot; for it is not in the nature of things that a bad man should injure
    a better than himself. I do not deny that he may, perhaps, kill him, or
    drive him into exile, or deprive him of civil rights; and he may imagine,
    and others may imagine, that he is doing him a great injury: but in that
    I do not agree with him; for the evil of doing as Anytus is doing - of
    unjustly taking away another man's life - is greater far. And now, Athenians,
    I am not going to argue for my own sake, as you may think, but for yours,
    that you may not sin against the God, or lightly reject his boon by condemning
    me. For if you kill me you will not easily find another like me, who, if
    I may use such a ludicrous figure of speech, am a sort of gadfly, given
    to the state by the God; and the state is like a great and noble steed
    who is tardy in his motions owing to his very size, and requires to be
    stirred into life. I am that gadfly which God has given the state and all
    day long and in all places am always fastening upon you, arousing and persuading
    and reproaching you. And as you will not easily find another like me, I
    would advise you to spare me. I dare say that you may feel irritated at
    being suddenly awakened when you are caught napping; and you may think
    that if you were to strike me dead, as Anytus advises, which you easily
    might, then you would sleep on for the remainder of your lives, unless
    God in his care of you gives you another gadfly. And that I am given to
    you by God is proved by this: - that if I had been like other men, I should
    not have neglected all my own concerns, or patiently seen the neglect of
    them during all these years, and have been doing yours, coming to you individually,
    like a father or elder brother, exhorting you to regard virtue; this I
    say, would not be like human nature. And had I gained anything, or if my
    exhortations had been paid, there would have been some sense in that: but
    now, as you will perceive, not even the impudence of my accusers dares
    to say that I have ever exacted or sought pay of anyone; they have no witness
    of that. And I have a witness of the truth of what I say; my poverty is
    a sufficient witness.
    Someone may wonder why I go about in private, giving advice and
    busying myself with the concerns of others, but do not venture to come
    forward in public and advise the state. I will tell you the reason of this.
    You have often heard me speak of an oracle or sign which comes to me, and
    is the divinity which Meletus ridicules in the indictment. This sign I
    have had ever since I was a child. The sign is a voice which comes to me
    and always forbids me to do something which I am going to do, but never
    commands me to do anything, and this is what stands in the way of my being
    a politician. And rightly, as I think. For I am certain, O men of Athens,
    that if I had engaged in politics, I should have perished long ago and
    done no good either to you or to myself. And don't be offended at my telling
    you the truth: for the truth is that no man who goes to war with you or
    any other multitude, honestly struggling against the commission of unrighteousness
    and wrong in the state, will save his life; he who will really fight for
    the right, if he would live even for a little while, must have a private
    station and not a public one.
    I can give you as proofs of this, not words only, but deeds, which
    you value more than words. Let me tell you a passage of my own life, which
    will prove to you that I should never have yielded to injustice from any
    fear of death, and that if I had not yielded I should have died at once.
    I will tell you a story - tasteless, perhaps, and commonplace, but nevertheless
    true. The only office of state which I ever held, O men of Athens, was
    that of senator; the tribe Antiochis, which is my tribe, had the presidency
    at the trial of the generals who had not taken up the bodies of the slain
    after the battle of Arginusae; and you proposed to try them all together,
    which was illegal, as you all thought afterwards; but at the time I was
    the only one of the Prytanes who was opposed to the illegality, and I gave
    my vote against you; and when the orators threatened to impeach and arrest
    me, and have me taken away, and you called and shouted, I made up my mind
    that I would run the risk, having law and justice with me, rather than
    take part in your injustice because I feared imprisonment and death. This
    happened in the days of the democracy. But when the oligarchy of the Thirty
    was in power, they sent for me and four others into the rotunda, and bade
    us bring Leon the Salaminian from Salamis, as they wanted to execute him.
    This was a specimen of the sort of commands which they were always giving
    with the view of implicating as many as possible in their crimes; and then
    I showed, not in words only, but in deed, that, if I may be allowed to
    use such an expression, I cared not a straw for death, and that my only
    fear was the fear of doing an unrighteous or unholy thing. For the strong
    arm of that oppressive power did not frighten me into doing wrong; and
    when we came out of the rotunda the other four went to Salamis and fetched
    Leon, but I went quietly home. For which I might have lost my life, had
    not the power of the Thirty shortly afterwards come to an end. And to this
    many will witness.
    Now do you really imagine that I could have survived all these
    years, if I had led a public life, supposing that like a good man I had
    always supported the right and had made justice, as I ought, the first
    thing? No, indeed, men of Athens, neither I nor any other. But I have been
    always the same in all my actions, public as well as private, and never
    have I yielded any base compliance to those who are slanderously termed
    my disciples or to any other.  For the truth is that I have no regular
    disciples: but if anyone likes to come and hear me while I am pursuing
    my mission, whether he be young or old, he may freely come. Nor do I converse
    with those who pay only, and not with those who do not pay; but anyone,
    whether he be rich or poor, may ask and answer me and listen to my words;
    and whether he turns out to be a bad man or a good one, that cannot be
    justly laid to my charge, as I never taught him anything. And if anyone
    says that he has ever learned or heard anything from me in private which
    all the world has not heard, I should like you to know that he is speaking
    an untruth.
    But I shall be asked, Why do people delight in continually conversing
    with you? I have told you already, Athenians, the whole truth about this:
    they like to hear the cross-examination of the pretenders to wisdom; there
    is amusement in this. And this is a duty which the God has imposed upon
    me, as I am assured by oracles, visions, and in every sort of way in which
    the will of divine power was ever signified to anyone. This is true, O
    Athenians; or, if not true, would be soon refuted. For if I am really corrupting
    the youth, and have corrupted some of them already, those of them who have
    grown up and have become sensible that I gave them bad advice in the days
    of their youth should come forward as accusers and take their revenge;
    and if they do not like to come themselves, some of their relatives, fathers,
    brothers, or other kinsmen, should say what evil their families suffered
    at my hands. Now is their time. Many of them I see in the court. There
    is Crito, who is of the same age and of the same deme with myself; and
    there is Critobulus his son, whom I also see. Then again there is Lysanias
    of Sphettus, who is the father of Aeschines - he is present; and also there
    is Antiphon of Cephisus, who is the father of Epignes; and there are the
    brothers of several who have associated with me. There is Nicostratus the
    son of Theosdotides, and the brother of Theodotus (now Theodotus himself
    is dead, and therefore he, at any rate, will not seek to stop him); and
    there is Paralus the son of Demodocus, who had a brother Theages; and Adeimantus
    the son of Ariston, whose brother Plato is present; and Aeantodorus, who
    is the brother of Apollodorus, whom I also see. I might mention a great
    many others, any of whom Meletus should have produced as witnesses in the
    course of his speech; and let him still produce them, if he has forgotten
    - I will make way for him. And let him say, if he has any testimony of
    the sort which he can produce. Nay, Athenians, the very opposite is the
    truth. For all these are ready to witness on behalf of the corrupter, of
    the destroyer of their kindred, as Meletus and Anytus call me; not the
    corrupted youth only - there might have been a motive for that - but their
    uncorrupted elder relatives. Why should they too support me with their
    testimony? Why, indeed, except for the sake of truth and justice, and because
    they know that I am speaking the truth, and that Meletus is
    lying.
    Well, Athenians, this and the like of this is nearly all the defence
    which I have to offer. Yet a word more. Perhaps there may be someone who
    is offended at me, when he calls to mind how he himself, on a similar or
    even a less serious occasion, had recourse to prayers and supplications
    with many tears, and how he produced his children in court, which was a
    moving spectacle, together with a posse of his relations and friends; whereas
    I, who am probably in danger of my life, will do none of these things.
    Perhaps this may come into his mind, and he may be set against me, and
    vote in anger because he is displeased at this. Now if there be such a
    person among you, which I am far from affirming, I may fairly reply to
    him: My friend, I am a man, and like other men, a creature of flesh and
    blood, and not of wood or stone, as Homer says; and I have a family, yes,
    and sons. O Athenians, three in number, one of whom is growing up, and
    the two others are still young; and yet I will not bring any of them hither
    in order to petition you for an acquittal. And why not? Not from any self-will
    or disregard of you. Whether I am or am not afraid of death is another
    question, of which I will not now speak. But my reason simply is that I
    feel such conduct to be discreditable to myself, and you, and the whole
    state. One who has reached my years, and who has a name for wisdom, whether
    deserved or not, ought not to debase himself. At any rate, the world has
    decided that Socrates is in some way superior to other men. And if those
    among you who are said to be superior in wisdom and courage, and any other
    virtue, demean themselves in this way, how shameful is their conduct! I
    have seen men of reputation, when they have been condemned, behaving in
    the strangest manner: they seemed to fancy that they were going to suffer
    something dreadful if they died, and that they could be immortal if you
    only allowed them to live; and I think that they were a dishonor to the
    state, and that any stranger coming in would say of them that the most
    eminent men of Athens, to whom the Athenians themselves give honor and
    command, are no better than women. And I say that these things ought not
    to be done by those of us who are of reputation; and if they are done,
    you ought not to permit them; you ought rather to show that you are more
    inclined to condemn, not the man who is quiet, but the man who gets up
    a doleful scene, and makes the city ridiculous.
    But, setting aside the question of dishonor, there seems to be
    something wrong in petitioning a judge, and thus procuring an acquittal
    instead of informing and convincing him. For his duty is, not to make a
    present of justice, but to give judgment; and he has sworn that he will
    judge according to the laws, and not according to his own good pleasure;
    and neither he nor we should get into the habit of perjuring ourselves
    - there can be no piety in that. Do not then require me to do what I consider
    dishonorable and impious and wrong, especially now, when I am being tried
    for impiety on the indictment of Meletus. For if, O men of Athens, by force
    of persuasion and entreaty, I could overpower your oaths, then I should
    be teaching you to believe that there are no gods, and convict myself,
    in my own defence, of not believing in them. But that is not the case;
    for I do believe that there are gods, and in a far higher sense than that
    in which any of my accusers believe in them. And to you and to God I commit
    my cause, to be determined by you as is best for you and
    me.
    The jury finds Socrates guilty.
    Socrates' Proposal for his Sentence
    There are many reasons why I am not grieved, O men of Athens, at
    the vote of condemnation. I expected it, and am only surprised that the
    votes are so nearly equal; for I had thought that the majority against
    me would have been far larger; but now, had thirty votes gone over to the
    other side, I should have been acquitted. And I may say that I have escaped
    Meletus. And I may say more; for without the assistance of Anytus and Lycon,
    he would not have had a fifth part of the votes, as the law requires, in
    which case he would have incurred a fine of a thousand drachmae, as is
    evident.
    And so he proposes death as the penalty. And what shall I propose
    on my part, O men of Athens? Clearly that which is my due. And what is
    that which I ought to pay or to receive? What shall be done to the man
    who has never had the wit to be idle during his whole life; but has been
    careless of what the many care about - wealth, and family interests, and
    military offices, and speaking in the assembly, and magistracies, and plots,
    and parties. Reflecting that I was really too honest a man to follow in
    this way and live, I did not go where I could do no good to you or to myself;
    but where I could do the greatest good privately to everyone of you, thither
    I went, and sought to persuade every man among you that he must look to
    himself, and seek virtue and wisdom before he looks to his private interests,
    and look to the state before he looks to the interests of the state; and
    that this should be the order which he observes in all his actions. What
    shall be done to such a one? Doubtless some good thing, O men of Athens,
    if he has his reward; and the good should be of a kind suitable to him.
    What would be a reward suitable to a poor man who is your benefactor, who
    desires leisure that he may instruct you? There can be no more fitting
    reward than maintenance in the Prytaneum, O men of Athens, a reward which
    he deserves far more than the citizen who has won the prize at Olympia
    in the horse or chariot race, whether the chariots were drawn by two horses
    or by many. For I am in want, and he has enough; and he only gives you
    the appearance of happiness, and I give you the reality. And if I am to
    estimate the penalty justly, I say that maintenance in the Prytaneum is
    the just return.
    Perhaps you may think that I am braving you in saying this, as
    in what I said before about the tears and prayers. But that is not the
    case. I speak rather because I am convinced that I never intentionally
    wronged anyone, although I cannot convince you of that - for we have had
    a short conversation only; but if there were a law at Athens, such as there
    is in other cities, that a capital cause should not be decided in one day,
    then I believe that I should have convinced you; but now the time is too
    short. I cannot in a moment refute great slanders; and, as I am convinced
    that I never wronged another, I will assuredly not wrong myself. I will
    not say of myself that I deserve any evil, or propose any penalty. Why
    should I? Because I am afraid of the penalty of death which Meletus proposes?
    When I do not know whether death is a good or an evil, why should I propose
    a penalty which would certainly be an evil? Shall I say imprisonment? And
    why should I live in prison, and be the slave of the magistrates of the
    year - of the Eleven? Or shall the penalty be a fine, and imprisonment
    until the fine is paid? There is the same objection. I should have to lie
    in prison, for money I have none, and I cannot pay. And if I say exile
    (and this may possibly be the penalty which you will affix), I must indeed
    be blinded by the love of life if I were to consider that when you, who
    are my own citizens, cannot endure my discourses and words, and have found
    them so grievous and odious that you would fain have done with them, others
    are likely to endure me. No, indeed, men of Athens, that is not very likely.
    And what a life should I lead, at my age, wandering from city to city,
    living in ever-changing exile, and always being driven out! For I am quite
    sure that into whatever place I go, as here so also there, the young men
    will come to me; and if I drive them away, their elders will drive me out
    at their desire: and if I let them come, their fathers and friends will
    drive me out for their sakes.
    Someone will say: Yes, Socrates, but cannot you hold your tongue,
    and then you may go into a foreign city, and no one will interfere with
    you? Now I have great difficulty in making you understand my answer to
    this. For if I tell you that this would be a disobedience to a divine command,
    and therefore that I cannot hold my tongue, you will not believe that I
    am serious; and if I say again that the greatest good of man is daily to
    converse about virtue, and all that concerning which you hear me examining
    myself and others, and that the life which is unexamined is not worth living
    - that you are still less likely to believe. And yet what I say is true,
    although a thing of which it is hard for me to persuade you. Moreover,
    I am not accustomed to think that I deserve any punishment. Had I money
    I might have proposed to give you what I had, and have been none the worse.
    But you see that I have none, and can only ask you to proportion the fine
    to my means. However, I think that I could afford a minae, and therefore
    I propose that penalty; Plato, Crito, Critobulus, and Apollodorus, my friends
    here, bid me say thirty minae, and they will be the sureties. Well then,
    say thirty minae, let that be the penalty; for that they will be ample
    security to you.
    The jury condemns Socrates to death.
    Socrates' Comments on his Sentence
    Not much time will be gained, O Athenians, in return for the evil
    name which you will get from the detractors of the city, who will say that
    you killed Socrates, a wise man; for they will call me wise even although
    I am not wise when they want to reproach you. If you had waited a little
    while, your desire would have been fulfilled in the course of nature. For
    I am far advanced in years, as you may perceive, and not far from death.
    I am speaking now only to those of you who have condemned me to death.
    And I have another thing to say to them: You think that I was convicted
    through deficiency of words - I mean, that if I had thought fit to leave
    nothing undone, nothing unsaid, I might have gained an acquittal. Not so;
    the deficiency which led to my conviction was not of words - certainly
    not. But I had not the boldness or impudence or inclination to address
    you as you would have liked me to address you, weeping and wailing and
    lamenting, and saying and doing many things which you have been accustomed
    to hear from others, and which, as I say, are unworthy of me. But I thought
    that I ought not to do anything common or mean in the hour of danger: nor
    do I now repent of the manner of my defence, and I would rather die having
    spoken after my manner, than speak in your manner and live. For neither
    in war nor yet at law ought any man to use every way of escaping death.
    For often in battle there is no doubt that if a man will throw away his
    arms, and fall on his knees before his pursuers, he may escape death; and
    in other dangers there are other ways of escaping death, if a man is willing
    to say and do anything. The difficulty, my friends, is not in avoiding
    death, but in avoiding unrighteousness; for that runs faster than death.
    I am old and move slowly, and the slower runner has overtaken me, and my
    accusers are keen and quick, and the faster runner, who is unrighteousness,
    has overtaken them. And now I depart hence condemned by you to suffer the
    penalty of death, and they, too, go their ways condemned by the truth to
    suffer the penalty of villainy and wrong; and I must abide by my award
    - let them abide by theirs. I suppose that these things may be regarded
    as fated, - and I think that they are well.
    And now, O men who have condemned me, I would fain prophesy to
    you; for I am about to die, and that is the hour in which men are gifted
    with prophetic power. And I prophesy to you who are my murderers, that
    immediately after my death punishment far heavier than you have inflicted
    on me will surely await you. Me you have killed because you wanted to escape
    the accuser, and not to give an account of your lives. But that will not
    be as you suppose: far otherwise. For I say that there will be more accusers
    of you than there are now; accusers whom hitherto I have restrained: and
    as they are younger they will be more severe with you, and you will be
    more offended at them. For if you think that by killing men you can avoid
    the accuser censuring your lives, you are mistaken; that is not a way of
    escape which is either possible or honorable; the easiest and noblest way
    is not to be crushing others, but to be improving yourselves. This is the
    prophecy which I utter before my departure, to the judges who have condemned
    me.
    Friends, who would have acquitted me, I would like also to talk
    with you about this thing which has happened, while the magistrates are
    busy, and before I go to the place at which I must die. Stay then awhile,
    for we may as well talk with one another while there is time. You are my
    friends, and I should like to show you the meaning of this event which
    has happened to me. O my judges - for you I may truly call judges - I should
    like to tell you of a wonderful circumstance. Hitherto the familiar oracle
    within me has constantly been in the habit of opposing me even about trifles,
    if I was going to make a slip or error about anything; and now as you see
    there has come upon me that which may be thought, and is generally believed
    to be, the last and worst evil. But the oracle made no sign of opposition,
    either as I was leaving my house and going out in the morning, or when
    I was going up into this court, or while I was speaking, at anything which
    I was going to say; and yet I have often been stopped in the middle of
    a speech; but now in nothing I either said or did touching this matter
    has the oracle opposed me. What do I take to be the explanation of this?
    I will tell you. I regard this as a proof that what has happened to me
    is a good, and that those of us who think that death is an evil are in
    error. This is a great proof to me of what I am saying, for the customary
    sign would surely have opposed me had I been going to evil and not to
    good.
    Let us reflect in another way, and we shall see that there is great
    reason to hope that death is a good, for one of two things: - either death
    is a state of nothingness and utter unconsciousness, or, as men say, there
    is a change and migration of the soul from this world to another. Now if
    you suppose that there is no consciousness, but a sleep like the sleep
    of him who is undisturbed even by the sight of dreams, death will be an
    unspeakable gain. For if a person were to select the night in which his
    sleep was undisturbed even by dreams, and were to compare with this the
    other days and nights of his life, and then were to tell us how many days
    and nights he had passed in the course of his life better and more pleasantly
    than this one, I think that any man, I will not say a private man, but
    even the great king, will not find many such days or nights, when compared
    with the others. Now if death is like this, I say that to die is gain;
    for eternity is then only a single night. But if death is the journey to
    another place, and there, as men say, all the dead are, what good, O my
    friends and judges, can be greater than this? If indeed when the pilgrim
    arrives in the world below, he is delivered from the professors of justice
    in this world, and finds the true judges who are said to give judgment
    there, Minos and Rhadamanthus and Aeacus and Triptolemus, and other sons
    of God who were righteous in their own life, that pilgrimage will be worth
    making. What would not a man give if he might converse with Orpheus and
    Musaeus and Hesiod and Homer? Nay, if this be true, let me die again and
    again. I, too, shall have a wonderful interest in a place where I can converse
    with Palamedes, and Ajax the son of Telamon, and other heroes of old, who
    have suffered death through an unjust judgment; and there will be no small
    pleasure, as I think, in comparing my own sufferings with theirs. Above
    all, I shall be able to continue my search into true and false knowledge;
    as in this world, so also in that; I shall find out who is wise, and who
    pretends to be wise, and is not. What would not a man give, O judges, to
    be able to examine the leader of the great Trojan expedition; or Odysseus
    or Sisyphus, or numberless others, men and women too! What infinite delight
    would there be in conversing with them and asking them questions! For in
    that world they do not put a man to death for this; certainly not. For
    besides being happier in that world than in this, they will be immortal,
    if what is said is true.
    Wherefore, O judges, be of good cheer about death, and know this
    of a truth - that no evil can happen to a good man, either in life or after
    death. He and his are not neglected by the gods; nor has my own approaching
    end happened by mere chance. But I see clearly that to die and be released
    was better for me; and therefore the oracle gave no sign. For which reason
    also, I am not angry with my accusers, or my condemners; they have done
    me no harm, although neither of them meant to do me any good; and for this
    I may gently blame them.
    Still I have a favor to ask of them. When my sons are grown up,
    I would ask you, O my friends, to punish them; and I would have you trouble
    them, as I have troubled you, if they seem to care about riches, or anything,
    more than about virtue; or if they pretend to be something when they are
    really nothing, - then reprove them, as I have reproved you, for not caring
    about that for which they ought to care, and thinking that they are something
    when they are really nothing. And if you do this, I and my sons will have
    received justice at your hands.
    The hour of departure has arrived, and we go our ways - I to die,
    and you to live. Which is better God only knows.
    THE END



```python
#save to CSV
#plato.to_csv(f"{folder_path}/plato_works_dataframe.csv", index=False)
```

# Summary of pandas functions

As you can see, we used pandas to draw plots, access and subset the data, etc. Thinking about data in terms of rows and columns is very natural so pandas is indeed **a very convenient tool for understanding data.** Thus, getting a hand of pandas is important.

As a summary, here are some useful functions that you can refer to when using pandas.
    
|Name|Example|Purpose|
|-|-|-|
|`DataFrame`|`DataFrame()`|Create an empty DataFrame, usually to extend with data|
|`pd.read_csv`|`pandas.read_table("my_data.csv")`|Create a DataFrame from a data file|
|`pd.DataFrame({})`|`df = pandas.DataFrame({"N": np.arange(5), "2*N": np.arange(0, 10, 2)})`|Create a copy of a DataFrame with specified columns|
|`loc`|`df.loc[df["N"] > 10]`|Create a copy of a DataFrame with only the rows that match some *predicate*|
|`loc`|`df.loc["N"]`|Create a copy of a DataFrame with only specified column names|
|`(subsetting)`|`df[["N"]]`|Another way to create a copy of a DataFrame with only specified column names|
|`iloc`|`df.iloc(np.arange(0, 6, 2))`|Create a copy of the DataFrame with only the rows whose indices are in the given array|
|`sort`|`df.sort(["N"])`|Create a copy of a DataFrame sorted by the values in a column|
|`index`|`len(df.index)`|Compute the number of rows in a DataFrame|
|`columns`|`len(tbl.columns)`|Compute the number of columns in a DataFrame|
|`drop`|`df.drop(columns=["2*N"])`|Create a copy of a DataFrame without some of the columns|



