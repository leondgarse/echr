# Part I - XML and HTML Data


**Dependencies:**


```python
import pandas as pd
import xml.etree.cElementTree as ET #XML Parser
from lxml import etree #ElementTree and lxml allow us to parse the XML file.
import requests #make request to server
import time #pause loop
from bs4 import BeautifulSoup
```


## The Data

In this notebook, you'll be working with XML files from the Old Bailey API (https://www.oldbaileyonline.org/obapi/). These files contain the proceedings of all trials from 1674 to 1913. For this lab, we'll go through the trials from 1754-1756. XML (eXtensible Markup Language) provides a hierarchical representation of data contained within different tags and nodes. We'll go over XML syntax later. We will learn how to parse through these XML files from Old Bailey and grab information from sections of an XML file.





## Web Scraping

First we will go through how to parse one XML file. The Old Bailey API has a total of **197751** cases. Fortunately, we are only going to use the ones from 1754-1756, but that still only narrows the number of cases to somewhere above 1300! 

Don't worry though, you're not going to manually download each case yourself. This is where web scraping comes into play. With web scraping, we can automate data collection to get all the cases. 

Before we start scraping, we need to know how `requests` works. The `requests` library gets (`.get`) you a response object from a web server and will automatically decode the content from the server, from which you can use `.json()` to see the document! Requests through the Old Bailey API will return a dictionary, embedded in which is the XML representation of the trial account, which we can then write as a file and save.


Let's see how to grab the specific trial account files.

Clicking the URL below returns a web search result of the total number of cases, and a listing of all the cases, containing the term "sheffield" and the offence categrory "deception" from June 14th, 1847 onward. Also, each trial ID that satisfies the terms is returned; the count parameter in this case returns 74 trial IDs. The API returns how many trial IDs it finds, but it only hands out the trial account files ten at a time. The query parameter for start year is `year_gte=` and for end year is `year_lte=`

https://www.dhi.ac.uk/api/data/oldbailey_record?month_gte=6&offence=deception&text=Sheffield&year_gte=1847#results


Let us use requests.get(...) to get all the trials between the years 1754 and 1756 and return them as a JSON object, and then find how many total trial accounts there are.


```python
url = "https://www.dhi.ac.uk/api/data/oldbailey_record"        # Base URL for the Old Bailey API endpoint
start_yr = 1754
end_yr = 1756

params = {"year_gte": start_yr, "year_lte": end_yr }

headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

r = requests.get(url, params=params, headers=headers)
r.raise_for_status()

trials = r.json()
```

At this point it might pay to look at the JSON object that `requests.get` returned to see what the list of trial accounts actually looks like. You can see that the first `hits` is a dictionary index which records a `total` for the query, and that the second `hits` is the key for the list of trial accounts.

You can see below that trials is a big dictionary, inside of which is a list of trials, each of which has a trial ID. The API documentation talks about what [data endpoints](https://www.oldbaileyonline.org/about/api#data_endpoints) and [query parameters](https://www.oldbaileyonline.org/about/api#supported_parameters) you can use. 

We will have to do some work to extract all the trials we want over the two year period (1754-1756) that we want to look at. The `total:` key below tells us that there are a total of 1312 trial records in the period 1754-1756. The numbering scheme for the `idkey:` also tells us something about the organization of the records; the prefix `f` is for front matter, while `t` is for trial account. After that is the date with digits for year, month, and day, followed by a hyphen with how many accounts are in the series for that day. Note that many trials occurred on a day when the Old Bailey was in session.


```python
trials
```




    {'took': 4,
     'timed_out': False,
     '_shards': {'total': 5, 'successful': 5, 'skipped': 0, 'failed': 0},
     'hits': {'total': 1312,
      'max_score': None,
      'hits': [{'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mD',
        '_score': None,
        '_source': {'idkey': 'f17540116-1',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160001.gif',
          'https://www.dhi.ac.uk/san/ob/1750s/175401160002.gif'],
         'text': "THE PROCEEDINGS ON THE King's Commissions of the Peace, Oyer and Terminer, and Gaol Delivery FOR THE CITY of LONDON; And also the Gaol Delivery for the County of MIDDLESEX, HELD AT JUSTICE-HALL in the OLD-BAILEY, On Wednesday the 16th, Thursday the 17th, Friday the 18th, Saturday the 19th, and Monday the 21st, of JANUARY. In the 27th Year of His MAJESTY's Reign. NUMBER II. for the Year 1754. BEING THE Second SESSIONS in the MAYORALTY of the Right Hon. Thomas Rawlinson , Esq; LORD-MAYOR of the CI",
         'title': 'Front matter. 16th January 1754.'},
        'sort': [-6814972800000, 0]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mE',
        '_score': None,
        '_source': {'idkey': 't17540116-1',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160002.gif'],
         'text': '80. Hannah Ash , spinster , was indicted for stealing one linen shift, one cotton gown, one linen handkerchief, one pair of silver sleeve buttons, and one half guinea , the goods and money of Richard Beach , Dec. 5 . To which she pleaded guilty . [Transportation. See summary.]',
         'title': 'Hannah Ash. Theft; grand larceny (to 1827). 16th January 1754.'},
        'sort': [-6814972800000, 1]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mF',
        '_score': None,
        '_source': {'idkey': 't17540116-2',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160002.gif'],
         'text': '81. (M.) Peter Foreman and Mary his wife were indicted for stealing one pair of linen sheets, value 6 d. one copper sauce-pan, value 2 d. one bolster, value 2 d. the goods of Joseph Sheers , in a certain lodging-room let by contract Dec. 1 . ++ The prosecutor missed the goods mentioned out of the prisoners lodgings; they were taken up, and both confessed the taking them, and also where they were pawned; and they were found accordingly. The prisoners had nothing to say in their defence. Both Guil',
         'title': 'Mary Foreman. Peter Foreman. Theft; theft from a specified place. 16th January 1754.'},
        'sort': [-6814972800000, 2]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mG',
        '_score': None,
        '_source': {'idkey': 't17540116-3',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160002.gif',
          'https://www.dhi.ac.uk/san/ob/1750s/175401160003.gif'],
         'text': '82. (M.) Sarah Williams , spinster , was indicted for stealing one brass kettle, value 10 s. the property of Joseph Smithson , Dec. 28 . ++ Ann Smithson . I am wife to Joseph Smithson , we live in Round Court, in the Strand , he is a broker . On the 27th of December last I was informed a woman had taken a large pot from the door; I followed the woman into Chandos-street, where she was running along with it, but seeing me come after her dropped it, and one Hawkins took her; I took the pot up, and',
         'title': 'Sarah Williams. Theft; grand larceny (to 1827). 16th January 1754.'},
        'sort': [-6814972800000, 3]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mH',
        '_score': None,
        '_source': {'idkey': 't17540116-4',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160003.gif'],
         'text': '83. (M.) Elizabeth wife of Joseph Kempster , was indicted for stealing one feather-bed, value 14 s. one bolster 2 s. one pillow, one flaxen sheet, one copper tea-kettle, one brass candlestick, one pair of bellows, the goods of Mary Kennedy , widow , in a certain lodging let by contract, &c. December 23 . ++ Mary Kennedy . I live in Scotland-Yard , the prisoner was with me six or seven weeks before she went away, then she took the key with her; the room was let to her, her husband she said was in',
         'title': 'Elizabeth Kempster. Theft; theft from a specified place. 16th January 1754.'},
        'sort': [-6814972800000, 4]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mI',
        '_score': None,
        '_source': {'idkey': 't17540116-5',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160003.gif'],
         'text': '84. (M.) John Allen was indicted for stealing one linen shirt, value 1 s. 6 d. the property of Thomas Fazakerley , Dec. 15 . ++ Acquitted .',
         'title': 'John Allen. Theft; grand larceny (to 1827). 16th January 1754.'},
        'sort': [-6814972800000, 5]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mJ',
        '_score': None,
        '_source': {'idkey': 't17540116-6',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160003.gif'],
         'text': '85. (M.) William Derter was indicted for stealing 70 lb. weight of rags, value 4 s. the goods of Thomas Wetworth , Jan. 11 . ++ Acquitted .',
         'title': 'William Derter. Theft; grand larceny (to 1827). 16th January 1754.'},
        'sort': [-6814972800000, 6]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mK',
        '_score': None,
        '_source': {'idkey': 't17540116-7',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160003.gif',
          'https://www.dhi.ac.uk/san/ob/1750s/175401160004.gif'],
         'text': '86. (M.) William Ford was indicted for stealing one mare, of a black colour, value 12 l. the property of Nicholas Healing , Oct. 4 . ++ Nicholas Healing . I am a butcher , and live at Hounslow ; I had a black mare, along with two other horses and a mare, in my ground, about a quarter of a mile from my house, they were all safe over night, but on the 4th of October in the morning the black mare was missing. I found her again on Tuesday the 9th at Spinham-land in Berkshire, in the possession of Ab',
         'title': 'William Ford. Theft; animal theft. 16th January 1754.'},
        'sort': [-6814972800000, 7]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mL',
        '_score': None,
        '_source': {'idkey': 't17540116-8',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160004.gif'],
         'text': "87. (L.) Anne Beezley , spinster , was indicted for stealing a set of green bed curtains and vallens, val. 10 s. one pair of linen sheets, two blankets, one copper saucepan, one copper teakettle, two flat irons, three brass candlesticks, one looking glass, the goods of John Jervas , the same being in a certain lodging room let by contract, &c. Dec. 13. * John Jervas. I let a ready-furnish'd lodging to the prisoner the 6th of last month. She went away the 13th in the evening; not returning, I wen",
         'title': 'Anne Beezley. Theft; theft from a specified place. 16th January 1754.'},
        'sort': [-6814972800000, 8]},
       {'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mM',
        '_score': None,
        '_source': {'idkey': 't17540116-9',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160004.gif'],
         'text': '88. Robert Barber was indicted for that he, together with John Thorp , not yet taken, did steal forty guineas, two half guineas, five 36 s. pieces, four moidores, one two moidore and half piece, one quarter of a guinea , the money of Abraham Julian , Dec. 16 . To which he pleaded Guilty . He was a second time indicted for forging a certain acquittance for the sum of 13 s. 11 d. 1/2, and uttering the same as true, well knowing it to have been forged, with intent to defraud , Dec. 4 . As no eviden',
         'title': 'Robert Barber. Theft; grand larceny (to 1827), Deception; forgery. 16th January 1754.'},
        'sort': [-6814972800000, 9]}]}}




```python
total_hits = trials['hits']['total']
total_hits
```




    1312



It is clear that the JSON object is pretty complicated. 

There is a top-level dictionary with the key `'hits'` and then an embedded dictionary with the key `'total'` (that is where we got the total number of trial accounts above) and then a second `'hits'` and then a list of dictionaries and more embedded dictionaries. This is very complicated! Notice also that the very first XML document is not a trial record but is instead the front matter from that printed version of the Old Bailey Proceedings.

So with the (new) API it looks like we may have to walk through each group of ten trial account documents until we have gotten the entire list, and then eventually parse the XML tree for each document using Beautiful Soup. 

For your projects, you can download the [dataset archived](https://www.oldbaileyonline.org/about/data#toc3) at the University of Sheffield so you do not have to scrape them from the site. Even the zipped file of Sessions Papers is pretty large (335MB). 

In the JSON object that we got back, [object.hits.hits](https://www.oldbaileyonline.org/about/api#response_format) is the list of the XML docs.


```python
len(trials['hits']['hits'])
```




    10




```python
# Define our  initial variables
xml_list = [] # This list will store all trial records collected across multiple API requests
i = 0            # Offset counter used for pagination (the API returns results in batches of 10)
```


```python
# Get the total count dynamically so the loop knows when to stop
url = "https://www.dhi.ac.uk/api/data/oldbailey_record"

params = {'year_gte': start_yr, 
          'year_lte': end_yr}

headers = {"User-Agent": "Mozilla/5.0"} # Prevents 403 errors

probe = requests.get(url, 
                     params=params, 
                     headers=headers).json()

total_hits = probe['hits']['total']  # 1312
```


```python
total_hits
```




    1312




```python
# Get the xmls
while i < total_hits:                                  # Continue requesting data until we have fetched all available trials
    query_url = f'{url}?from={i}&size=10&year_gte={start_yr}&year_lte={end_yr}' # Construct the API request URL with pagination (from/size) and year filters
    try:                                           # Send the GET request to the server using the browser-mimicking headers defined earlier
        response = requests.get(query_url,  
                                headers=headers)
        response.raise_for_status()                 # Raise an error if the request failed (e.g., 403 Forbidden or 500 Server Error)
        trials = response.json()                    # Parse the JSON response body into a Python dictionary
        batch = trials['hits']['hits']              # Extract the list of records (the "hits") from the nested JSON structure
        if not batch:                              # Safety break if the server returns empty results
            break 
            
        xml_list.extend(batch)                                        # Add the 10 records from this batch into our master list (xml_list)
        if len(xml_list) % 100 == 0 or len(xml_list) >= total_hits:    # Logic to print progress every 100 records - Provide a visual update
            print(f"Progress: {len(xml_list)} / {total_hits}")
            
        i += 10         # Increment the offset by 10 to "turn the page" for the next API request
        time.sleep(0.1) # Pause for 0.1 seconds to avoid triggering the server's anti-bot/rate-limiting protections
        
    except Exception as e:     # If any error occurs (network issue, JSON error, etc.), print the error and stop
        print(f"Error at offset {i}: {e}")
        break
```

    Progress: 100 / 1312
    Progress: 200 / 1312
    Progress: 300 / 1312
    Progress: 400 / 1312
    Progress: 500 / 1312
    Progress: 600 / 1312
    Progress: 700 / 1312
    Progress: 800 / 1312
    Progress: 900 / 1312
    Progress: 1000 / 1312
    Progress: 1100 / 1312
    Progress: 1200 / 1312
    Progress: 1300 / 1312
    Progress: 1312 / 1312


Remember that each item in `xml_list` is a dictionary. To get the id's, we will need to look at `idkey`  and to get the xml we'll need to look at the `xml` 


```python
xml_list[-10:] # last 10 entries - to check if we got the 1756
```




    [{'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vh',
      '_score': None,
      '_source': {'idkey': 't17561208-39',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080028.gif',
        'https://www.dhi.ac.uk/san/ob/1750s/175612080029.gif'],
       'text': '45. (L.) John Milward was indicted for forging a counterfeit bill of exchange. purporting to have been drawn by Joseph Sill , Thomas Bridges , and Roger Blunt , directed to Francis Blunt , merchant, requiring him to pay to him, the said John Milward , or order, the sum of 60 l. six weeks after date, for value received, as per advice, accepted F. Blunt, November 4, 1756 , and for publishing the same, well knowing it to have been forged and counterfeited, with an intent to defraud John Swallow ++ ',
       'title': 'John Milward. Deception; forgery. 8th December 1756.'},
      'sort': [-6723648000000, 39]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vi',
      '_score': None,
      '_source': {'idkey': 't17561208-40',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080029.gif',
        'https://www.dhi.ac.uk/san/ob/1750s/175612080030.gif'],
       'text': '46. (M.) Joseph Goston was indicted for stealing one pair of leather breeches, value 5 s. the property of Charles Earles , Oct. 22 ++ Charles Earles . I employed the prisoner at the bar to serve me at Sittingbourn-Fair and also at Maidstone-Fair, in Kent, He was with me at Sittingbourn, as a journeyman , to sell goods. At my going out of town I had lent him a pair of leather breeches. They were ripped in the side. He left me in the country, and would not serve me at Maidstone-Fair, and I challen',
       'title': 'Joseph Goston. Theft; grand larceny (to 1827). 8th December 1756.'},
      'sort': [-6723648000000, 40]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vj',
      '_score': None,
      '_source': {'idkey': 't17561208-41',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080030.gif',
        'https://www.dhi.ac.uk/san/ob/1750s/175612080031.gif'],
       'text': "47. (M.) Ann wife of Thomas Sanders was indicted for stealing one metal watch, value 20 s. and two seals, value 2 s. the property of Charles Drew , December 2 . ++ Charles Drew . On the second of this instant I had a mind to go in and hear Mr. Whitfield preach at his chapel in Long-Acre . But not meeting with the satisfaction I expected, I came out very soon; (I found he had not alter'd his method of preaching) and coming out at the chapel door the woman at the bar laid hold of my watch. I layin",
       'title': 'Ann Sanders. Theft; grand larceny (to 1827). 8th December 1756.'},
      'sort': [-6723648000000, 41]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vk',
      '_score': None,
      '_source': {'idkey': 't17561208-42',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080031.gif',
        'https://www.dhi.ac.uk/san/ob/1750s/175612080032.gif'],
       'text': "48. (M.) James Brannock was indicted for stealing one pair of leather pumps, value is. 6 d. the property of James Murray , Nov. 30 . ++ James Murray . The prisoner worked with me as a servant Q. What are you ? Murray. I am a shoemaker ; on the 30th of November he went out about eight o'clock, and return'd about four in the afternoon; he took these two pumps, and put one in each pocket. I followed him and took him with them upon him. Q. Did you see him put them into his pocket ? Murray. No, I did",
       'title': 'James Brannock. Theft; grand larceny (to 1827). 8th December 1756.'},
      'sort': [-6723648000000, 42]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vl',
      '_score': None,
      '_source': {'idkey': 't17561208-43',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080032.gif'],
       'text': '49. (M.) Elizabeth Broomfield , widow , was indicted for stealing one copper pottage pot, value 4 s. one copper tea kettle, one copper stew pan, one copper sauce pan, two blankets, and two linen sheets , the goods of John Votier , Nov. 22 . ++ Mary Votier . The prisoner nursed me in my lying-in, and took away the things mentioned in the indictment, when I was sick in bed. Q. How do you know that? Votier. I am very sure they were in the kitchen, and no body was there but she. Q. How long was she ',
       'title': 'Elizabeth Broomfield. Theft; grand larceny (to 1827). 8th December 1756.'},
      'sort': [-6723648000000, 43]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vm',
      '_score': None,
      '_source': {'idkey': 't17561208-44',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080032.gif',
        'https://www.dhi.ac.uk/san/ob/1750s/175612080033.gif'],
       'text': "50. (L.) John Anderson was indicted for stealing 59 women's hats, call'd chip hats, value 38 s. the goods of Samuel Lloyd , Esq ; December 19 . ++ John Dudley . Last night, betwixt the hours of four and five, I was sent to a lighter on duty. I saw her moor'd in, I was on the shore and the vessel just by. I saw the prisoner poking his head down in a dark place by a hogshead that was on shore. Q. What was he doing? Dudley. He was pudling and mudling about near where my lighter lay. I ask'd him wha",
       'title': 'John Anderson. Theft; grand larceny (to 1827). 8th December 1756.'},
      'sort': [-6723648000000, 44]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vn',
      '_score': None,
      '_source': {'idkey': 'o17561208-1',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080033.gif'],
       'text': "John Cartwright , capitally convicted in September sessions, Jonathan Hirst and Francis Mugford , in October sessions, Bartholomew Ball , John Jolley , Edward M'Collister , and John Milward were executed on Monday the 20th of December.",
       'title': "Supplementary material. John Cartwright. Jonathan Hirst. Francis Mugford. Bartholomew Ball. John Jolley. Edward M'Collister. John Milward. 8th December 1756."},
      'sort': [-6723648000000, 45]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vo',
      '_score': None,
      '_source': {'idkey': 's17561208-1',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080033.gif',
        'https://www.dhi.ac.uk/san/ob/1750s/175612080034.gif',
        'https://www.dhi.ac.uk/san/ob/1750s/175612080034.gif'],
       'text': "The trials being ended, the court proceeded to give judgment as follows: Received sentence of death 5. Bartholomew Ball , Edward M'Collister, John Milward , William Pallister, and John Jolley . Transportation for seven years 22. Thomas Holland , Daniel Jones , William Wells , Mary Peck, Edward Ware , Moses Abraham , Mary Martin , Charles Sawyer , John Anderson , John Phelps , John Simpson , William Parker, Jane Philips , John Ilford , Mary Plumer , G - W - , Ann Fauklin , Mary Field , Ann Sander",
       'title': "Punishment summary. John Cartwright. Jonathan Hirst. Francis Mugford. Bartholomew Ball. John Jolley. Edward M'Collister. John Milward. 8th December 1756."},
      'sort': [-6723648000000, 46]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vp',
      '_score': None,
      '_source': {'idkey': 's17561208-1',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080034.gif'],
       'text': "John Cartwright , capitally convicted in September sessions, Jonathan Hirst and Francis Mugford , in October sessions, Bartholomew Ball , John Jolley , Edward M'Collister , and John Milward were executed on Monday the 20th of December.",
       'title': "Supplementary material. John Cartwright. Jonathan Hirst. Francis Mugford. Bartholomew Ball. John Jolley. Edward M'Collister. John Milward. 8th December 1756."},
      'sort': [-6723648000000, 46]},
     {'_index': 'dhids_oldbailey_record',
      '_type': 'doc',
      '_id': 'AYxEMBqEPld_y58Rz6vq',
      '_score': None,
      '_source': {'idkey': 'a17561208-1',
       'images': ['https://www.dhi.ac.uk/san/ob/1750s/175612080034.gif'],
       'text': 'Just published (2d Edition) Price bound 8 s. BRACHYGRAPHY: OR, SHORT-WRITING Made easy to the meanest Capacity: The Persons, Moods and Tenses, being comprized in such a Manner, that little more than the Knowledge of the Alphabet is required to the writing Hundreds of Sentences in less Time than spoken. The whole is founded on so just a Plan, that it is wrote with greater Expedition than any yet invented, and likewise may be read with the greatest Ease. Improved (after upwards o Thirty Years Prac',
       'title': 'Advertisements. 8th December 1756.'},
      'sort': [-6723648000000, 48]}]




```python
# inspect the first trial xml in the list
xml_list[0]
```




    {'_index': 'dhids_oldbailey_record',
     '_type': 'doc',
     '_id': 'AYxEMNu7Pld_y58Rz_mD',
     '_score': None,
     '_source': {'idkey': 'f17540116-1',
      'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160001.gif',
       'https://www.dhi.ac.uk/san/ob/1750s/175401160002.gif'],
      'text': "THE PROCEEDINGS ON THE King's Commissions of the Peace, Oyer and Terminer, and Gaol Delivery FOR THE CITY of LONDON; And also the Gaol Delivery for the County of MIDDLESEX, HELD AT JUSTICE-HALL in the OLD-BAILEY, On Wednesday the 16th, Thursday the 17th, Friday the 18th, Saturday the 19th, and Monday the 21st, of JANUARY. In the 27th Year of His MAJESTY's Reign. NUMBER II. for the Year 1754. BEING THE Second SESSIONS in the MAYORALTY of the Right Hon. Thomas Rawlinson , Esq; LORD-MAYOR of the CI",
      'title': 'Front matter. 16th January 1754.'},
     'sort': [-6814972800000, 0]}




```python
# and another
xml_list[1002]
```




    {'_index': 'dhids_oldbailey_record',
     '_type': 'doc',
     '_id': 'AYxEL1POPld_y58Rz1_N',
     '_score': None,
     '_source': {'idkey': 't17560225-46',
      'images': ['https://www.dhi.ac.uk/san/ob/1750s/175602250045.gif',
       'https://www.dhi.ac.uk/san/ob/1750s/175602250046.gif'],
      'text': "164. (M.) Mary Kingston , spinster , was indicted for stealing 10 yards of ribband , the property of Sarah Young , Feb. 10 . + Sarah Young . I live in Great-turnstile, Holbourn , and keep a millener's shop ; the ribband was taken out of my shop on the 10th of February, between 8 and 9 o'clock in the morning, it was taken from my maid, I can swear to one of the pieces. Q. When did you see the ribband last, before it was lost? S. Young. It was in my shop that morning, my maid came and told me what",
      'title': 'Mary Kingston. Theft; other. 25th February 1756.'},
     'sort': [-6748444800000, 48]}



Now we can see clearly what the requests call got us. It is a list of dictionaries that has various keys, including 'idkey' for the document ID, 'images' for the page images of the paper Old Bailey Proceeds, and 'text', which shows **just the plain text** for the document (which is incomplete). Notice the texts tend to end mid-sentence.

We did **not** get the XML representation of the trial record, which has tags and labels for the parties etc., as we saw when we got the list of terms above. 

What we need to do next is
* get the list of ID keys for all 1312 documents from 1754 to 1756 (some of these will not be trial session records but will be front matter or advertising--we won't worry about that now)
* use the list of individual ID keys to make requests for the xml files for each ID key
* write the xml files to the local data directory
* use those xml files to build a dataframe

First we need to pick out the document ID from each item in the list.


```python
xml_list[1001]['_source']['idkey']
```




    't17560225-45'



Now, let's get a list of document IDs we can work with. Then we can use the first ten to demonstrate the next step of calling the API to get the xml file for each document ID.


```python
doc_ids = [xml_list[i]['_source']['idkey'] for i in range(0, len(xml_list))]
first_10 = doc_ids[:10]
first_10
```




    ['f17540116-1',
     't17540116-1',
     't17540116-2',
     't17540116-3',
     't17540116-4',
     't17540116-5',
     't17540116-6',
     't17540116-7',
     't17540116-8',
     't17540116-9']



Using the trial IDs from the previous cell, we are going to format the URL in a way so that we can get the XML file for each trial. In order to get the XML file using the Old Bailey API, we must follow this URL format:

<p style="text-align: center;">`https://www.dhi.ac.uk/api/data/oldbailey_record_single?idkey=`  </p>

For example, https://www.dhi.ac.uk/api/data/oldbailey_record_single?idkey=t16740429-1 gives you the link to the XML file of the first proceeding in the database.


**To get the XML file of the fifth trial in first_10, do the following:.** (A successful `.get` request returns `<Response [200]>`.)


```python
fifth_trial_id = first_10[4]
fifth_trial_id
```




    't17540116-4'




```python
url = f'https://www.dhi.ac.uk/api/data/oldbailey_record_single?idkey={fifth_trial_id}'

#  Make the request with the headers
response = requests.get(url, 
                        headers=headers)
```

Run the next cell to see the XML format of the text


```python
response.json()
```




    {'took': 1,
     'timed_out': False,
     '_shards': {'total': 20, 'successful': 20, 'skipped': 0, 'failed': 0},
     'hits': {'total': 1,
      'max_score': 10.195236,
      'hits': [{'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMNu7Pld_y58Rz_mH',
        '_score': 10.195236,
        '_source': {'metadata': '<table xmlns:dhids="https://www.dhi.ac.uk/data/" class="table small"><tbody><tr><th scope="row">Text type</th><td>Trial account</td></tr><tr><th scope="row">Defendants</th><td>Elizabeth Kempster</td></tr><tr><th scope="row">Offences</th><td><a href="../about/crimes#theft">Theft</a> &gt; <a href="../about/crimes#theftfromaspecifiedplace">Theft from place</a></td></tr><tr><th scope="row">Session Date</th><td><a href="17540116#t17540116-4">16th January 1754</a></td></tr><tr><th scope="row">Reference Number</th><td>t17540116-4</td></tr><tr><th scope="row">Verdicts</th><td><a href="../about/verdicts#guilty">Guilty</a></td></tr><tr><th scope="row">Punishments</th><td><a href="../about/punishment#transportation">Transportation</a></td></tr><tr><th scope="row">Related Material</th><td><ul><li><b>Digital Panopticon</b> <a href="https://www.digitalpanopticon.org/life?id=obpt17540116-4-defend46">Elizabeth Joseph Kempster</a> 2 records</li><li><b>Associated Record</b> <a href="ar_4437_48007">Document. 00 1754. London Metropolitan Archives MJ/SP/1754/01/041.</a></li></ul></td></tr><tr><th scope="row">Navigation</th><td><a href="t17540116-3">&lt; Previous section (Trial account)</a> | <a href="t17540116-5">Next section (Trial account) &gt;</a></td></tr></tbody></table>',
         'images': ['https://www.dhi.ac.uk/san/ob/1750s/175401160003.gif'],
         'exturl': 'https://www.oldbaileyonline.org/browse.jsp?div=t17540116-4',
         'citation': '<span><i>Old Bailey Proceedings Online</i> (www.oldbaileyonline.org, version 9.0) January 1754. Trial of Elizabeth wife of Joseph Kempster (t17540116-4).</span>',
         'project': 'oldbailey',
         'collection': 'record',
         'title': 'Elizabeth Kempster. Theft; theft from a specified place. 16th January 1754.',
         'image_titles': ['Proceedings of the Old Bailey, 16th January .'],
         'idkey': 't17540116-4',
         'navigation': '<span><a href="t17540116-3">&lt; Previous section (Trial account)</a> | <a href="t17540116-5">Next section (Trial account) &gt;</a></span>',
         'xml': '<div1 type="trialAccount" id="t17540116-4"> <interp inst="t17540116-4" type="prevdiv" value="t17540116-3" divtype="trialAccount"/> <interp inst="t17540116-4" type="nextdiv" value="t17540116-5" divtype="trialAccount"/> <interp inst="t17540116-4" type="div0" value="17540116" divtype="sessionsPaper"/> <interp inst="t17540116-4" type="assocrec" value="ar_4437_48007" title="Document. 00 1754. London Metropolitan Archives MJ/SP/1754/01/041."/> <interp inst="t17540116-4" type="collection" value="BAILEY"/> <interp inst="t17540116-4" type="year" value="1754"/> <interp inst="t17540116-4" type="uri" value="sessionsPapers/17540116"/> <interp inst="t17540116-4" type="date" value="17540116"/> <xptr imgpath="ob/1750s/175401160003.gif" imgtitle="Proceedings of the Old Bailey, 16th January ." imgrights="This image is reproduced by permission of Harvard University Library from the microfilm, &quot;The Old Bailey Proceedings&quot;, (Harvester Microform, a former imprint of the Gale Group, 1983). Commercial use is prohibited without permission of the owner of the original." type="pageFacsimile" value="preceding" doc="175401160003"/> <join result="criminalCharge" id="t17540116-4-off20-c41" targOrder="Y" targets="t17540116-4-defend46 t17540116-4-off20 t17540116-4-verdict24"/> <p>83. (M.) <interp inst="t17540116-4-defend46" type="digipan" value="Elizabeth Joseph Kempster" title="2 records"/> <persName id="t17540116-4-defend46" type="defendantName"> Elizabeth <rs id="t17540116-4-deflabel19" type="occupation">wife</rs> <interp inst="t17540116-4-deflabel19" type="defendantNameLabel" value="wife"/> <join result="persNameOccupation" targOrder="Y" targets="t17540116-4-defend46 t17540116-4-deflabel19"/> of <persName id="t17540116-4-person47"> Joseph Kempster <interp inst="t17540116-4-person47" type="surname" value="Kempster"/> <interp inst="t17540116-4-person47" type="given" value="Joseph"/> <interp inst="t17540116-4-person47" type="gender" value="male"/> <interp inst="t17540116-4-person47" type="age" value="unknown"/> </persName> <interp inst="t17540116-4-defend46" type="surname" value="Kempster"/> <interp inst="t17540116-4-defend46" type="given" value="Elizabeth"/> <interp inst="t17540116-4-defend46" type="gender" value="female"/> </persName> , was indicted for <rs id="t17540116-4-off20" type="offenceDescription"> <interp inst="t17540116-4-off20" type="offenceCategory" value="theft"/> <interp inst="t17540116-4-off20" type="offenceSubcategory" value="theftFromPlace"/> stealing one feather-bed, value 14 s. one bolster 2 s. one pillow, one flaxen sheet, one copper tea-kettle, one brass candlestick, one pair of bellows, the goods of <interp inst="t17540116-4-victim49" type="victims-institution" value="no"/> <interp inst="t17540116-4-victim49" type="victims-hisco-code" value="0"/> <persName id="t17540116-4-victim49" type="victimName"> Mary Kennedy <interp inst="t17540116-4-victim49" type="surname" value="Kennedy"/> <interp inst="t17540116-4-victim49" type="given" value="Mary"/> <interp inst="t17540116-4-victim49" type="gender" value="female"/> <join result="offenceVictim" targOrder="Y" targets="t17540116-4-off20 t17540116-4-victim49"/> </persName> , <rs id="t17540116-4-viclabel21" type="occupation">widow</rs> <interp inst="t17540116-4-viclabel21" type="victimNameLabel" value="widow"/> <join result="persNameOccupation" targOrder="Y" targets="t17540116-4-victim49 t17540116-4-viclabel21"/>, in a certain lodging </rs> let by contract, &amp;c. <rs id="t17540116-4-cd22" type="crimeDate">December 23</rs> <join result="offenceCrimeDate" targOrder="Y" targets="t17540116-4-off20 t17540116-4-cd22"/>. ++</p> <p> <persName id="t17540116-4-person50"> Mary Kennedy <interp inst="t17540116-4-person50" type="surname" value="Kennedy"/> <interp inst="t17540116-4-person50" type="given" value="Mary"/> <interp inst="t17540116-4-person50" type="gender" value="female"/> <interp inst="t17540116-4-person50" type="age" value="unknown"/> </persName> . I live in <placeName id="t17540116-4-crimeloc23">Scotland-Yard</placeName> <interp inst="t17540116-4-crimeloc23" type="placeName" value="Scotland-Yard"/> <interp inst="t17540116-4-crimeloc23" type="type" value="crimeLocation"/> <join result="offencePlace" targOrder="Y" targets="t17540116-4-off20 t17540116-4-crimeloc23"/> <interp inst="t17540116-4-crimeloc23" type="crimeLocation" value="Scotland-Yard"/>, the prisoner was with me six or seven weeks before she went away, then she took the key with her; the room was let to her, her husband she said was in Jamaica. I opened the door, and missed the goods mentioned. She was taken in less than a fortnight; when taken she told me where the things were pawned and sold, and that she took them in the morning before I was up; we went as she directed, and found some of the things. [produced in court, and deposed to.]</p> <p> <persName id="t17540116-4-person51"> Thomas Stevens <interp inst="t17540116-4-person51" type="surname" value="Stevens"/> <interp inst="t17540116-4-person51" type="given" value="Thomas"/> <interp inst="t17540116-4-person51" type="gender" value="male"/> <interp inst="t17540116-4-person51" type="age" value="unknown"/> </persName> . I belong to Mr. Perry in Castle-street, Leicester Fields, he is a pawnbroker, the prisoner pawned these two pillows and tea-kettle with me.</p> <p>Prisoner\'s defence.</p> <p>I do not know the last evidence. I am not guilty.</p> <p> <rs id="t17540116-4-verdict24" type="verdictDescription"> <interp inst="t17540116-4-verdict24" type="verdictCategory" value="guilty"/> <interp inst="t17540116-4-verdict24" type="verdictSubcategory" value="guiltyNoDetail"/> <interp inst="t17540116-4-verdict24" type="plea" value="notGuilty"/> Guilty </rs>.</p> <p> <rs id="t17540116-4-punish25" type="punishmentDescription"> <interp inst="t17540116-4-punish25" type="punishmentCategory" value="transport"/> <interp inst="t17540116-4-punish25" type="punishmentSubcategory" value="transportNoDetail"/> <join result="defendantPunishment" targOrder="Y" targets="t17540116-4-defend46 t17540116-4-punish25"/> <note>[Transportation. See summary.]</note> </rs> </p> </div1>',
         'html': '<div> <p>83. (M.) Elizabeth wife of Joseph Kempster , was indicted for stealing one feather-bed, value 14 s. one bolster 2 s. one pillow, one flaxen sheet, one copper tea-kettle, one brass candlestick, one pair of bellows, the goods of Mary Kennedy , widow , in a certain lodging let by contract, &amp;c. December 23 . ++</p> <p> Mary Kennedy . I live in Scotland-Yard , the prisoner was with me six or seven weeks before she went away, then she took the key with her; the room was let to her, her husband she said was in Jamaica. I opened the door, and missed the goods mentioned. She was taken in less than a fortnight; when taken she told me where the things were pawned and sold, and that she took them in the morning before I was up; we went as she directed, and found some of the things. [produced in court, and deposed to.]</p> <p> Thomas Stevens . I belong to Mr. Perry in Castle-street, Leicester Fields, he is a pawnbroker, the prisoner pawned these two pillows and tea-kettle with me.</p> <p>Prisoner\'s defence.</p> <p>I do not know the last evidence. I am not guilty.</p> <p> Guilty .</p> <p> <span class="punishment-summary-link-container"><a class="punishment-summary-link" href="s17540116-1">[Transportation. See summary.]</a></span> </p> </div>',
         'text': "83. (M.) Elizabeth wife of Joseph Kempster , was indicted for stealing one feather-bed, value 14 s. one bolster 2 s. one pillow, one flaxen sheet, one copper tea-kettle, one brass candlestick, one pair of bellows, the goods of Mary Kennedy , widow , in a certain lodging let by contract, &c. December 23 . ++ Mary Kennedy . I live in Scotland-Yard , the prisoner was with me six or seven weeks before she went away, then she took the key with her; the room was let to her, her husband she said was in Jamaica. I opened the door, and missed the goods mentioned. She was taken in less than a fortnight; when taken she told me where the things were pawned and sold, and that she took them in the morning before I was up; we went as she directed, and found some of the things. [produced in court, and deposed to.] Thomas Stevens . I belong to Mr. Perry in Castle-street, Leicester Fields, he is a pawnbroker, the prisoner pawned these two pillows and tea-kettle with me. Prisoner's defence. I do not know the last evidence. I am not guilty. Guilty . [Transportation. See summary.]",
         'image_rights': ['This image is reproduced by permission of Harvard University Library from the microfilm, "The Old Bailey Proceedings", (Harvester Microform, a former imprint of the Gale Group, 1983). Commercial use is prohibited without permission of the owner of the original.']}}]}}



Notice that if you click the .gif image to see the original text, it conatins a funny looking "S" letter, which is known as the "long S" - an archaic version of the S used in english. 


```python
response.json()['hits']['hits'][0]['_source']['xml']
```




    '<div1 type="trialAccount" id="t17540116-4"> <interp inst="t17540116-4" type="prevdiv" value="t17540116-3" divtype="trialAccount"/> <interp inst="t17540116-4" type="nextdiv" value="t17540116-5" divtype="trialAccount"/> <interp inst="t17540116-4" type="div0" value="17540116" divtype="sessionsPaper"/> <interp inst="t17540116-4" type="assocrec" value="ar_4437_48007" title="Document. 00 1754. London Metropolitan Archives MJ/SP/1754/01/041."/> <interp inst="t17540116-4" type="collection" value="BAILEY"/> <interp inst="t17540116-4" type="year" value="1754"/> <interp inst="t17540116-4" type="uri" value="sessionsPapers/17540116"/> <interp inst="t17540116-4" type="date" value="17540116"/> <xptr imgpath="ob/1750s/175401160003.gif" imgtitle="Proceedings of the Old Bailey, 16th January ." imgrights="This image is reproduced by permission of Harvard University Library from the microfilm, &quot;The Old Bailey Proceedings&quot;, (Harvester Microform, a former imprint of the Gale Group, 1983). Commercial use is prohibited without permission of the owner of the original." type="pageFacsimile" value="preceding" doc="175401160003"/> <join result="criminalCharge" id="t17540116-4-off20-c41" targOrder="Y" targets="t17540116-4-defend46 t17540116-4-off20 t17540116-4-verdict24"/> <p>83. (M.) <interp inst="t17540116-4-defend46" type="digipan" value="Elizabeth Joseph Kempster" title="2 records"/> <persName id="t17540116-4-defend46" type="defendantName"> Elizabeth <rs id="t17540116-4-deflabel19" type="occupation">wife</rs> <interp inst="t17540116-4-deflabel19" type="defendantNameLabel" value="wife"/> <join result="persNameOccupation" targOrder="Y" targets="t17540116-4-defend46 t17540116-4-deflabel19"/> of <persName id="t17540116-4-person47"> Joseph Kempster <interp inst="t17540116-4-person47" type="surname" value="Kempster"/> <interp inst="t17540116-4-person47" type="given" value="Joseph"/> <interp inst="t17540116-4-person47" type="gender" value="male"/> <interp inst="t17540116-4-person47" type="age" value="unknown"/> </persName> <interp inst="t17540116-4-defend46" type="surname" value="Kempster"/> <interp inst="t17540116-4-defend46" type="given" value="Elizabeth"/> <interp inst="t17540116-4-defend46" type="gender" value="female"/> </persName> , was indicted for <rs id="t17540116-4-off20" type="offenceDescription"> <interp inst="t17540116-4-off20" type="offenceCategory" value="theft"/> <interp inst="t17540116-4-off20" type="offenceSubcategory" value="theftFromPlace"/> stealing one feather-bed, value 14 s. one bolster 2 s. one pillow, one flaxen sheet, one copper tea-kettle, one brass candlestick, one pair of bellows, the goods of <interp inst="t17540116-4-victim49" type="victims-institution" value="no"/> <interp inst="t17540116-4-victim49" type="victims-hisco-code" value="0"/> <persName id="t17540116-4-victim49" type="victimName"> Mary Kennedy <interp inst="t17540116-4-victim49" type="surname" value="Kennedy"/> <interp inst="t17540116-4-victim49" type="given" value="Mary"/> <interp inst="t17540116-4-victim49" type="gender" value="female"/> <join result="offenceVictim" targOrder="Y" targets="t17540116-4-off20 t17540116-4-victim49"/> </persName> , <rs id="t17540116-4-viclabel21" type="occupation">widow</rs> <interp inst="t17540116-4-viclabel21" type="victimNameLabel" value="widow"/> <join result="persNameOccupation" targOrder="Y" targets="t17540116-4-victim49 t17540116-4-viclabel21"/>, in a certain lodging </rs> let by contract, &amp;c. <rs id="t17540116-4-cd22" type="crimeDate">December 23</rs> <join result="offenceCrimeDate" targOrder="Y" targets="t17540116-4-off20 t17540116-4-cd22"/>. ++</p> <p> <persName id="t17540116-4-person50"> Mary Kennedy <interp inst="t17540116-4-person50" type="surname" value="Kennedy"/> <interp inst="t17540116-4-person50" type="given" value="Mary"/> <interp inst="t17540116-4-person50" type="gender" value="female"/> <interp inst="t17540116-4-person50" type="age" value="unknown"/> </persName> . I live in <placeName id="t17540116-4-crimeloc23">Scotland-Yard</placeName> <interp inst="t17540116-4-crimeloc23" type="placeName" value="Scotland-Yard"/> <interp inst="t17540116-4-crimeloc23" type="type" value="crimeLocation"/> <join result="offencePlace" targOrder="Y" targets="t17540116-4-off20 t17540116-4-crimeloc23"/> <interp inst="t17540116-4-crimeloc23" type="crimeLocation" value="Scotland-Yard"/>, the prisoner was with me six or seven weeks before she went away, then she took the key with her; the room was let to her, her husband she said was in Jamaica. I opened the door, and missed the goods mentioned. She was taken in less than a fortnight; when taken she told me where the things were pawned and sold, and that she took them in the morning before I was up; we went as she directed, and found some of the things. [produced in court, and deposed to.]</p> <p> <persName id="t17540116-4-person51"> Thomas Stevens <interp inst="t17540116-4-person51" type="surname" value="Stevens"/> <interp inst="t17540116-4-person51" type="given" value="Thomas"/> <interp inst="t17540116-4-person51" type="gender" value="male"/> <interp inst="t17540116-4-person51" type="age" value="unknown"/> </persName> . I belong to Mr. Perry in Castle-street, Leicester Fields, he is a pawnbroker, the prisoner pawned these two pillows and tea-kettle with me.</p> <p>Prisoner\'s defence.</p> <p>I do not know the last evidence. I am not guilty.</p> <p> <rs id="t17540116-4-verdict24" type="verdictDescription"> <interp inst="t17540116-4-verdict24" type="verdictCategory" value="guilty"/> <interp inst="t17540116-4-verdict24" type="verdictSubcategory" value="guiltyNoDetail"/> <interp inst="t17540116-4-verdict24" type="plea" value="notGuilty"/> Guilty </rs>.</p> <p> <rs id="t17540116-4-punish25" type="punishmentDescription"> <interp inst="t17540116-4-punish25" type="punishmentCategory" value="transport"/> <interp inst="t17540116-4-punish25" type="punishmentSubcategory" value="transportNoDetail"/> <join result="defendantPunishment" targOrder="Y" targets="t17540116-4-defend46 t17540116-4-punish25"/> <note>[Transportation. See summary.]</note> </rs> </p> </div1>'



We can save just the XML portion in a local file:


```python
from pathlib import Path

directory = Path("data/old-bailey")
file_path = directory / f"old-bailey-{fifth_trial_id}.xml"

directory.mkdir(parents=True, exist_ok=True)

with open(file_path, 'w') as file:
    file.write(response.json()['hits']['hits'][0]['_source']['xml'])
```

Now we'll get trial `t17031013-13` specifically, for the examples below, concerning a man named "Davis":


```python
davis_trial_id = 't17031013-13'

url = f'https://www.dhi.ac.uk/api/data/oldbailey_record_single?idkey={davis_trial_id}'

response = requests.get(url, 
                        headers=headers)

response.raise_for_status()

trial_xml = response.json()['hits']['hits'][0]['_source']['xml']

# Save to file
with open(f'data/old-bailey/old-bailey-{davis_trial_id}.xml', 'w', encoding='utf-8') as file:
    file.write(trial_xml)
```


```python
response.json()
```




    {'took': 1,
     'timed_out': False,
     '_shards': {'total': 20, 'successful': 20, 'skipped': 0, 'failed': 0},
     'hits': {'total': 1,
      'max_score': 10.208518,
      'hits': [{'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMK2rPld_y58Rz-bk',
        '_score': 10.208518,
        '_source': {'metadata': '<table xmlns:dhids="https://www.dhi.ac.uk/data/" class="table small"><tbody><tr><th scope="row">Text type</th><td>Trial account</td></tr><tr><th scope="row">Defendants</th><td>Samuel Davis</td></tr><tr><th scope="row">Offences</th><td><a href="../about/crimes#theft">Theft</a> &gt; <a href="../about/crimes#grandlarceny">Grand larceny</a></td></tr><tr><th scope="row">Session Date</th><td><a href="17031013#t17031013-13">13th October 1703</a></td></tr><tr><th scope="row">Reference Number</th><td>t17031013-13</td></tr><tr><th scope="row">Verdicts</th><td><a href="../about/verdicts#guilty">Guilty</a></td></tr><tr><th scope="row">Punishments</th><td><a href="../about/punishment#miscellaneouspunishments">Miscellaneous Punishment</a> &gt; <a href="../about/punishment#branding">Branding on cheek</a></td></tr><tr><th scope="row">Navigation</th><td><a href="t17031013-12">&lt; Previous section (Trial account)</a> | <a href="t17031013-14">Next section (Trial account) &gt;</a></td></tr></tbody></table>',
         'images': ['https://www.dhi.ac.uk/san/ob/1700s/17031013002.gif'],
         'exturl': 'https://www.oldbaileyonline.org/browse.jsp?div=t17031013-13',
         'citation': '<span><i>Old Bailey Proceedings Online</i> (www.oldbaileyonline.org, version 9.0) October 1703. Trial of Samuel Davis (t17031013-13).</span>',
         'project': 'oldbailey',
         'collection': 'record',
         'title': 'Samuel Davis. Theft; grand larceny (to 1827). 13th October 1703.',
         'image_titles': ['Proceedings of the Old Bailey, 13th October .'],
         'idkey': 't17031013-13',
         'navigation': '<span><a href="t17031013-12">&lt; Previous section (Trial account)</a> | <a href="t17031013-14">Next section (Trial account) &gt;</a></span>',
         'xml': '<div1 type="trialAccount" id="t17031013-13"> <interp inst="t17031013-13" type="prevdiv" value="t17031013-12" divtype="trialAccount"/> <interp inst="t17031013-13" type="nextdiv" value="t17031013-14" divtype="trialAccount"/> <interp inst="t17031013-13" type="div0" value="17031013" divtype="sessionsPaper"/> <interp inst="t17031013-13" type="collection" value="BAILEY"/> <interp inst="t17031013-13" type="year" value="1703"/> <interp inst="t17031013-13" type="uri" value="sessionsPapers/17031013"/> <interp inst="t17031013-13" type="date" value="17031013"/> <xptr imgpath="ob/1700s/17031013002.gif" imgtitle="Proceedings of the Old Bailey, 13th October ." imgrights="This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original." type="pageFacsimile" value="preceding" doc="17031013002"/> <join result="criminalCharge" id="t17031013-13-off60-c52" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-off60 t17031013-13-verdict64"/> <p> <persName id="t17031013-13-defend52" type="defendantName"> Samuel Davis <interp inst="t17031013-13-defend52" type="surname" value="Davis"/> <interp inst="t17031013-13-defend52" type="given" value="Samuel"/> <interp inst="t17031013-13-defend52" type="gender" value="male"/> </persName> , of the Parish of <placeName id="t17031013-13-defloc59">St. James Westminster</placeName> <interp inst="t17031013-13-defloc59" type="placeName" value="St. James Westminster"/> <interp inst="t17031013-13-defloc59" type="type" value="defendantHome"/> <join result="persNamePlace" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-defloc59"/> <interp inst="t17031013-13-defloc59" type="defendantNameHome" value="St. James Westminster"/>, was indicted for <rs id="t17031013-13-off60" type="offenceDescription"> <interp inst="t17031013-13-off60" type="offenceCategory" value="theft"/> <interp inst="t17031013-13-off60" type="offenceSubcategory" value="grandLarceny"/> feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. </rs> the Goods of the Honourable <interp inst="t17031013-13-victim53" type="victims-institution" value="no"/> <interp inst="t17031013-13-victim53" type="victims-occupation" value="Lady"/> <interp inst="t17031013-13-victim53" type="victims-hisco-label" value="Aristocrat"/> <interp inst="t17031013-13-victim53" type="victims-hisco-code" value="1700"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-1" value="0.5"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-3" value="Upper (0.5-3)"/> <persName id="t17031013-13-victim53" type="victimName"> Catherine <rs id="t17031013-13-viclabel61" type="occupation">Lady</rs> <interp inst="t17031013-13-viclabel61" type="victimNameLabel" value="Lady"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-victim53 t17031013-13-viclabel61"/> Herbert <interp inst="t17031013-13-victim53" type="surname" value="Herbert"/> <interp inst="t17031013-13-victim53" type="given" value="Catherine"/> <interp inst="t17031013-13-victim53" type="gender" value="female"/> <join result="offenceVictim" targOrder="Y" targets="t17031013-13-off60 t17031013-13-victim53"/> </persName> , on the <rs id="t17031013-13-cd62" type="crimeDate">28th of July</rs> <join result="offenceCrimeDate" targOrder="Y" targets="t17031013-13-off60 t17031013-13-cd62"/> last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a <rs id="t17031013-13-deflabel63" type="occupation">Coachman</rs> <interp inst="t17031013-13-deflabel63" type="defendantNameLabel" value="Coachman"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-deflabel63"/> in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him <rs id="t17031013-13-verdict64" type="verdictDescription"> <interp inst="t17031013-13-verdict64" type="verdictCategory" value="guilty"/> <interp inst="t17031013-13-verdict64" type="verdictSubcategory" value="guiltyNoDetail"/> <interp inst="t17031013-13-verdict64" type="plea" value="notGuilty"/> guilty </rs>.</p> <p> <rs id="t17031013-13-punish65" type="punishmentDescription"> <interp inst="t17031013-13-punish65" type="punishmentCategory" value="miscPunish"/> <interp inst="t17031013-13-punish65" type="punishmentSubcategory" value="brandingOnCheek"/> <join result="defendantPunishment" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-punish65"/> <note>[Branding. See summary.]</note> </rs> </p> </div1>',
         'html': '<div> <p> Samuel Davis , of the Parish of St. James Westminster , was indicted for feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. the Goods of the Honourable Catherine Lady Herbert , on the 28th of July last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a Coachman in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him guilty .</p> <p> <span class="punishment-summary-link-container"><a class="punishment-summary-link" href="s17031013-1">[Branding. See summary.]</a></span> </p> </div>',
         'text': 'Samuel Davis , of the Parish of St. James Westminster , was indicted for feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. the Goods of the Honourable Catherine Lady Herbert , on the 28th of July last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a Coachman in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him guilty . [Branding. See summary.]',
         'image_rights': ['This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original.']}}]}}




```python
trial_xml
```




    '<div1 type="trialAccount" id="t17031013-13"> <interp inst="t17031013-13" type="prevdiv" value="t17031013-12" divtype="trialAccount"/> <interp inst="t17031013-13" type="nextdiv" value="t17031013-14" divtype="trialAccount"/> <interp inst="t17031013-13" type="div0" value="17031013" divtype="sessionsPaper"/> <interp inst="t17031013-13" type="collection" value="BAILEY"/> <interp inst="t17031013-13" type="year" value="1703"/> <interp inst="t17031013-13" type="uri" value="sessionsPapers/17031013"/> <interp inst="t17031013-13" type="date" value="17031013"/> <xptr imgpath="ob/1700s/17031013002.gif" imgtitle="Proceedings of the Old Bailey, 13th October ." imgrights="This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original." type="pageFacsimile" value="preceding" doc="17031013002"/> <join result="criminalCharge" id="t17031013-13-off60-c52" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-off60 t17031013-13-verdict64"/> <p> <persName id="t17031013-13-defend52" type="defendantName"> Samuel Davis <interp inst="t17031013-13-defend52" type="surname" value="Davis"/> <interp inst="t17031013-13-defend52" type="given" value="Samuel"/> <interp inst="t17031013-13-defend52" type="gender" value="male"/> </persName> , of the Parish of <placeName id="t17031013-13-defloc59">St. James Westminster</placeName> <interp inst="t17031013-13-defloc59" type="placeName" value="St. James Westminster"/> <interp inst="t17031013-13-defloc59" type="type" value="defendantHome"/> <join result="persNamePlace" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-defloc59"/> <interp inst="t17031013-13-defloc59" type="defendantNameHome" value="St. James Westminster"/>, was indicted for <rs id="t17031013-13-off60" type="offenceDescription"> <interp inst="t17031013-13-off60" type="offenceCategory" value="theft"/> <interp inst="t17031013-13-off60" type="offenceSubcategory" value="grandLarceny"/> feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. </rs> the Goods of the Honourable <interp inst="t17031013-13-victim53" type="victims-institution" value="no"/> <interp inst="t17031013-13-victim53" type="victims-occupation" value="Lady"/> <interp inst="t17031013-13-victim53" type="victims-hisco-label" value="Aristocrat"/> <interp inst="t17031013-13-victim53" type="victims-hisco-code" value="1700"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-1" value="0.5"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-3" value="Upper (0.5-3)"/> <persName id="t17031013-13-victim53" type="victimName"> Catherine <rs id="t17031013-13-viclabel61" type="occupation">Lady</rs> <interp inst="t17031013-13-viclabel61" type="victimNameLabel" value="Lady"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-victim53 t17031013-13-viclabel61"/> Herbert <interp inst="t17031013-13-victim53" type="surname" value="Herbert"/> <interp inst="t17031013-13-victim53" type="given" value="Catherine"/> <interp inst="t17031013-13-victim53" type="gender" value="female"/> <join result="offenceVictim" targOrder="Y" targets="t17031013-13-off60 t17031013-13-victim53"/> </persName> , on the <rs id="t17031013-13-cd62" type="crimeDate">28th of July</rs> <join result="offenceCrimeDate" targOrder="Y" targets="t17031013-13-off60 t17031013-13-cd62"/> last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a <rs id="t17031013-13-deflabel63" type="occupation">Coachman</rs> <interp inst="t17031013-13-deflabel63" type="defendantNameLabel" value="Coachman"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-deflabel63"/> in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him <rs id="t17031013-13-verdict64" type="verdictDescription"> <interp inst="t17031013-13-verdict64" type="verdictCategory" value="guilty"/> <interp inst="t17031013-13-verdict64" type="verdictSubcategory" value="guiltyNoDetail"/> <interp inst="t17031013-13-verdict64" type="plea" value="notGuilty"/> guilty </rs>.</p> <p> <rs id="t17031013-13-punish65" type="punishmentDescription"> <interp inst="t17031013-13-punish65" type="punishmentCategory" value="miscPunish"/> <interp inst="t17031013-13-punish65" type="punishmentSubcategory" value="brandingOnCheek"/> <join result="defendantPunishment" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-punish65"/> <note>[Branding. See summary.]</note> </rs> </p> </div1>'



### Challenge: Scraping all trials from 1754 - 1756

Now we have extracted trial IDs and trial XML trees using `requests.get(some_url)`, so we can iterate through each ID in a of trials (use `doc_ids` from above for the list of IDs). You can choose how many trials you want to save--maybe 50 to start?


```python
len(doc_ids)
```




    1312




```python
doc_ids[0]
```




    'f17540116-1'




```python
for doc_id in doc_ids[:50]:
    url = 'https://www.dhi.ac.uk/api/data/oldbailey_record_single?idkey={}'.format(doc_id)
    
    try:
       
        response = requests.get(url, 
                                headers=headers)  # Get the JSON with the User-Agent header
        response.raise_for_status() # Check if request was successful
        tree = response.json()
        
        # Extract the XML content
        xml_content = tree['hits']['hits'][0]['_source']['xml']
        
        # Save the file with UTF-8 encoding
        file_path = 'data/old-bailey/old-bailey-{}.xml'.format(doc_id)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(xml_content)
            
        print("Successfully saved: {}".format(doc_id))

    except Exception as e:
        print("Failed to download {}: {}".format(doc_id, e))
    
    # 0.1-second pause to be a polite crawler
    time.sleep(0.1)
```

    Successfully saved: f17540116-1
    Successfully saved: t17540116-1
    Successfully saved: t17540116-2
    Successfully saved: t17540116-3
    Successfully saved: t17540116-4
    Successfully saved: t17540116-5
    Successfully saved: t17540116-6
    Successfully saved: t17540116-7
    Successfully saved: t17540116-8
    Successfully saved: t17540116-9
    Successfully saved: t17540116-10
    Successfully saved: t17540116-11
    Successfully saved: t17540116-12
    Successfully saved: t17540116-13
    Successfully saved: t17540116-14
    Successfully saved: t17540116-15
    Successfully saved: t17540116-16
    Successfully saved: t17540116-17
    Successfully saved: t17540116-18
    Successfully saved: t17540116-19
    Successfully saved: t17540116-20
    Successfully saved: t17540116-21
    Successfully saved: t17540116-22
    Successfully saved: t17540116-23
    Successfully saved: t17540116-24
    Successfully saved: t17540116-25
    Successfully saved: t17540116-26
    Successfully saved: t17540116-27
    Successfully saved: t17540116-28
    Successfully saved: t17540116-29
    Successfully saved: t17540116-30
    Successfully saved: t17540116-31
    Successfully saved: t17540116-32
    Successfully saved: t17540116-33
    Successfully saved: t17540116-34
    Successfully saved: t17540116-34
    Successfully saved: t17540116-35
    Successfully saved: t17540116-36
    Successfully saved: t17540116-37
    Successfully saved: t17540116-38
    Successfully saved: t17540116-39
    Successfully saved: t17540116-40
    Successfully saved: t17540116-41
    Successfully saved: t17540116-42
    Successfully saved: t17540116-43
    Successfully saved: t17540116-44
    Successfully saved: t17540116-45
    Successfully saved: t17540116-46
    Successfully saved: t17540116-47
    Successfully saved: t17540116-48


You can check if you saved the XML files by executing the cell below


```python
import glob
from pathlib import Path

directory = Path("data/old-bailey")

xml_files = list(directory.glob("*.xml"))
xml_files
```




    [PosixPath('data/old-bailey/old-bailey-t17540116-34.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-32.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-41.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-30.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-1.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-2.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-18.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-13.xml'),
     PosixPath('data/old-bailey/old-bailey-t17031013-13.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-38.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-7.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-33.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-12.xml'),
     PosixPath('data/old-bailey/old-bailey-f17540116-1.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-21.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-43.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-27.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-47.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-16.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-29.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-26.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-23.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-28.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-24.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-9.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-4.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-15.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-35.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-46.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-5.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-8.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-10.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-42.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-37.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-22.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-31.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-20.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-39.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-11.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-36.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-40.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-48.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-19.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-44.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-17.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-45.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-14.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-3.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-25.xml'),
     PosixPath('data/old-bailey/old-bailey-t17540116-6.xml')]



This cell will show you the XML file.


```python
file_path = xml_files[5]

with open(file_path, "r", encoding="utf-8") as f:
    xml_content = f.read()

print(xml_content[:2000])  # print first 2000 characters to avoid flooding
```

    <div1 type="trialAccount" id="t17540116-2"> <interp inst="t17540116-2" type="prevdiv" value="t17540116-1" divtype="trialAccount"/> <interp inst="t17540116-2" type="nextdiv" value="t17540116-3" divtype="trialAccount"/> <interp inst="t17540116-2" type="div0" value="17540116" divtype="sessionsPaper"/> <interp inst="t17540116-2" type="collection" value="BAILEY"/> <interp inst="t17540116-2" type="year" value="1754"/> <interp inst="t17540116-2" type="uri" value="sessionsPapers/17540116"/> <interp inst="t17540116-2" type="date" value="17540116"/> <xptr imgpath="ob/1750s/175401160002.gif" imgtitle="Proceedings of the Old Bailey, 16th January ." imgrights="This image is reproduced by permission of Harvard University Library from the microfilm, &quot;The Old Bailey Proceedings&quot;, (Harvester Microform, a former imprint of the Gale Group, 1983). Commercial use is prohibited without permission of the owner of the original." type="pageFacsimile" value="preceding" doc="175401160002"/> <join result="criminalCharge" id="t17540116-2-off7-c31" targOrder="Y" targets="t17540116-2-defend32 t17540116-2-off7 t17540116-2-verdict9"/> <join result="criminalCharge" id="t17540116-2-off7-c32" targOrder="Y" targets="t17540116-2-defend34 t17540116-2-off7 t17540116-2-verdict9"/> <p>81. (M.) <persName id="t17540116-2-defend32" type="defendantName"> <persName id="t17540116-2-defend34" type="defendantName"> Peter Foreman <interp inst="t17540116-2-defend34" type="surname" value="Foreman"/> <interp inst="t17540116-2-defend34" type="given" value="Peter"/> <interp inst="t17540116-2-defend34" type="gender" value="male"/> </persName> and Mary his <rs id="t17540116-2-deflabel6" type="occupation">wife</rs> <interp inst="t17540116-2-deflabel6" type="defendantNameLabel" value="wife"/> <join result="persNameOccupation" targOrder="Y" targets="t17540116-2-defend32 t17540116-2-deflabel6"/> <interp inst="t17540116-2-defend32" type="surname" value="Foreman"/> <interp inst="t17540116-2-defend32" type="given" value


## XML Syntax

First, we'll go over the syntax of a XML file. The basic unit of XML code is called an "element" or "node" and has a start and ending tag. The tags for each element look something like this:

 `<exampletag>some text</exampletag>`  

Run the next cell to look at the XML file of one of the cases from the OldBailey API!


```python
# Samuel Davis trial from above
davis_trial_id = "t17031013-13"

example = requests.get(
    "https://www.dhi.ac.uk/api/data/oldbailey_record_single",
    params={"idkey": davis_trial_id},
    headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
).json()

example  
```




    {'took': 1,
     'timed_out': False,
     '_shards': {'total': 20, 'successful': 20, 'skipped': 0, 'failed': 0},
     'hits': {'total': 1,
      'max_score': 10.208518,
      'hits': [{'_index': 'dhids_oldbailey_record',
        '_type': 'doc',
        '_id': 'AYxEMK2rPld_y58Rz-bk',
        '_score': 10.208518,
        '_source': {'metadata': '<table xmlns:dhids="https://www.dhi.ac.uk/data/" class="table small"><tbody><tr><th scope="row">Text type</th><td>Trial account</td></tr><tr><th scope="row">Defendants</th><td>Samuel Davis</td></tr><tr><th scope="row">Offences</th><td><a href="../about/crimes#theft">Theft</a> &gt; <a href="../about/crimes#grandlarceny">Grand larceny</a></td></tr><tr><th scope="row">Session Date</th><td><a href="17031013#t17031013-13">13th October 1703</a></td></tr><tr><th scope="row">Reference Number</th><td>t17031013-13</td></tr><tr><th scope="row">Verdicts</th><td><a href="../about/verdicts#guilty">Guilty</a></td></tr><tr><th scope="row">Punishments</th><td><a href="../about/punishment#miscellaneouspunishments">Miscellaneous Punishment</a> &gt; <a href="../about/punishment#branding">Branding on cheek</a></td></tr><tr><th scope="row">Navigation</th><td><a href="t17031013-12">&lt; Previous section (Trial account)</a> | <a href="t17031013-14">Next section (Trial account) &gt;</a></td></tr></tbody></table>',
         'images': ['https://www.dhi.ac.uk/san/ob/1700s/17031013002.gif'],
         'exturl': 'https://www.oldbaileyonline.org/browse.jsp?div=t17031013-13',
         'citation': '<span><i>Old Bailey Proceedings Online</i> (www.oldbaileyonline.org, version 9.0) October 1703. Trial of Samuel Davis (t17031013-13).</span>',
         'project': 'oldbailey',
         'collection': 'record',
         'title': 'Samuel Davis. Theft; grand larceny (to 1827). 13th October 1703.',
         'image_titles': ['Proceedings of the Old Bailey, 13th October .'],
         'idkey': 't17031013-13',
         'navigation': '<span><a href="t17031013-12">&lt; Previous section (Trial account)</a> | <a href="t17031013-14">Next section (Trial account) &gt;</a></span>',
         'xml': '<div1 type="trialAccount" id="t17031013-13"> <interp inst="t17031013-13" type="prevdiv" value="t17031013-12" divtype="trialAccount"/> <interp inst="t17031013-13" type="nextdiv" value="t17031013-14" divtype="trialAccount"/> <interp inst="t17031013-13" type="div0" value="17031013" divtype="sessionsPaper"/> <interp inst="t17031013-13" type="collection" value="BAILEY"/> <interp inst="t17031013-13" type="year" value="1703"/> <interp inst="t17031013-13" type="uri" value="sessionsPapers/17031013"/> <interp inst="t17031013-13" type="date" value="17031013"/> <xptr imgpath="ob/1700s/17031013002.gif" imgtitle="Proceedings of the Old Bailey, 13th October ." imgrights="This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original." type="pageFacsimile" value="preceding" doc="17031013002"/> <join result="criminalCharge" id="t17031013-13-off60-c52" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-off60 t17031013-13-verdict64"/> <p> <persName id="t17031013-13-defend52" type="defendantName"> Samuel Davis <interp inst="t17031013-13-defend52" type="surname" value="Davis"/> <interp inst="t17031013-13-defend52" type="given" value="Samuel"/> <interp inst="t17031013-13-defend52" type="gender" value="male"/> </persName> , of the Parish of <placeName id="t17031013-13-defloc59">St. James Westminster</placeName> <interp inst="t17031013-13-defloc59" type="placeName" value="St. James Westminster"/> <interp inst="t17031013-13-defloc59" type="type" value="defendantHome"/> <join result="persNamePlace" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-defloc59"/> <interp inst="t17031013-13-defloc59" type="defendantNameHome" value="St. James Westminster"/>, was indicted for <rs id="t17031013-13-off60" type="offenceDescription"> <interp inst="t17031013-13-off60" type="offenceCategory" value="theft"/> <interp inst="t17031013-13-off60" type="offenceSubcategory" value="grandLarceny"/> feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. </rs> the Goods of the Honourable <interp inst="t17031013-13-victim53" type="victims-institution" value="no"/> <interp inst="t17031013-13-victim53" type="victims-occupation" value="Lady"/> <interp inst="t17031013-13-victim53" type="victims-hisco-label" value="Aristocrat"/> <interp inst="t17031013-13-victim53" type="victims-hisco-code" value="1700"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-1" value="0.5"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-3" value="Upper (0.5-3)"/> <persName id="t17031013-13-victim53" type="victimName"> Catherine <rs id="t17031013-13-viclabel61" type="occupation">Lady</rs> <interp inst="t17031013-13-viclabel61" type="victimNameLabel" value="Lady"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-victim53 t17031013-13-viclabel61"/> Herbert <interp inst="t17031013-13-victim53" type="surname" value="Herbert"/> <interp inst="t17031013-13-victim53" type="given" value="Catherine"/> <interp inst="t17031013-13-victim53" type="gender" value="female"/> <join result="offenceVictim" targOrder="Y" targets="t17031013-13-off60 t17031013-13-victim53"/> </persName> , on the <rs id="t17031013-13-cd62" type="crimeDate">28th of July</rs> <join result="offenceCrimeDate" targOrder="Y" targets="t17031013-13-off60 t17031013-13-cd62"/> last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a <rs id="t17031013-13-deflabel63" type="occupation">Coachman</rs> <interp inst="t17031013-13-deflabel63" type="defendantNameLabel" value="Coachman"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-deflabel63"/> in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him <rs id="t17031013-13-verdict64" type="verdictDescription"> <interp inst="t17031013-13-verdict64" type="verdictCategory" value="guilty"/> <interp inst="t17031013-13-verdict64" type="verdictSubcategory" value="guiltyNoDetail"/> <interp inst="t17031013-13-verdict64" type="plea" value="notGuilty"/> guilty </rs>.</p> <p> <rs id="t17031013-13-punish65" type="punishmentDescription"> <interp inst="t17031013-13-punish65" type="punishmentCategory" value="miscPunish"/> <interp inst="t17031013-13-punish65" type="punishmentSubcategory" value="brandingOnCheek"/> <join result="defendantPunishment" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-punish65"/> <note>[Branding. See summary.]</note> </rs> </p> </div1>',
         'html': '<div> <p> Samuel Davis , of the Parish of St. James Westminster , was indicted for feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. the Goods of the Honourable Catherine Lady Herbert , on the 28th of July last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a Coachman in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him guilty .</p> <p> <span class="punishment-summary-link-container"><a class="punishment-summary-link" href="s17031013-1">[Branding. See summary.]</a></span> </p> </div>',
         'text': 'Samuel Davis , of the Parish of St. James Westminster , was indicted for feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. the Goods of the Honourable Catherine Lady Herbert , on the 28th of July last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a Coachman in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him guilty . [Branding. See summary.]',
         'image_rights': ['This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original.']}}]}}



We want to parse XML, so let's separate that out--the code above that saved files locally did that, as you can see.

You can see that the key 'xml' is shortly after the 'idkey' that we used when we made the list of trial records above. Let's inspect the XML from Samuel Davis's trial.


```python
# get just the xml from the Samuel Davis trial account
xml_data = example['hits']['hits'][0]['_source']['xml']
xml_data
```




    '<div1 type="trialAccount" id="t17031013-13"> <interp inst="t17031013-13" type="prevdiv" value="t17031013-12" divtype="trialAccount"/> <interp inst="t17031013-13" type="nextdiv" value="t17031013-14" divtype="trialAccount"/> <interp inst="t17031013-13" type="div0" value="17031013" divtype="sessionsPaper"/> <interp inst="t17031013-13" type="collection" value="BAILEY"/> <interp inst="t17031013-13" type="year" value="1703"/> <interp inst="t17031013-13" type="uri" value="sessionsPapers/17031013"/> <interp inst="t17031013-13" type="date" value="17031013"/> <xptr imgpath="ob/1700s/17031013002.gif" imgtitle="Proceedings of the Old Bailey, 13th October ." imgrights="This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original." type="pageFacsimile" value="preceding" doc="17031013002"/> <join result="criminalCharge" id="t17031013-13-off60-c52" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-off60 t17031013-13-verdict64"/> <p> <persName id="t17031013-13-defend52" type="defendantName"> Samuel Davis <interp inst="t17031013-13-defend52" type="surname" value="Davis"/> <interp inst="t17031013-13-defend52" type="given" value="Samuel"/> <interp inst="t17031013-13-defend52" type="gender" value="male"/> </persName> , of the Parish of <placeName id="t17031013-13-defloc59">St. James Westminster</placeName> <interp inst="t17031013-13-defloc59" type="placeName" value="St. James Westminster"/> <interp inst="t17031013-13-defloc59" type="type" value="defendantHome"/> <join result="persNamePlace" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-defloc59"/> <interp inst="t17031013-13-defloc59" type="defendantNameHome" value="St. James Westminster"/>, was indicted for <rs id="t17031013-13-off60" type="offenceDescription"> <interp inst="t17031013-13-off60" type="offenceCategory" value="theft"/> <interp inst="t17031013-13-off60" type="offenceSubcategory" value="grandLarceny"/> feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. </rs> the Goods of the Honourable <interp inst="t17031013-13-victim53" type="victims-institution" value="no"/> <interp inst="t17031013-13-victim53" type="victims-occupation" value="Lady"/> <interp inst="t17031013-13-victim53" type="victims-hisco-label" value="Aristocrat"/> <interp inst="t17031013-13-victim53" type="victims-hisco-code" value="1700"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-1" value="0.5"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-3" value="Upper (0.5-3)"/> <persName id="t17031013-13-victim53" type="victimName"> Catherine <rs id="t17031013-13-viclabel61" type="occupation">Lady</rs> <interp inst="t17031013-13-viclabel61" type="victimNameLabel" value="Lady"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-victim53 t17031013-13-viclabel61"/> Herbert <interp inst="t17031013-13-victim53" type="surname" value="Herbert"/> <interp inst="t17031013-13-victim53" type="given" value="Catherine"/> <interp inst="t17031013-13-victim53" type="gender" value="female"/> <join result="offenceVictim" targOrder="Y" targets="t17031013-13-off60 t17031013-13-victim53"/> </persName> , on the <rs id="t17031013-13-cd62" type="crimeDate">28th of July</rs> <join result="offenceCrimeDate" targOrder="Y" targets="t17031013-13-off60 t17031013-13-cd62"/> last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a <rs id="t17031013-13-deflabel63" type="occupation">Coachman</rs> <interp inst="t17031013-13-deflabel63" type="defendantNameLabel" value="Coachman"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-deflabel63"/> in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him <rs id="t17031013-13-verdict64" type="verdictDescription"> <interp inst="t17031013-13-verdict64" type="verdictCategory" value="guilty"/> <interp inst="t17031013-13-verdict64" type="verdictSubcategory" value="guiltyNoDetail"/> <interp inst="t17031013-13-verdict64" type="plea" value="notGuilty"/> guilty </rs>.</p> <p> <rs id="t17031013-13-punish65" type="punishmentDescription"> <interp inst="t17031013-13-punish65" type="punishmentCategory" value="miscPunish"/> <interp inst="t17031013-13-punish65" type="punishmentSubcategory" value="brandingOnCheek"/> <join result="defendantPunishment" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-punish65"/> <note>[Branding. See summary.]</note> </rs> </p> </div1>'



The `interp` tags at the beginning of the file are elements that don't have any plain text content. Note that elements may possibly be empty and not contain any text (i.e. `interp` elements mentioned earlier). If the element is empty, the tag may follow a format that looks similar to `<exampletag/>`, which is equivalent to `<exampletag></exampletag>`.

Elements may also contain other elements, which we call "children". Most children are indented, but the indents aren't necessary in XML and are used for clarity to show nesting. For example, if we go down to `<persName id="t17540116-4-defend46" type="defendantName">` , we see that the `rs` tag is a child of `persName`. We will explore about children in XML more in the next section. 

Lastly, elements may have attributes, which are in the format `<exampletag name_of_attribute="somevalue">`. Attributes are designed to store data related to a specific elements. Attributes **must** follow the quotes format (`name = "value"`). As you can tell, in this XML file, attributes are everywhere!


What was the verdict of this case? Was there a punsihment and if so, what was it?

List both and state whether you found it as plain text content or as an attribute.


## Using Beautiful Soup to parse XML

Now that we know what the syntax and structure of an XML file, let's figure out how to parse through one! We are going to load the same file from the second section and use [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to navigate through elements in this file. 

First, we need to import the file into a Beautiful Soup instance. 


```python
#from bs4 import BeautifulSoup # already imported 

xml_file = Path(f"data/old-bailey/old-bailey-{davis_trial_id}.xml")

xml_content = xml_file.read_text(encoding="utf-8")

soup = BeautifulSoup(xml_content, 
                    "xml")  # specify XML parser

print(soup.prettify())
```

    <?xml version="1.0" encoding="utf-8"?>
    <div1 id="t17031013-13" type="trialAccount">
     <interp divtype="trialAccount" inst="t17031013-13" type="prevdiv" value="t17031013-12"/>
     <interp divtype="trialAccount" inst="t17031013-13" type="nextdiv" value="t17031013-14"/>
     <interp divtype="sessionsPaper" inst="t17031013-13" type="div0" value="17031013"/>
     <interp inst="t17031013-13" type="collection" value="BAILEY"/>
     <interp inst="t17031013-13" type="year" value="1703"/>
     <interp inst="t17031013-13" type="uri" value="sessionsPapers/17031013"/>
     <interp inst="t17031013-13" type="date" value="17031013"/>
     <xptr doc="17031013002" imgpath="ob/1700s/17031013002.gif" imgrights="This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original." imgtitle="Proceedings of the Old Bailey, 13th October ." type="pageFacsimile" value="preceding"/>
     <join id="t17031013-13-off60-c52" result="criminalCharge" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-off60 t17031013-13-verdict64"/>
     <p>
      <persName id="t17031013-13-defend52" type="defendantName">
       Samuel Davis
       <interp inst="t17031013-13-defend52" type="surname" value="Davis"/>
       <interp inst="t17031013-13-defend52" type="given" value="Samuel"/>
       <interp inst="t17031013-13-defend52" type="gender" value="male"/>
      </persName>
      , of the Parish of
      <placeName id="t17031013-13-defloc59">
       St. James Westminster
      </placeName>
      <interp inst="t17031013-13-defloc59" type="placeName" value="St. James Westminster"/>
      <interp inst="t17031013-13-defloc59" type="type" value="defendantHome"/>
      <join result="persNamePlace" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-defloc59"/>
      <interp inst="t17031013-13-defloc59" type="defendantNameHome" value="St. James Westminster"/>
      , was indicted for
      <rs id="t17031013-13-off60" type="offenceDescription">
       <interp inst="t17031013-13-off60" type="offenceCategory" value="theft"/>
       <interp inst="t17031013-13-off60" type="offenceSubcategory" value="grandLarceny"/>
       feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l.
      </rs>
      the Goods of the Honourable
      <interp inst="t17031013-13-victim53" type="victims-institution" value="no"/>
      <interp inst="t17031013-13-victim53" type="victims-occupation" value="Lady"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-label" value="Aristocrat"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-code" value="1700"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-class-1" value="0.5"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-class-3" value="Upper (0.5-3)"/>
      <persName id="t17031013-13-victim53" type="victimName">
       Catherine
       <rs id="t17031013-13-viclabel61" type="occupation">
        Lady
       </rs>
       <interp inst="t17031013-13-viclabel61" type="victimNameLabel" value="Lady"/>
       <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-victim53 t17031013-13-viclabel61"/>
       Herbert
       <interp inst="t17031013-13-victim53" type="surname" value="Herbert"/>
       <interp inst="t17031013-13-victim53" type="given" value="Catherine"/>
       <interp inst="t17031013-13-victim53" type="gender" value="female"/>
       <join result="offenceVictim" targOrder="Y" targets="t17031013-13-off60 t17031013-13-victim53"/>
      </persName>
      , on the
      <rs id="t17031013-13-cd62" type="crimeDate">
       28th of July
      </rs>
      <join result="offenceCrimeDate" targOrder="Y" targets="t17031013-13-off60 t17031013-13-cd62"/>
      last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a
      <rs id="t17031013-13-deflabel63" type="occupation">
       Coachman
      </rs>
      <interp inst="t17031013-13-deflabel63" type="defendantNameLabel" value="Coachman"/>
      <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-deflabel63"/>
      in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him
      <rs id="t17031013-13-verdict64" type="verdictDescription">
       <interp inst="t17031013-13-verdict64" type="verdictCategory" value="guilty"/>
       <interp inst="t17031013-13-verdict64" type="verdictSubcategory" value="guiltyNoDetail"/>
       <interp inst="t17031013-13-verdict64" type="plea" value="notGuilty"/>
       guilty
      </rs>
      .
     </p>
     <p>
      <rs id="t17031013-13-punish65" type="punishmentDescription">
       <interp inst="t17031013-13-punish65" type="punishmentCategory" value="miscPunish"/>
       <interp inst="t17031013-13-punish65" type="punishmentSubcategory" value="brandingOnCheek"/>
       <join result="defendantPunishment" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-punish65"/>
       <note>
        [Branding. See summary.]
       </note>
      </rs>
     </p>
    </div1>
    


We can examine `davis_trial_soup` using `.contents`, which puts all children of a tag in a list.


```python
soup.contents
```




    [<div1 id="t17031013-13" type="trialAccount"> <interp divtype="trialAccount" inst="t17031013-13" type="prevdiv" value="t17031013-12"/> <interp divtype="trialAccount" inst="t17031013-13" type="nextdiv" value="t17031013-14"/> <interp divtype="sessionsPaper" inst="t17031013-13" type="div0" value="17031013"/> <interp inst="t17031013-13" type="collection" value="BAILEY"/> <interp inst="t17031013-13" type="year" value="1703"/> <interp inst="t17031013-13" type="uri" value="sessionsPapers/17031013"/> <interp inst="t17031013-13" type="date" value="17031013"/> <xptr doc="17031013002" imgpath="ob/1700s/17031013002.gif" imgrights="This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original." imgtitle="Proceedings of the Old Bailey, 13th October ." type="pageFacsimile" value="preceding"/> <join id="t17031013-13-off60-c52" result="criminalCharge" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-off60 t17031013-13-verdict64"/> <p> <persName id="t17031013-13-defend52" type="defendantName"> Samuel Davis <interp inst="t17031013-13-defend52" type="surname" value="Davis"/> <interp inst="t17031013-13-defend52" type="given" value="Samuel"/> <interp inst="t17031013-13-defend52" type="gender" value="male"/> </persName> , of the Parish of <placeName id="t17031013-13-defloc59">St. James Westminster</placeName> <interp inst="t17031013-13-defloc59" type="placeName" value="St. James Westminster"/> <interp inst="t17031013-13-defloc59" type="type" value="defendantHome"/> <join result="persNamePlace" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-defloc59"/> <interp inst="t17031013-13-defloc59" type="defendantNameHome" value="St. James Westminster"/>, was indicted for <rs id="t17031013-13-off60" type="offenceDescription"> <interp inst="t17031013-13-off60" type="offenceCategory" value="theft"/> <interp inst="t17031013-13-off60" type="offenceSubcategory" value="grandLarceny"/> feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l. </rs> the Goods of the Honourable <interp inst="t17031013-13-victim53" type="victims-institution" value="no"/> <interp inst="t17031013-13-victim53" type="victims-occupation" value="Lady"/> <interp inst="t17031013-13-victim53" type="victims-hisco-label" value="Aristocrat"/> <interp inst="t17031013-13-victim53" type="victims-hisco-code" value="1700"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-1" value="0.5"/> <interp inst="t17031013-13-victim53" type="victims-hisco-class-3" value="Upper (0.5-3)"/> <persName id="t17031013-13-victim53" type="victimName"> Catherine <rs id="t17031013-13-viclabel61" type="occupation">Lady</rs> <interp inst="t17031013-13-viclabel61" type="victimNameLabel" value="Lady"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-victim53 t17031013-13-viclabel61"/> Herbert <interp inst="t17031013-13-victim53" type="surname" value="Herbert"/> <interp inst="t17031013-13-victim53" type="given" value="Catherine"/> <interp inst="t17031013-13-victim53" type="gender" value="female"/> <join result="offenceVictim" targOrder="Y" targets="t17031013-13-off60 t17031013-13-victim53"/> </persName> , on the <rs id="t17031013-13-cd62" type="crimeDate">28th of July</rs> <join result="offenceCrimeDate" targOrder="Y" targets="t17031013-13-off60 t17031013-13-cd62"/> last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a <rs id="t17031013-13-deflabel63" type="occupation">Coachman</rs> <interp inst="t17031013-13-deflabel63" type="defendantNameLabel" value="Coachman"/> <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-deflabel63"/> in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him <rs id="t17031013-13-verdict64" type="verdictDescription"> <interp inst="t17031013-13-verdict64" type="verdictCategory" value="guilty"/> <interp inst="t17031013-13-verdict64" type="verdictSubcategory" value="guiltyNoDetail"/> <interp inst="t17031013-13-verdict64" type="plea" value="notGuilty"/> guilty </rs>.</p> <p> <rs id="t17031013-13-punish65" type="punishmentDescription"> <interp inst="t17031013-13-punish65" type="punishmentCategory" value="miscPunish"/> <interp inst="t17031013-13-punish65" type="punishmentSubcategory" value="brandingOnCheek"/> <join result="defendantPunishment" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-punish65"/> <note>[Branding. See summary.]</note> </rs> </p> </div1>]



We notice that all information we care about is contained within `<div>` and `</div1>` tags, so we navigate to it. The simplest way to navigate the parse tree is to say the name of the tag you want (`.`). In this case, we want to access div1 tag (but sometimes you can also have html or body tags).

If you are scraping, always put the data in chatgpt or gemini to let it tell how to navigate to the tag you are interested in.


```python
body = soup.find('div1')
print(body.prettify())
```

    <div1 id="t17031013-13" type="trialAccount">
     <interp divtype="trialAccount" inst="t17031013-13" type="prevdiv" value="t17031013-12"/>
     <interp divtype="trialAccount" inst="t17031013-13" type="nextdiv" value="t17031013-14"/>
     <interp divtype="sessionsPaper" inst="t17031013-13" type="div0" value="17031013"/>
     <interp inst="t17031013-13" type="collection" value="BAILEY"/>
     <interp inst="t17031013-13" type="year" value="1703"/>
     <interp inst="t17031013-13" type="uri" value="sessionsPapers/17031013"/>
     <interp inst="t17031013-13" type="date" value="17031013"/>
     <xptr doc="17031013002" imgpath="ob/1700s/17031013002.gif" imgrights="This image is reproduced courtesy of the British Library. Commercial use is prohibited without permission of the owner of the original." imgtitle="Proceedings of the Old Bailey, 13th October ." type="pageFacsimile" value="preceding"/>
     <join id="t17031013-13-off60-c52" result="criminalCharge" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-off60 t17031013-13-verdict64"/>
     <p>
      <persName id="t17031013-13-defend52" type="defendantName">
       Samuel Davis
       <interp inst="t17031013-13-defend52" type="surname" value="Davis"/>
       <interp inst="t17031013-13-defend52" type="given" value="Samuel"/>
       <interp inst="t17031013-13-defend52" type="gender" value="male"/>
      </persName>
      , of the Parish of
      <placeName id="t17031013-13-defloc59">
       St. James Westminster
      </placeName>
      <interp inst="t17031013-13-defloc59" type="placeName" value="St. James Westminster"/>
      <interp inst="t17031013-13-defloc59" type="type" value="defendantHome"/>
      <join result="persNamePlace" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-defloc59"/>
      <interp inst="t17031013-13-defloc59" type="defendantNameHome" value="St. James Westminster"/>
      , was indicted for
      <rs id="t17031013-13-off60" type="offenceDescription">
       <interp inst="t17031013-13-off60" type="offenceCategory" value="theft"/>
       <interp inst="t17031013-13-off60" type="offenceSubcategory" value="grandLarceny"/>
       feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l.
      </rs>
      the Goods of the Honourable
      <interp inst="t17031013-13-victim53" type="victims-institution" value="no"/>
      <interp inst="t17031013-13-victim53" type="victims-occupation" value="Lady"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-label" value="Aristocrat"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-code" value="1700"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-class-1" value="0.5"/>
      <interp inst="t17031013-13-victim53" type="victims-hisco-class-3" value="Upper (0.5-3)"/>
      <persName id="t17031013-13-victim53" type="victimName">
       Catherine
       <rs id="t17031013-13-viclabel61" type="occupation">
        Lady
       </rs>
       <interp inst="t17031013-13-viclabel61" type="victimNameLabel" value="Lady"/>
       <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-victim53 t17031013-13-viclabel61"/>
       Herbert
       <interp inst="t17031013-13-victim53" type="surname" value="Herbert"/>
       <interp inst="t17031013-13-victim53" type="given" value="Catherine"/>
       <interp inst="t17031013-13-victim53" type="gender" value="female"/>
       <join result="offenceVictim" targOrder="Y" targets="t17031013-13-off60 t17031013-13-victim53"/>
      </persName>
      , on the
      <rs id="t17031013-13-cd62" type="crimeDate">
       28th of July
      </rs>
      <join result="offenceCrimeDate" targOrder="Y" targets="t17031013-13-off60 t17031013-13-cd62"/>
      last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a
      <rs id="t17031013-13-deflabel63" type="occupation">
       Coachman
      </rs>
      <interp inst="t17031013-13-deflabel63" type="defendantNameLabel" value="Coachman"/>
      <join result="persNameOccupation" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-deflabel63"/>
      in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him
      <rs id="t17031013-13-verdict64" type="verdictDescription">
       <interp inst="t17031013-13-verdict64" type="verdictCategory" value="guilty"/>
       <interp inst="t17031013-13-verdict64" type="verdictSubcategory" value="guiltyNoDetail"/>
       <interp inst="t17031013-13-verdict64" type="plea" value="notGuilty"/>
       guilty
      </rs>
      .
     </p>
     <p>
      <rs id="t17031013-13-punish65" type="punishmentDescription">
       <interp inst="t17031013-13-punish65" type="punishmentCategory" value="miscPunish"/>
       <interp inst="t17031013-13-punish65" type="punishmentSubcategory" value="brandingOnCheek"/>
       <join result="defendantPunishment" targOrder="Y" targets="t17031013-13-defend52 t17031013-13-punish65"/>
       <note>
        [Branding. See summary.]
       </note>
      </rs>
     </p>
    </div1>
    


We can now start working down the tree. With the body, we can find each child of the body by printing the tags. This will also help us for future reference, if we every want to go through other children in the XML file.

Note that the scheme includes data like child, type and value. 
```xml 
<interp inst="t17031013-13-defend52" type="surname" value="Davis"/>
<interp inst="t17031013-13-defend52" type="given" value="Samuel"/>
<interp inst="t17031013-13-defend52" type="gender" value="male"/>




```python
for child in body.children:
    if child.name:
        print(child.name, 
              child.get("type"), 
              child.get("value"))
```

    interp prevdiv t17031013-12
    interp nextdiv t17031013-14
    interp div0 17031013
    interp collection BAILEY
    interp year 1703
    interp uri sessionsPapers/17031013
    interp date 17031013
    xptr pageFacsimile preceding
    join None None
    p None None
    p None None


Although this may look confusing, it makes a lot of sense if you look at the file in the data folder itself - where the tags are visible and ordered.

Note that "interp" stands for "interpretation". It’s a tag that contains metadata about the parent.

It is usually a leaf node (final node), meaning it does not have any child elements.

Its purpose is to store structured info like type="surname" or value="Davis".

For example, in the above tags, the structure looks something like this:

```xml
<persName id="t17031013-13-defend52" type="defendantName">
   Samuel Davis
   <interp type="surname" value="Davis"/>
   <interp type="given" value="Samuel"/>
   <interp type="gender" value="male"/>
</persName>

Now that we have a list of children to work with let's select one using `.`. Using `.` navigates through the hierarchical structure of XML and helps us keep track of the path we are taking through this file.


```python
choose_p = body.find("p")  # first "p" of the trial - note, that if you want the second <p> tag, 
                           # you'll need to use -> for child in body.find_all("p")[1].find_all(recursive=False) - findall"p"[1] would find the 2nd "p"

# loop over only named tags and show type/value attributes
for child in choose_p.find_all(recursive=False):  # recursive = false means only direct children - ie, dont go deeper into the tags (grandchildren)
    print(child.name, 
          child.get("type"), 
          child.get("value"))
```

    persName defendantName None
    placeName None None
    interp placeName St. James Westminster
    interp type defendantHome
    join None None
    interp defendantNameHome St. James Westminster
    rs offenceDescription None
    interp victims-institution no
    interp victims-occupation Lady
    interp victims-hisco-label Aristocrat
    interp victims-hisco-code 1700
    interp victims-hisco-class-1 0.5
    interp victims-hisco-class-3 Upper (0.5-3)
    persName victimName None
    rs crimeDate None
    join None None
    rs occupation None
    interp defendantNameLabel Coachman
    join None None
    rs verdictDescription None


This isn't very helpful, since we're still left with a bunch of tags and on top of that, we have a lot of repeating tags and names. Let's choose `placename` as our next tag and see what happens.


```python
place_name = body.find("p").find("persName") # you can play around with the tags, for example persName

# iterate over its direct children
for child in place_name.find_all(recursive=True):
    print(child.name, 
          child.get("type"), 
          child.get("value"), 
          child.get_text(strip=True)) ## remove trailing Whitespace
```

    interp surname Davis 
    interp given Samuel 
    interp gender male 


Nothing was printed, so it looks like we hit the end!

Note that, when `None` shows up, it usually means the tag exists, but it doesn’t have a `type` or `value` attribute. In other words:

The information is just in the text content of the tag, not in a structured attribute like `<interp>`.

So you have to look at `.string` or `.text` to get the actual text inside the tag.

Let's use `.string` to examine the data in this element, following the `.` path we used to get here.


```python
body.p.placeName.string
```




    'St. James Westminster'



 Find the defendant's name by traversing through the correct elements. You can check your answer with printed XML using `soup.contents`

You may find `body.p.persname.string` returns None. If a tag, `body.p.persname.string` in this case, contains more than one thing, then it’s not clear what .string should refer to, so .string is defined to be None. Which functions could help us locate the name instead?


```python
soup = BeautifulSoup(xml_content,   ## just repeating what was above - ie where the soup comes from 
                     "xml")

#  Defendant 
defendant_tag = body.find("persName", {"type": "defendantName"})
defendant_name = f"{defendant_tag.find('interp', {'type':'given'})['value']} {defendant_tag.find('interp', {'type':'surname'})['value']}"
defendant_gender = defendant_tag.find("interp", {"type": "gender"})["value"]

# Defendant occupation (by id containing 'deflabel')
defendant_occupation_tag = body.find("rs", id=lambda x: x and "deflabel" in x)
defendant_occupation = defendant_occupation_tag.get_text(strip=True)

#  Victim 
victim_tag = body.find("persName", {"type": "victimName"})
victim_name = f"{victim_tag.find('interp', {'type':'given'})['value']} {victim_tag.find('interp', {'type':'surname'})['value']}"
victim_occupation_tag = victim_tag.find("rs", {"type": "occupation"})
victim_occupation = victim_occupation_tag.get_text(strip=True)

#  Offence 
offence_tag = body.find("rs", {"type": "offenceDescription"})
offence_text = offence_tag.get_text(strip=True)
offence_category = offence_tag.find("interp", {"type": "offenceCategory"})["value"]
offence_subcategory = offence_tag.find("interp", {"type": "offenceSubcategory"})["value"]

#  Verdict 
verdict_tag = body.find("rs", {"type": "verdictDescription"})
verdict_text = verdict_tag.get_text(strip=True)
verdict_category = verdict_tag.find("interp", {"type": "verdictCategory"})["value"]
plea = verdict_tag.find("interp", {"type": "plea"})["value"]

#  Punishment 
punishment_tag = body.find("rs", {"type": "punishmentDescription"})
punishment_subcategory = punishment_tag.find("interp", {"type": "punishmentSubcategory"})["value"]

#  Print all information 
print("Defendant:", defendant_name, f"({defendant_gender})")
print("Defendant occupation:", defendant_occupation)
print("Victim:", victim_name)
print("Victim occupation:", victim_occupation)
print("Offence:", offence_text)
print("Category:", offence_category, "/", offence_subcategory)
print("Verdict:", verdict_text)
print("Plea:", plea)
print("Punishment Subcategory:", punishment_subcategory)


```

    Defendant: Samuel Davis (male)
    Defendant occupation: Coachman
    Victim: Catherine Herbert
    Victim occupation: Lady
    Offence: feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l.
    Category: theft / grandLarceny
    Verdict: guilty
    Plea: notGuilty
    Punishment Subcategory: brandingOnCheek



```python
# Extract main trial text (from first <p>) excluding structured tags
trial_text_tag = body.find("p")
trial_text_parts = []

for elem in trial_text_tag.children:
    if elem.name in ["rs", "interp", "join"]:  # skip structured tags
        continue
    if isinstance(elem, str):
        trial_text_parts.append(elem)
    else:
        trial_text_parts.append(elem.get_text())

# Join and normalize whitespace
trial_text = " ".join(" ".join(trial_text_parts).split())

trial_text
```




    'Samuel Davis , of the Parish of St. James Westminster , was indicted for the Goods of the Honourable Catherine Lady Herbert , on the last. It appeared that the Jewels were put up in a Closet, which was lockt, and the Prisoner being a in the House, took his opportunity to take them; the Lady, when missing them, offered a Reward of Fourscore Pounds to any that could give any notice of it; upon enquiry, the Lady heard that a Diamond was sold on London-Bridge, and they described the Prisoner who sold it, and pursuing him, found the Prisoner at East-Ham, with all his Goods bundled up ready to be gone, and in his Trunk found all the Diamonds but one, which was found upon him in the Role of his Stocking, when searcht before the Justice. He denied the Fact, saying, He found them upon a great Heap of Rubbish, but could not prove it; and that being but a weak Excuse, the Jury found him .'




## Section 4: Putting it all in a dataframe

Now that we have a bunch of XML files and know how to parse through them to extract data, let's put the data from the XML files into a dataframe. Take a look at the XML for [this trial](https://www.dhi.ac.uk/api/data/oldbailey_record_single?idkey=t17031013-13) (and even better, look at what is or isn't consistent between that one and some others), and think about the structure of the data. How would you identify the people involved in a case? How would you identify their roles (witness/defendant/victim/other), or their genders? What can you learn about the alleged offence?

*Note:* Some cases have multiple defendants, multiple victims or multiple witnesses; however, most cases only have at most one of each. You can represent this in a dataframe by having $N$ columns for each property of a defendant, victim, etc., but this results in many many empty cells, and may not be amenable to analysis for the questions you come up with.

Think about the kinds of questions you may want to ask about this data, and refer to the XML for how you might answer them. For example, you may be interested in

- the words used specifically in describing the crime (notice that the text specifically between `<rs id="..." type="offenceDescription">` and `</rs>` gives you this)

- whether any victim was female

- whether any defendant was female

- the `category` (or `subCategory`) of the offense, etc.

- the entire text of the trial (sans tags)

These are questions that can be answered for most if not all cases, so they make good candidates for names of columns.

Consider the following function to get the `date` of the case and return it.


```python
def case_date(soup):                       # Return the case date from a trial soup object
    date_tag = soup.find("interp", {"type": "date"})
    return date_tag.get("value") if date_tag else "Unknown"

print("Case", davis_trial_id, "happened on", case_date(soup))
```

    Case t17031013-13 happened on 17031013


We can write the following function finding every person in a trial, and returning a list of dictionaries of their attributes (e.g. `[{"surname": "FINCH", "given": "JOHN", "gender": "male", "type": "witnessName"}]`). Test it on `davis_trial_soup` used before. **Note:** If you use `find_all`, specify the tag name in lowercase, as beautifulsoup lowercases all tag names.


```python
def people_in_case(case_soup):                      # Return a list of people in the trial with their metadata.
    people = []
    div1 = case_soup.find("div1")
    if not div1:
        return people

    for persName in div1.find_all('persName'):  # capital N
        person = {"type": persName.get("type")}
        for interp in persName.find_all('interp'):
            field_name = interp.get("type")
            field_value = interp.get("value")
            if field_name and field_value:
                person[field_name] = field_value
        people.append(person)
    return people
```


```python
people = people_in_case(soup)
people
```




    [{'type': 'defendantName',
      'surname': 'Davis',
      'given': 'Samuel',
      'gender': 'male'},
     {'type': 'victimName',
      'victimNameLabel': 'Lady',
      'surname': 'Herbert',
      'given': 'Catherine',
      'gender': 'female'}]



We can write the following function to find the `offenseDescription` and `verdictDescription` in a trial. Think about how [the XML](https://www.oldbaileyonline.org/obapi/text?div=t17031013-13) expresses the offenseDescription and verdictDescription, and see if you can write the code without specifically looking for the labels "offenseDescription" and "verdictDescription" (i.e. so that it will work even if a case came up with something like `<rs type="sentencingDescription">`). Get the category, subCategory and textual description of the offense:


```python
"""
Extract all <rs> elements from the case XML and return a dictionary.
Each key is the 'type' of <rs>, and the value is a dictionary with:
  - all <interp> fields as key-value pairs
  - 'text': the visible text inside the <rs> element
"""

def case_descriptions(case_soup):
    descriptions = {}
    for rs in case_soup.find_all('rs'):
        rs_type = rs.get('type', 'unknown')
        desc = {}
        for interp in rs.find_all('interp'):       # Collect interp metadata
            field_name = interp.get("type")
            field_value = interp.get("value")
            desc[field_name] = field_value  
        desc["text"] = rs.get_text(strip=True)      # Add visible text (strip leading/trailing whitespace)
        descriptions[rs_type] = desc
    return descriptions
```


```python
descriptions = case_descriptions(soup)
descriptions
```




    {'offenceDescription': {'offenceCategory': 'theft',
      'offenceSubcategory': 'grandLarceny',
      'text': 'feloniously Stealing 58 Diamonds set in Silver gilt, value 250 l.'},
     'occupation': {'text': 'Coachman'},
     'crimeDate': {'text': '28th of July'},
     'verdictDescription': {'verdictCategory': 'guilty',
      'verdictSubcategory': 'guiltyNoDetail',
      'plea': 'notGuilty',
      'text': 'guilty'},
     'punishmentDescription': {'punishmentCategory': 'miscPunish',
      'punishmentSubcategory': 'brandingOnCheek',
      'text': '[Branding. See summary.]'}}



Once you get this far, you've learned how to parse most of the data provided in the Old Bailey XML. Now, think about the data you now have access to for each case, and complete the following function creating a dataframe describing all of trials in `trials["hits"][:100]`.

One easy way to do this is to make a **list of dictionaries**, where each dictionary is a row, and pass this to `pd.DataFrame`, as in the following example:


```python
pd.DataFrame([{"x": 1, "y": 10}, 
              {"x": 12, "y": 111, "z": 999}])
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
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>111</td>
      <td>999.0</td>
    </tr>
  </tbody>
</table>
</div>



Let us put this all together in a dataframe.

When working with these types of rich datasets, always consider the questions you may want to ask about the data. 

*Note!:* This is not easy. Take it one step at a time, initially just making one a DataFrame with one column, and building up from there.  **The rewarding thing** is that once you've written this up, as you come up with new questions to ask about the case data you'll be able to easily add columns to use in your analysis - always think in terms of new columns. 

When in doubt, use AI by providing it the data that you have and asking it to create a new column from your data. 


```python
from math import nan

def table_of_cases(xml_file_names):
    rows = []

    for xml_file in xml_file_names:                          # Load and parse XML
        with open(f'data/old-bailey/old-bailey-{xml_file}.xml', "r") as f:
            case_soup = BeautifulSoup(f, "xml")

        #  Map occupations via joins 
        occupation_map = {}
        for join_tag in case_soup.find_all("join", {"result": "persNameOccupation"}):
            targets = join_tag.attrs.get("targets", "").split()
            if len(targets) >= 2:
                pers_id = targets[0]                        # persName ID
                rs_tag = case_soup.find(id=targets[1])
                if rs_tag and rs_tag.attrs.get("type") == "occupation":
                    occupation_map[pers_id] = rs_tag.get_text(strip=True)

        #  Extract people 
        people = []
        for persName in case_soup.find_all('persName'):
            person = {"type": persName.attrs.get("type")}                                              
            for interp in persName.find_all('interp'):          # interp metadata
                field = interp.attrs.get("type")
                value = interp.attrs.get("value", "")
                person[field] = value
            # Full name
            given = person.get("given", "")
            surname = person.get("surname", "")
            person["fullName"] = f"{given} {surname}".strip()
            # Occupation from join mapping
            person["occupation"] = occupation_map.get(persName.attrs.get("id"), "")
            people.append(person)

        #  Extract case info 
        date = case_date(case_soup)
        descriptions = case_descriptions(case_soup)
        case_id = case_soup.div1.attrs.get("id", nan)

        # Initialize row
        row = {
            "date": date,
            "id": case_id,
            "text": " ".join(case_soup.get_text(separator=" ").split()),  # cleaned full text
            "any_defendant_female": False,
            "any_defendant_male": False,
            "any_victim_female": False,
            "any_victim_male": False,
            # Offence
            "offenceText": nan,
            "offenceCategory": nan,
            "offenceSubcategory": nan,
            # Verdict
            "verdictText": nan,
            "verdictCategory": nan,
            # Punishment
            "punishmentText": nan,
            "punishmentCategory": nan,
            "punishmentSubcategory": nan,
            # Names and occupations
            "defendantNames": nan,
            "defendantOccupations": nan,
            "victimNames": nan,
            "victimOccupations": nan
        }

        #  Extract <rs> info 
        offence = descriptions.get("offenceDescription", {})
        row["offenceText"] = offence.get("text", nan)
        row["offenceCategory"] = offence.get("offenceCategory", nan)
        row["offenceSubcategory"] = offence.get("offenceSubcategory", nan)

        verdict = descriptions.get("verdictDescription", {})
        row["verdictText"] = verdict.get("text", nan)
        row["verdictCategory"] = verdict.get("verdictCategory", nan)

        punishment = descriptions.get("punishmentDescription", {})
        row["punishmentText"] = punishment.get("text", nan)
        row["punishmentCategory"] = punishment.get("punishmentCategory", nan)
        row["punishmentSubcategory"] = punishment.get("punishmentSubcategory", nan)

        #  Collect gender flags, names, and occupations 
        defendants = [p for p in people if p.get("type")=="defendantName"]
        victims = [p for p in people if p.get("type")=="victimName"]

        # Gender flags
        for d in defendants:
            g = d.get("gender", "").lower()
            if g == "female":
                row["any_defendant_female"] = True
            elif g == "male":
                row["any_defendant_male"] = True

        for v in victims:
            g = v.get("gender", "").lower()
            if g == "female":
                row["any_victim_female"] = True
            elif g == "male":
                row["any_victim_male"] = True

        # Names and occupations as comma-separated strings
        def join_nonempty(items):
            items = [i.strip() for i in items if i and i.strip()]
            return ", ".join(items) if items else nan

        row["defendantNames"] = join_nonempty([d.get("fullName","") for d in defendants])
        row["defendantOccupations"] = join_nonempty([d.get("occupation","") for d in defendants])
        row["victimNames"] = join_nonempty([v.get("fullName","") for v in victims])
        row["victimOccupations"] = join_nonempty([v.get("occupation","") for v in victims])
        
        rows.append(row)

    return pd.DataFrame(rows)

```

Save the dataframe


```python
df = table_of_cases(doc_ids[:50])
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
      <th>date</th>
      <th>id</th>
      <th>text</th>
      <th>any_defendant_female</th>
      <th>any_defendant_male</th>
      <th>any_victim_female</th>
      <th>any_victim_male</th>
      <th>offenceText</th>
      <th>offenceCategory</th>
      <th>offenceSubcategory</th>
      <th>verdictText</th>
      <th>verdictCategory</th>
      <th>punishmentText</th>
      <th>punishmentCategory</th>
      <th>punishmentSubcategory</th>
      <th>defendantNames</th>
      <th>defendantOccupations</th>
      <th>victimNames</th>
      <th>victimOccupations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17540116</td>
      <td>f17540116-1</td>
      <td>THE PROCEEDINGS ON THE King's Commissions of t...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>1</th>
      <td>17540116</td>
      <td>t17540116-1</td>
      <td>80. Hannah Ash , spinster , was indicted for s...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one linen shift, one cotton gown, one...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>pleaded guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Hannah Ash</td>
      <td>spinster</td>
      <td>Richard Beach</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17540116</td>
      <td>t17540116-2</td>
      <td>81. (M.) Peter Foreman and Mary his wife were ...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one pair of linen sheets, value 6 d. ...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Branding. See summary.]</td>
      <td>miscPunish</td>
      <td>branding</td>
      <td>Mary Foreman, Peter Foreman</td>
      <td>wife</td>
      <td>Joseph Sheers</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17540116</td>
      <td>t17540116-3</td>
      <td>82. (M.) Sarah Williams , spinster , was indic...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one brass kettle, value 10 s.</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Sarah Williams</td>
      <td>spinster</td>
      <td>Joseph Smithson</td>
      <td>broker</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17540116</td>
      <td>t17540116-4</td>
      <td>83. (M.) Elizabeth wife of Joseph Kempster , w...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>stealing one feather-bed, value 14 s. one bols...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Elizabeth Kempster</td>
      <td>wife</td>
      <td>Mary Kennedy</td>
      <td>widow</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv("output_trials.csv", index=False)
```

Below is a much much more comlicated way of doing it with XML element tree (instead of beautiful soup): 



```python
def get_clean_counts(series):
    """Splits, explodes, and counts unique values in a series."""
    return series.dropna().str.split(',').explode().str.strip().value_counts()

# Use the function for both
print("Defendant Occupations Counts:")
print(get_clean_counts(df['defendantOccupations']))

print("\nVictim Occupations Counts:")
print(get_clean_counts(df['victimOccupations']))
```

    Defendant Occupations Counts:
    defendantOccupations
    spinster         10
    widow             6
    wife              4
    spinsters         4
    servant           2
    master carman     2
    butcher           1
    Name: count, dtype: int64
    
    Victim Occupations Counts:
    victimOccupations
    widow                        4
    wife                         2
    Esq                          2
    malt factor                  2
    printer                      1
    butcher                      1
    broker                       1
    keep a slop shop             1
    chairman                     1
    brass turner                 1
    hosier                       1
    milliner                     1
    chairmaker                   1
    clerk                        1
    I keep a victualing house    1
    Name: count, dtype: int64


Note that the document that was front matter in the printed Old Bailey Proceedings has NaN in the `offenceCategory` column--it is not a document with a criminal trial record. You could easily eliminate the rows that are not trial records. 

Phew, that's it! Now you know how to parse through XML files using Beautiful Soup and web scrape using the `requests` library! 



# Part II -  NLP tasks 


Recall that one of the readings was to check out the https://nlpprogress.com/ website.

Importantly, NLP practicioners think about NLP in terms of tasks - therefore, transforming your problem into an NLP task is an important step in being able to use these tools correctly!

Now that we have our text data, we can do some NLP. Let's first open the CSV file that we saved above.


```python
df = pd.read_csv("output_trials.csv")
```


```python
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
      <th>date</th>
      <th>id</th>
      <th>text</th>
      <th>any_defendant_female</th>
      <th>any_defendant_male</th>
      <th>any_victim_female</th>
      <th>any_victim_male</th>
      <th>offenceText</th>
      <th>offenceCategory</th>
      <th>offenceSubcategory</th>
      <th>verdictText</th>
      <th>verdictCategory</th>
      <th>punishmentText</th>
      <th>punishmentCategory</th>
      <th>punishmentSubcategory</th>
      <th>defendantNames</th>
      <th>defendantOccupations</th>
      <th>victimNames</th>
      <th>victimOccupations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17540116</td>
      <td>f17540116-1</td>
      <td>THE PROCEEDINGS ON THE King's Commissions of t...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>1</th>
      <td>17540116</td>
      <td>t17540116-1</td>
      <td>80. Hannah Ash , spinster , was indicted for s...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one linen shift, one cotton gown, one...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>pleaded guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Hannah Ash</td>
      <td>spinster</td>
      <td>Richard Beach</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17540116</td>
      <td>t17540116-2</td>
      <td>81. (M.) Peter Foreman and Mary his wife were ...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one pair of linen sheets, value 6 d. ...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Branding. See summary.]</td>
      <td>miscPunish</td>
      <td>branding</td>
      <td>Mary Foreman, Peter Foreman</td>
      <td>wife</td>
      <td>Joseph Sheers</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17540116</td>
      <td>t17540116-3</td>
      <td>82. (M.) Sarah Williams , spinster , was indic...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one brass kettle, value 10 s.</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Sarah Williams</td>
      <td>spinster</td>
      <td>Joseph Smithson</td>
      <td>broker</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17540116</td>
      <td>t17540116-4</td>
      <td>83. (M.) Elizabeth wife of Joseph Kempster , w...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>stealing one feather-bed, value 14 s. one bols...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Elizabeth Kempster</td>
      <td>wife</td>
      <td>Mary Kennedy</td>
      <td>widow</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17540116</td>
      <td>t17540116-5</td>
      <td>84. (M.) John Allen was indicted for stealing ...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one linen shirt, value 1 s. 6 d.</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Acquitted</td>
      <td>notGuilty</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>John Allen</td>
      <td>NaN</td>
      <td>Thomas Fazakerley</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>17540116</td>
      <td>t17540116-6</td>
      <td>85. (M.) William Derter was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing 70 lb. weight of rags, value 4 s.</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Acquitted</td>
      <td>notGuilty</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>William Derter</td>
      <td>NaN</td>
      <td>Thomas Wetworth</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17540116</td>
      <td>t17540116-7</td>
      <td>86. (M.) William Ford was indicted for stealin...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one mare, of a black colour, value 12 l.</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>William Ford</td>
      <td>NaN</td>
      <td>Nicholas Healing</td>
      <td>butcher</td>
    </tr>
    <tr>
      <th>8</th>
      <td>17540116</td>
      <td>t17540116-8</td>
      <td>87. (L.) Anne Beezley , spinster , was indicte...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing a set of green bed curtains and valle...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Anne Beezley</td>
      <td>spinster</td>
      <td>John Jervas</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17540116</td>
      <td>t17540116-9</td>
      <td>88. Robert Barber was indicted for that he, to...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>forging a certain acquittance for the sum of 1...</td>
      <td>deception</td>
      <td>forgery</td>
      <td>acquitted</td>
      <td>notGuilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Robert Barber</td>
      <td>NaN</td>
      <td>Abraham Julian</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17540116</td>
      <td>t17540116-10</td>
      <td>89, 90. (M.) Elizabeth Eaton and Catherine Dav...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one steel tobacco box, val. 2 s. and ...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>guilty of felony only</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Elizabeth Eaton, Catherine Davis</td>
      <td>spinsters, spinsters</td>
      <td>Mark Verit</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>17540116</td>
      <td>t17540116-11</td>
      <td>91. (M.) Samuel Portman was indicted for the w...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>for the wilful murder</td>
      <td>kill</td>
      <td>murder</td>
      <td>Guilty of manslaughter</td>
      <td>guilty</td>
      <td>[Imprisonment. See summary.]</td>
      <td>imprison</td>
      <td>newgate</td>
      <td>Samuel Portman</td>
      <td>NaN</td>
      <td>Elizabeth Norman, George Norman</td>
      <td>wife, wife</td>
    </tr>
    <tr>
      <th>12</th>
      <td>17540116</td>
      <td>t17540116-12</td>
      <td>92. (L.) Anne Ashley , widow , was indicted fo...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one stuff gown, value 6 d. one pair o...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Anne Ashley</td>
      <td>widow</td>
      <td>John Smith</td>
      <td>printer</td>
    </tr>
    <tr>
      <th>13</th>
      <td>17540116</td>
      <td>t17540116-13</td>
      <td>93. (M.) Sarah, wife of Charles Griffice , was...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one copper tea-kettle, value 2 s. one...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty 10 d.</td>
      <td>guilty</td>
      <td>[Whipping. See summary.]</td>
      <td>corporal</td>
      <td>whipping</td>
      <td>Sarah Griffice</td>
      <td>wife</td>
      <td>George Cole</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>17540116</td>
      <td>t17540116-14</td>
      <td>91. (M) Elizabeth Pettit , was indicted for st...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>stealing one stock bed, value 1 s. 6 d. one bl...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty 10 d.</td>
      <td>guilty</td>
      <td>[Whipping. See summary.]</td>
      <td>corporal</td>
      <td>whipping</td>
      <td>Elizabeth Pettit</td>
      <td>NaN</td>
      <td>Rebecca Smith</td>
      <td>widow</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17540116</td>
      <td>t17540116-15</td>
      <td>95. (M.) Sarah Barefoot , spinster , was indic...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one silver watch, value 40 s. the pro...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>Acquitted</td>
      <td>notGuilty</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sarah Barefoot</td>
      <td>spinster</td>
      <td>Richard Tomley</td>
      <td>chairman</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17540116</td>
      <td>t17540116-16</td>
      <td>96, 97, 98, 99, 100. (M ) Thomas Radborn , oth...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing three Holland shirts, value 13 s. and...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Thomas Radborn, John Bell, Daniel Pugh, John R...</td>
      <td>spinster</td>
      <td>Jeffery Burton</td>
      <td>keep a slop shop</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17540116</td>
      <td>t17540116-17</td>
      <td>101. (L.) Thomas Waters was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing 1 Ounces of brass, val. 8 d.</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Thomas Waters</td>
      <td>servant</td>
      <td>Thomas Jeffery</td>
      <td>brass turner</td>
    </tr>
    <tr>
      <th>18</th>
      <td>17540116</td>
      <td>t17540116-18</td>
      <td>102. (L.) Michael Riley was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one linen handkerchief. val. 2 d.</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Michael Riley</td>
      <td>NaN</td>
      <td>John Randall</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>17540116</td>
      <td>t17540116-19</td>
      <td>103. (M.) Edward Allen was indicted for steali...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one black weather sheep, val. 5 s.</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>Edward Allen</td>
      <td>NaN</td>
      <td>Thomas Rant</td>
      <td>Esq</td>
    </tr>
    <tr>
      <th>20</th>
      <td>17540116</td>
      <td>t17540116-20</td>
      <td>104. (L.) John Skelt was indicted for stealing...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing 5 pounds weight of raisins</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>John Skelt</td>
      <td>NaN</td>
      <td>John Porter</td>
      <td>Esq</td>
    </tr>
    <tr>
      <th>21</th>
      <td>17540116</td>
      <td>t17540116-21</td>
      <td>105. (L.) Anne M'Cormeck , spinster , was indi...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one linen handkerchief</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Anne M'Cormeck</td>
      <td>spinster</td>
      <td>James Bulkley</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>17540116</td>
      <td>t17540116-22</td>
      <td>106, 107. (M.) Elizabeth Humphrys and Catharin...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>stealing one silver watch, value 3 l. one gold...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>Both guilty of felony only</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Elizabeth Humphrys, Catharine Brown</td>
      <td>spinsters, spinsters</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>17540116</td>
      <td>t17540116-23</td>
      <td>108. (M.) Henry Champness , was indicted for s...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one pair of linen sheets, two blanket...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Branding. See summary.]</td>
      <td>miscPunish</td>
      <td>branding</td>
      <td>Henry Champness</td>
      <td>NaN</td>
      <td>John Tempest</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>17540116</td>
      <td>t17540116-24</td>
      <td>109. (M) Thomas Cooke , was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>for stealing 144 silk handkerchiefs, value 10 ...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>Guilty of stealing, but not in the shop</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Thomas Cooke</td>
      <td>NaN</td>
      <td>Godard Williams</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>17540116</td>
      <td>t17540116-25</td>
      <td>110. (M.) Elizabeth Hore , widow , was indicte...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one silk handkerchief, value 3 d. one...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty 10 d.</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Elizabeth Hore</td>
      <td>widow</td>
      <td>John Nichols</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>17540116</td>
      <td>t17540116-26</td>
      <td>111. (M.) Martha Mingest , spinster , was indi...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing ninety pounds weight of lead, the pro...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Martha Mingest</td>
      <td>spinster</td>
      <td>George Cellea</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>17540116</td>
      <td>t17540116-27</td>
      <td>112, 113. (M.) Anne Purvise , widow , and Fran...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one pair of linen sheets, value 2 s. ...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>guilty, 10 d.</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Anne Purvise, Frances Hayes</td>
      <td>widow, spinster</td>
      <td>George Carlow</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>17540116</td>
      <td>t17540116-28</td>
      <td>114. (M.) Sarah Conyers , widow , was indicted...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one China mug, value 4 s. and one loo...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty 10 d.</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Sarah Conyers</td>
      <td>widow</td>
      <td>Isaac Bullock</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>17540116</td>
      <td>t17540116-29</td>
      <td>115. (L.) William James was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing six dozen of worsted hose, val. 10 l....</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>William James</td>
      <td>NaN</td>
      <td>John Roberts</td>
      <td>hosier</td>
    </tr>
    <tr>
      <th>30</th>
      <td>17540116</td>
      <td>t17540116-30</td>
      <td>116. (L.) George Butler was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing ten pounds weight of tobacco, val. 5 s.</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>George Butler</td>
      <td>NaN</td>
      <td>Richard Osbourne, Richard Osbourne</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>17540116</td>
      <td>t17540116-31</td>
      <td>117, 118. (L.) Grace Bunn , widow , and Cathar...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing thirty yards of woollen cloth, val. 8 l.</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>acquitted</td>
      <td>notGuilty</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Grace Bunn, Catharine Bunn, John Bunn</td>
      <td>widow</td>
      <td>John Howel</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>17540116</td>
      <td>t17540116-32</td>
      <td>119. (L.) Benjamin Ditto was indicted for that...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>together withSarah HoltandMary Merrit, not yet...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>Acquitted</td>
      <td>notGuilty</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Benjamin Ditto</td>
      <td>NaN</td>
      <td>Robert Colter</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>17540116</td>
      <td>t17540116-33</td>
      <td>120. (M.) George Cole was indicted for stealin...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one pinchbeck metal watch, value 7 l....</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>Guilty of stealing, but not privately from his...</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>George Cole</td>
      <td>NaN</td>
      <td>Thomas Minett</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>17540116</td>
      <td>t17540116-34</td>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>that he, together with one other person unknow...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>Thomas Barnard</td>
      <td>master carman</td>
      <td>Boyce Tree</td>
      <td>malt factor</td>
    </tr>
    <tr>
      <th>35</th>
      <td>17540116</td>
      <td>t17540116-34</td>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>that he, together with one other person unknow...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>Thomas Barnard</td>
      <td>master carman</td>
      <td>Boyce Tree</td>
      <td>malt factor</td>
    </tr>
    <tr>
      <th>36</th>
      <td>17540116</td>
      <td>t17540116-35</td>
      <td>122. (M.) Daniel Wood , butcher , was indicted...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing four ewe sheep, val. 20 s. one ewe la...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>Daniel Wood</td>
      <td>butcher</td>
      <td>John Marsh</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>17540116</td>
      <td>t17540116-36</td>
      <td>123. (M.) Anne Jones , widow , was indicted fo...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>stealing six silk handkerchiefs, val. 12 s. th...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>Guilty 4 s. 10 d.</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Anne Jones</td>
      <td>widow</td>
      <td>Kethurah Vincent</td>
      <td>milliner</td>
    </tr>
    <tr>
      <th>38</th>
      <td>17540116</td>
      <td>t17540116-37</td>
      <td>124. (M.) Grace Riley , spinster , was indicte...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one gold watch, val. 10 l. one cornel...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>Guilty of stealing, but not privately from his...</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Grace Riley</td>
      <td>spinster</td>
      <td>Samuel Collins</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>17540116</td>
      <td>t17540116-38</td>
      <td>125, 126. (M.) William Irons , otherwise Isles...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>that they, on the16th of December, about the h...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>William Irons, Benjamin Richford</td>
      <td>NaN</td>
      <td>William Briley</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>17540116</td>
      <td>t17540116-39</td>
      <td>127. (M.) Mary Jones , spinster , was indicted...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>stealing two linen sheets, value 2 s. one line...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Branding. See summary.]</td>
      <td>miscPunish</td>
      <td>branding</td>
      <td>Mary Jones</td>
      <td>spinster</td>
      <td>Mary Casanover</td>
      <td>widow</td>
    </tr>
    <tr>
      <th>41</th>
      <td>17540116</td>
      <td>t17540116-40</td>
      <td>128. (L.) John Hudson , was indicted for the w...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>wilful murder</td>
      <td>kill</td>
      <td>murder</td>
      <td>Guilty of Manslaughter</td>
      <td>guilty</td>
      <td>[Imprisonment. See summary.]</td>
      <td>imprison</td>
      <td>newgate</td>
      <td>John Hudson</td>
      <td>NaN</td>
      <td>Thomas Moss</td>
      <td>chairmaker</td>
    </tr>
    <tr>
      <th>42</th>
      <td>17540116</td>
      <td>t17540116-41</td>
      <td>129. (M.) Joshua Kidden was indicted for that ...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>that he, on the king's highway, onMary Joneswi...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>Joshua Kidden</td>
      <td>NaN</td>
      <td>Mary Jones</td>
      <td>widow</td>
    </tr>
    <tr>
      <th>43</th>
      <td>17540116</td>
      <td>t17540116-42</td>
      <td>130. (L.) Richard Gandy was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one man's hat. val. 5 s.</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Acq</td>
      <td>notGuilty</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Richard Gandy</td>
      <td>NaN</td>
      <td>Samuel Hall</td>
      <td>clerk</td>
    </tr>
    <tr>
      <th>44</th>
      <td>17540116</td>
      <td>t17540116-43</td>
      <td>131. (L.) Samuel Witham was indicted for break...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>breaking the dwelling house ofThomas Upton, on...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>Death</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>Samuel Witham</td>
      <td>servant</td>
      <td>Thomas Upton</td>
      <td>I keep a victualing house</td>
    </tr>
    <tr>
      <th>45</th>
      <td>17540116</td>
      <td>t17540116-44</td>
      <td>132. (M.) John Watson , was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one cloth coat, value 5 s. two cloth ...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>John Watson</td>
      <td>NaN</td>
      <td>John Campbel</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46</th>
      <td>17540116</td>
      <td>t17540116-45</td>
      <td>133. (M.) Mary, wife of Joseph Durant , was in...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one pair of leather breeches, with si...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Acquitted</td>
      <td>notGuilty</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mary Durant</td>
      <td>wife</td>
      <td>Francis Platt</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>47</th>
      <td>17540116</td>
      <td>t17540116-46</td>
      <td>134. (L.) John Stewart , was indicted for stea...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one great coat, value 2 s. 6 d. one w...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>John Stewart</td>
      <td>NaN</td>
      <td>Thomas Mason</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>48</th>
      <td>17540116</td>
      <td>t17540116-47</td>
      <td>135. (L.) Isaac Angel , was indicted for steal...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing twelve pounds weight of sugar, value ...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Isaac Angel</td>
      <td>NaN</td>
      <td>Benjamin Vaughan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49</th>
      <td>17540116</td>
      <td>t17540116-48</td>
      <td>136. (L.) Matth.ew Minott , was indicted for s...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>stealing one linen handkerchief, value 6 d.</td>
      <td>theft</td>
      <td>pettyLarceny</td>
      <td>Guilty</td>
      <td>guilty</td>
      <td>[Transportation. See summary.]</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>Matth.ew Minott</td>
      <td>NaN</td>
      <td>William Whiteman</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Take a closer look at row 11 - it concerns a murder. Let's read it


```python
text = df['text'][11]
text
```




    "91. (M.) Samuel Portman was indicted for the wilful murder of Elizabeth wife of George Norman . He also stood charged on the coroner's inquisition with manslaughter, Nov. 12 . + Charles Lane. I am fourteen the 19th of April. On the 12th of November my mother was ironing at the window, with me standing by her. We live in Gloucester-court, White-cross street . This was betwixt four and five o'clock; we heard a great noise, and my mother ordered me to open the door, which I did, and saw the prisoner stand on the other side of the way with a piece of brick bat in his hand; his wife was near Mrs. Norman's door. Q. How far was he from his wife at that time? Lane. About nine or ten yards, the alley in which they were is about two yards broad. His wife saw Mrs. Norman put her head out at her door to see what was the matter, and cried out, For God's sake take care. But before these words were well out of her mouth the brick bat hit Mrs. Norman on the forehead, and then bounded into the kennel. Q. Did you hear any words between the prisoner and his wife before this? Lane. Before I opened the door I heard his wife say, You Dog, I keep you. Q. Was the noise you before heard a noise of quarreling? Lane. It was. Q. How far is the kennel from the door? Lane. A little above a yard. Q. How large was the brick bat? Lane. It is a quarter of a brick. It was produced in court. This is it. Mrs. Norman clapped her hand to her forehead, and said, For God's sake send for Mrs. Willis; for I am almost killed; then the deceased's husband took him, and carried him into Mr. Norman's house; they asked him how he came to throw that stone? He own'd several times that he threw it, but not with a design to hurt that poor woman; meaning Mrs. Norman. Elizabeth Willis . I was call'd in, and saw the poor woman bleeding. Mr. Norman and his son brought in the prisoner. They sent for an apothecary, who said the place was very bad; the prisoner said, he threw it at his wife, but did not know he had hurt any body; this he said several times. I had heard much talk between the prisoner and his wife. She said, You rogue, don't I keep you? and such aggravating words. The witness Charles Lane is my son by my first husband. Q. When did she die? Willis. She died on the 5th of December. This was on the 12th of November. I attended her, being a neighbour, all the time of her illness; she found herself as well as could be expected the first fortnight, afterwards complained of a pain shooting thro' her head; then her stomach fell off. Q. Was this wound the occasion of her death, do you think? Willis. I take it it was; for she was in perfect health to the time she looked out at her door. George Norman . I am husband to the deceased. On the 12th of November I was sitting by the fireside, and heard a great noise by a woman; my wife being washing in the house, she opened the door to see what was the matter. She had not opened it above a minute and half before she said she was almost killed. I got up and went to the door; there was Charles Lane standing at his mother's door, I said, Who threw this? and he pointed to the prisoner; there was none but him near. My son and I then went and laid hold on him, and brought him into my house. He told me he did not throw it at my wife, but at his own wife. His wife came in, and he said, This comes by your tongue, this is all along of you, and was going to beat her. I asked him where he lived; he said, he was out of business, and could make no satisfaction. I took him before Justice Withers, and a surgeon giving an account that the wound was a very bad one, he was committed. Q. When did she die? Willis On the fifth of December. She kept her bed for about a week; we were in hopes, for near a fortnight, that she would have recovered. She always complained of a very great pain in her head, and that when she bent her head forward she had a violent weight upon the forepart of it. I judge that was the cause of her death; for she was in good health before, and was then a washing. Q. from Prisoner. What was her opinion of it? Willis. She often said this wound would be the death of her; but she was of opinion, from his words, that he did not intend to throw it at her. Elizabeth Norman . I am daughter to the deceased, and was standing by my mother when she was washing. Hearing a noise, she opened the door and looked out, and in about a minute the brick bat cut her in the face. I saw the brick bat fly from her face to the kennel, which is about a yard and half. She turned into the house, and put her hands to her forehead, which was covered with blood. The prisoner was brought in, and said he flung it at his wife. Q. to Charles Lane. How far was the prisoner's wife from Mrs. Norman's door? Lane. Not a yard. Q. from prisoner. Have you heard your mother say it would be the occasion of her death? Lane. She always said she thought it would. Q. from prisoner. Did not you hear her say it was done by accident? Lane. I don't remember she said any thing about it. Q. to Norman. Who pick'd up the brickbat? Norman. Charles Lane and my son picked it up. Richard Riley . I am a surgeon, and attended the deceased. I was called in that day to dress the wound. I found a pretty large one across the right eye-brow, and the upper part of the skull was quite bare. I could find no fissure nor fracture. A fissure is a crack like a hair, and we try it with ink. There was a large effusion of blood from the wound, and she complained of being very faint and weak. The next morning she told me she had had a good night, and there were no fever or bad symptoms for about a fortnight. My man attended her two days. I went next day, when she complained of a pain in her head, back, and neck; she had a fever coming on, and her pulse was quick. She continued growing worse and worse for about a week, and then died, which was in the morning, the fifth of December. After her death, I opened her head before the coroner's jury. Upon taking off her scalp, we found the member that closes the brain quite free from any coagulated blood. When we came to divide the dura mater, we found a collection of matter on the right side the head, the same side the wound was given, but the left globe of the brain was in its natural state. Q. Upon the whole, what in your opinion was the cause of her death? Riley. In all probability it might be from a rupture of a small vessel, which might be occasioned by the blow given, from the concussion of the brain. Such a concussion of the brain is much more dangerous, than if the skull had only been broke. When she was so well as I have mentioned, Mr. Norman applied to me to certify to the Justice she was out of danger, to get the prisoner out of Jail. The deceased said she thought it was an accidental thing, that she unfortunately opened the door, and the man threw the stone at the same time. Q. What occasioned that fever? Riley. It is my opinion it was from that eruption of the brain. Robert Taylor . I am headborough. About four o'clock Mr. Norman came and told me to come to his house, and said a man had thrown a stone at his wife, and had hit her. There was the prisoner, he said he did not intend it for her, but to affright his own wife; but she opening the door, it happened to hit her. Prisoner's Defence. I happening to go into a publick house, my wife came in and called me all the names she could, and up with a stone to break the window, but I desired she'd desist till I had paid for my two pints of beer. I took the stone out of her hand, and jerk'd it away, and it happened to hit this poor woman. For the Prisoner. James Smith . About eight days after this accident I went to the deceased's house in behalf of the prisoner, and the surgeon was there, who said she was in a fair way of recovery. She said she was better, but in pain, and was very willing to let the man out of jail, but her husband was against it, and wanted five shillings per week till she could work, and to pay all charges. She said to him, let him out by all means, as it was an accident, and was not flung with an intent to hurt her, and his abiding in jail will be no satisfaction to her. Q. to Norman. Do you remember your wife saying these words? Norman. No, I remember no such words. Isaac Campion . The prisoner was my servant five or six years, and I never knew him behave amiss during that time, nor ever heard to the contrary but that he is an honest man. John Burnham . I have known the prisoner about seven years, and take him to be a very honest man. Guilty of manslaughter . [Imprisonment. See summary.]"



To get started, as last time, we import spaCy, one of the many libraries available for natural language processing in Python.


```python
import spacy
```


```python
nlp = spacy.load('en_core_web_sm') # english small model

# Call the variable to examine the object
nlp
```




    <spacy.lang.en.English at 0x7cd68a9f35c0>



One of the foundational tasks in NLP - that is useful in law as well - is known as Named Entity Recognition (NER). 

Named entity recognition (NER) is the task of recognising and classifying entities named in a text.

spaCy can recognise the named entities annotated in the OntoNotes 5 corpus, such as persons, geographic locations and products, to name but a few examples.

We can use the Doc object's .ents attribute to get the named entities.



```python
doc = nlp(text)
```


```python
# Call the variable to examine the object
doc.ents
```




    (91,
     Samuel Portman,
     Elizabeth,
     George Norman,
     Nov. 12,
     Charles Lane,
     fourteen the 19th of April,
     the 12th of November,
     Gloucester,
     White-cross,
     four and five o'clock,
     Norman,
     Lane,
     About nine or,
     about two yards,
     Norman,
     Norman,
     Lane,
     Lane,
     Lane,
     Lane,
     a quarter,
     Norman,
     Willis,
     Norman,
     Norman,
     Elizabeth Willis,
     Norman,
     Charles Lane,
     first,
     Willis,
     the 5th of December,
     the 12th of November,
     first,
     thro,
     Willis,
     George Norman,
     the 12th of November,
     a minute and,
     half,
     Charles Lane,
     Willis,
     the fifth of December,
     about a week,
     Prisoner,
     Willis,
     Elizabeth Norman,
     about a minute,
     half,
     Charles Lane,
     Norman,
     Lane,
     Lane,
     Lane,
     Norman,
     Norman,
     Charles Lane,
     Richard Riley,
     that day,
     The next morning,
     a good night,
     two days,
     next day,
     about a week,
     the morning,
     the fifth of December,
     Norman,
     Justice,
     Robert Taylor,
     About four o'clock,
     Norman,
     Prisoner,
     two,
     Prisoner,
     James Smith,
     About eight days,
     five,
     Norman,
     Norman,
     Isaac Campion,
     five or six years,
     John Burnham,
     about seven years)




```python

from spacy import displacy
displacy.render(doc, style='ent')
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    91
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
. (M.) 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Samuel Portman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 was indicted for the wilful murder of 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Elizabeth
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 wife of 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    George Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . He also stood charged on the coroner's inquisition with manslaughter, 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Nov. 12
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 . + 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Charles Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. I am 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    fourteen the 19th of April
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. On 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the 12th of November
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 my mother was ironing at the window, with me standing by her. We live in 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Gloucester
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
-court, 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    White-cross
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 street . This was betwixt 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    four and five o'clock
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
; we heard a great noise, and my mother ordered me to open the door, which I did, and saw the prisoner stand on the other side of the way with a piece of brick bat in his hand; his wife was near Mrs. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
's door. Q. How far was he from his wife at that time? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    About nine or
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 ten yards, the alley in which they were is 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    about two yards
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">QUANTITY</span>
</mark>
 broad. His wife saw Mrs. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 put her head out at her door to see what was the matter, and cried out, For God's sake take care. But before these words were well out of her mouth the brick bat hit Mrs. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 on the forehead, and then bounded into the kennel. Q. Did you hear any words between the prisoner and his wife before this? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. Before I opened the door I heard his wife say, You Dog, I keep you. Q. Was the noise you before heard a noise of quarreling? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. It was. Q. How far is the kennel from the door? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. A little above a yard. Q. How large was the brick bat? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. It is 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    a quarter
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 of a brick. It was produced in court. This is it. Mrs. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 clapped her hand to her forehead, and said, For God's sake send for Mrs. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Willis
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
; for I am almost killed; then the deceased's husband took him, and carried him into Mr. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
's house; they asked him how he came to throw that stone? He own'd several times that he threw it, but not with a design to hurt that poor woman; meaning Mrs. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Elizabeth Willis
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . I was call'd in, and saw the poor woman bleeding. Mr. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 and his son brought in the prisoner. They sent for an apothecary, who said the place was very bad; the prisoner said, he threw it at his wife, but did not know he had hurt any body; this he said several times. I had heard much talk between the prisoner and his wife. She said, You rogue, don't I keep you? and such aggravating words. The witness 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Charles Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 is my son by my 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    first
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORDINAL</span>
</mark>
 husband. Q. When did she die? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Willis
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. She died on 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the 5th of December
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. This was on 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the 12th of November
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. I attended her, being a neighbour, all the time of her illness; she found herself as well as could be expected the 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    first
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORDINAL</span>
</mark>
 fortnight, afterwards complained of a pain shooting 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    thro
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
' her head; then her stomach fell off. Q. Was this wound the occasion of her death, do you think? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Willis
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. I take it it was; for she was in perfect health to the time she looked out at her door. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    George Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . I am husband to the deceased. On 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the 12th of November
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 I was sitting by the fireside, and heard a great noise by a woman; my wife being washing in the house, she opened the door to see what was the matter. She had not opened it above 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    a minute and
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>

<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    half
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 before she said she was almost killed. I got up and went to the door; there was 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Charles Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 standing at his mother's door, I said, Who threw this? and he pointed to the prisoner; there was none but him near. My son and I then went and laid hold on him, and brought him into my house. He told me he did not throw it at my wife, but at his own wife. His wife came in, and he said, This comes by your tongue, this is all along of you, and was going to beat her. I asked him where he lived; he said, he was out of business, and could make no satisfaction. I took him before Justice Withers, and a surgeon giving an account that the wound was a very bad one, he was committed. Q. When did she die? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Willis
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 On 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the fifth of December
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. She kept her bed for 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    about a week
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
; we were in hopes, for near a fortnight, that she would have recovered. She always complained of a very great pain in her head, and that when she bent her head forward she had a violent weight upon the forepart of it. I judge that was the cause of her death; for she was in good health before, and was then a washing. Q. from 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Prisoner
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. What was her opinion of it? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Willis
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. She often said this wound would be the death of her; but she was of opinion, from his words, that he did not intend to throw it at her. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Elizabeth Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . I am daughter to the deceased, and was standing by my mother when she was washing. Hearing a noise, she opened the door and looked out, and in 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    about a minute
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
 the brick bat cut her in the face. I saw the brick bat fly from her face to the kennel, which is about a yard and 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    half
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
. She turned into the house, and put her hands to her forehead, which was covered with blood. The prisoner was brought in, and said he flung it at his wife. Q. to 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Charles Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. How far was the prisoner's wife from Mrs. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
's door? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. Not a yard. Q. from prisoner. Have you heard your mother say it would be the occasion of her death? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. She always said she thought it would. Q. from prisoner. Did not you hear her say it was done by accident? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. I don't remember she said any thing about it. Q. to 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. Who pick'd up the brickbat? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Charles Lane
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 and my son picked it up. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Richard Riley
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . I am a surgeon, and attended the deceased. I was called in 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    that day
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 to dress the wound. I found a pretty large one across the right eye-brow, and the upper part of the skull was quite bare. I could find no fissure nor fracture. A fissure is a crack like a hair, and we try it with ink. There was a large effusion of blood from the wound, and she complained of being very faint and weak. 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    The next morning
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
 she told me she had had 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    a good night
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
, and there were no fever or bad symptoms for about a fortnight. My man attended her 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    two days
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. I went 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    next day
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
, when she complained of a pain in her head, back, and neck; she had a fever coming on, and her pulse was quick. She continued growing worse and worse for 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    about a week
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
, and then died, which was in 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the morning
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
, 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the fifth of December
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. After her death, I opened her head before the coroner's jury. Upon taking off her scalp, we found the member that closes the brain quite free from any coagulated blood. When we came to divide the dura mater, we found a collection of matter on the right side the head, the same side the wound was given, but the left globe of the brain was in its natural state. Q. Upon the whole, what in your opinion was the cause of her death? Riley. In all probability it might be from a rupture of a small vessel, which might be occasioned by the blow given, from the concussion of the brain. Such a concussion of the brain is much more dangerous, than if the skull had only been broke. When she was so well as I have mentioned, Mr. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 applied to me to certify to the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Justice
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 she was out of danger, to get the prisoner out of Jail. The deceased said she thought it was an accidental thing, that she unfortunately opened the door, and the man threw the stone at the same time. Q. What occasioned that fever? Riley. It is my opinion it was from that eruption of the brain. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Robert Taylor
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . I am headborough. 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    About four o'clock
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
 Mr. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 came and told me to come to his house, and said a man had thrown a stone at his wife, and had hit her. There was the prisoner, he said he did not intend it for her, but to affright his own wife; but she opening the door, it happened to hit her. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Prisoner
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
's Defence. I happening to go into a publick house, my wife came in and called me all the names she could, and up with a stone to break the window, but I desired she'd desist till I had paid for my 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    two
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 pints of beer. I took the stone out of her hand, and jerk'd it away, and it happened to hit this poor woman. For the 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Prisoner
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    James Smith
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    About eight days
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 after this accident I went to the deceased's house in behalf of the prisoner, and the surgeon was there, who said she was in a fair way of recovery. She said she was better, but in pain, and was very willing to let the man out of jail, but her husband was against it, and wanted 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    five
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 shillings per week till she could work, and to pay all charges. She said to him, let him out by all means, as it was an accident, and was not flung with an intent to hurt her, and his abiding in jail will be no satisfaction to her. Q. to 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. Do you remember your wife saying these words? 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Norman
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. No, I remember no such words. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Isaac Campion
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . The prisoner was my servant 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    five or six years
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
, and I never knew him behave amiss during that time, nor ever heard to the contrary but that he is an honest man. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    John Burnham
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 . I have known the prisoner 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    about seven years
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
, and take him to be a very honest man. Guilty of manslaughter . [Imprisonment. See summary.]</div></span>


What if we want to see only organizations? or "NORPS"? What is even a NORP?


```python
spacy.explain("NORP")
```




    'Nationalities or religious or political groups'




```python

def organizations(doc):
    return [x for x in (nlp(doc)).ents if x.label_ == "ORG"] #list comprehension
```


```python
organizations(text)
```




    [White-cross, Justice]




```python

def persons(doc):
    return [x for x in (nlp(doc)).ents if x.label_ == "PERSON"] #list comprehension
```


```python
persons(text)
```




    [Samuel Portman,
     Elizabeth,
     George Norman,
     Charles Lane,
     Norman,
     Lane,
     Norman,
     Norman,
     Lane,
     Lane,
     Lane,
     Lane,
     Norman,
     Willis,
     Norman,
     Norman,
     Elizabeth Willis,
     Norman,
     Charles Lane,
     Willis,
     thro,
     Willis,
     George Norman,
     Charles Lane,
     Willis,
     Prisoner,
     Willis,
     Elizabeth Norman,
     Charles Lane,
     Norman,
     Lane,
     Lane,
     Lane,
     Norman,
     Charles Lane,
     Richard Riley,
     Norman,
     Robert Taylor,
     Norman,
     Prisoner,
     Prisoner,
     James Smith,
     Norman,
     Isaac Campion,
     John Burnham]




```python
# Loop over items in the Doc object, using the variable 'token' to refer to items in the list
for token in doc[5:20]:
    
    # Print the token and the POS tags
    print(token, token.pos_, 
          token.tag_)
```

    Samuel PROPN NNP
    Portman PROPN NNP
    was AUX VBD
    indicted VERB VBN
    for ADP IN
    the DET DT
    wilful ADJ JJ
    murder NOUN NN
    of ADP IN
    Elizabeth PROPN NNP
    wife NOUN NN
    of ADP IN
    George PROPN NNP
    Norman PROPN NNP
    . PUNCT .



```python
# Loop over items in the Doc object, using the variable 'token' to refer to items in the list
for token in doc[5:20]:
    
    # Print the token and its dependency tag
    print(token, 
          token.dep_)

```

    Samuel compound
    Portman nsubjpass
    was auxpass
    indicted ROOT
    for prep
    the det
    wilful amod
    murder pobj
    of prep
    Elizabeth compound
    wife pobj
    of prep
    George compound
    Norman pobj
    . punct



```python
displacy.render(doc[5:20], style='dep', options={'compact': False})
```


```python
# Loop over sentences in the Doc object and count them using enumerate()
sentences = []
for number, sent in enumerate(doc.sents):
    sentences.append(sent)
    # Print the token and its dependency tag
    print(number, sent)
```

    0 91. (M.) Samuel Portman was indicted for the wilful murder of Elizabeth wife of George Norman .
    1 He also stood charged on the coroner's inquisition with manslaughter, Nov. 12 .
    2 + Charles Lane.
    3 I am fourteen the 19th of April.
    4 On the 12th of November my mother was ironing at the window, with me standing by her.
    5 We live in Gloucester-court, White-cross street .
    6 This was betwixt four and five o'clock; we heard a great noise, and my mother ordered me to open the door, which I did, and saw the prisoner stand on the other side of the way with a piece of brick bat in his hand; his wife was near Mrs. Norman's door.
    7 Q. How far was he from his wife at that time?
    8 Lane.
    9 About nine or ten yards, the alley in which they were is about two yards broad.
    10 His wife saw Mrs. Norman put her head out at her door to see what was the matter, and cried out, For God's sake take care.
    11 But before these words were well out of her mouth the brick bat hit Mrs. Norman on the forehead, and then bounded into the kennel.
    12 Q. Did you hear any words between the prisoner and his wife before this?
    13 Lane.
    14 Before I opened the door I heard his wife say, You Dog, I keep you.
    15 Q. Was the noise you before heard a noise of quarreling?
    16 Lane.
    17 It was.
    18 Q. How far is the kennel from the door?
    19 Lane.
    20 A little above a yard.
    21 Q.
    22 How large was the brick bat?
    23 Lane.
    24 It is a quarter of a brick.
    25 It was produced in court.
    26 This is it.
    27 Mrs. Norman clapped her hand to her forehead, and said, For God's sake send for Mrs. Willis; for I am almost killed; then the deceased's husband took him, and carried him into Mr. Norman's house; they asked him how he came to throw that stone?
    28 He own'd several times that he threw it, but not with a design to hurt that poor woman; meaning Mrs. Norman.
    29 Elizabeth Willis .
    30 I was call'd in, and saw the poor woman bleeding.
    31 Mr. Norman and his son brought in the prisoner.
    32 They sent for an apothecary, who said the place was very bad; the prisoner said, he threw it at his wife, but did not know he had hurt any body; this he said several times.
    33 I had heard much talk between the prisoner and his wife.
    34 She said, You rogue, don't I keep you?
    35 and such aggravating words.
    36 The witness Charles Lane is my son by my first husband.
    37 Q.
    38 When did she die?
    39 Willis.
    40 She died on the 5th of December.
    41 This was on the 12th of November.
    42 I attended her, being a neighbour, all the time of her illness; she found herself as well as could be expected the first fortnight, afterwards complained of a pain shooting thro' her head; then her stomach fell off.
    43 Q. Was this wound the occasion of her death, do you think?
    44 Willis.
    45 I take it it was; for she was in perfect health to the time she looked out at her door.
    46 George Norman .
    47 I am husband to the deceased.
    48 On the 12th of November I was sitting by the fireside, and heard a great noise by a woman; my wife being washing in the house, she opened the door to see what was the matter.
    49 She had not opened it above a minute and half before she said she was almost killed.
    50 I got up and went to the door; there was Charles Lane standing at his mother's door, I said, Who threw this?
    51 and he pointed to the prisoner; there was none but him near.
    52 My son and I then went and laid hold on him, and brought him into my house.
    53 He told me he did not throw it at my wife, but at his own wife.
    54 His wife came in, and he said, This comes by your tongue, this is all along of you, and was going to beat her.
    55 I asked him where he lived; he said, he was out of business, and could make no satisfaction.
    56 I took him before Justice Withers, and a surgeon giving an account that the wound was a very bad one, he was committed.
    57 Q.
    58 When did she die?
    59 Willis On the fifth of December.
    60 She kept her bed for about a week; we were in hopes, for near a fortnight, that she would have recovered.
    61 She always complained of a very great pain in her head, and that when she bent her head forward she had a violent weight upon the forepart of it.
    62 I judge that was the cause of her death; for she was in good health before, and was then a washing.
    63 Q. from Prisoner.
    64 What was her opinion of it?
    65 Willis.
    66 She often said this wound would be the death of her; but she was of opinion, from his words, that he did not intend to throw it at her.
    67 Elizabeth Norman .
    68 I am daughter to the deceased, and was standing by my mother when she was washing.
    69 Hearing a noise, she opened the door and looked out, and in about a minute the brick bat cut her in the face.
    70 I saw the brick bat fly from her face to the kennel, which is about a yard and half.
    71 She turned into the house, and put her hands to her forehead, which was covered with blood.
    72 The prisoner was brought in, and said he flung it at his wife.
    73 Q. to Charles Lane.
    74 How far was the prisoner's wife from Mrs. Norman's door?
    75 Lane.
    76 Not a yard.
    77 Q. from prisoner.
    78 Have you heard your mother say it would be the occasion of her death?
    79 Lane.
    80 She always said she thought it would.
    81 Q. from prisoner.
    82 Did not you hear her say it was done by accident?
    83 Lane.
    84 I don't remember she said any thing about it.
    85 Q. to Norman.
    86 Who pick'd up the brickbat?
    87 Norman.
    88 Charles Lane and my son picked it up.
    89 Richard Riley .
    90 I am a surgeon, and attended the deceased.
    91 I was called in that day to dress the wound.
    92 I found a pretty large one across the right eye-brow, and the upper part of the skull was quite bare.
    93 I could find no fissure nor fracture.
    94 A fissure is a crack like a hair, and we try it with ink.
    95 There was a large effusion of blood from the wound, and she complained of being very faint and weak.
    96 The next morning she told me she had had a good night, and there were no fever or bad symptoms for about a fortnight.
    97 My man attended her two days.
    98 I went next day, when she complained of a pain in her head, back, and neck; she had a fever coming on, and her pulse was quick.
    99 She continued growing worse and worse for about a week, and then died, which was in the morning, the fifth of December.
    100 After her death, I opened her head before the coroner's jury.
    101 Upon taking off her scalp, we found the member that closes the brain quite free from any coagulated blood.
    102 When we came to divide the dura mater, we found a collection of matter on the right side the head, the same side the wound was given, but the left globe of the brain was in its natural state.
    103 Q. Upon the whole, what in your opinion was the cause of her death?
    104 Riley.
    105 In all probability it might be from a rupture of a small vessel, which might be occasioned by the blow given, from the concussion of the brain.
    106 Such a concussion of the brain is much more dangerous, than if the skull had only been broke.
    107 When she was so well as I have mentioned, Mr. Norman applied to me to certify to the Justice she was out of danger, to get the prisoner out of Jail.
    108 The deceased said she thought it was an accidental thing, that she unfortunately opened the door, and the man threw the stone at the same time.
    109 Q.
    110 What occasioned that fever?
    111 Riley.
    112 It is my opinion it was from that eruption of the brain.
    113 Robert Taylor .
    114 I am headborough.
    115 About four o'clock Mr. Norman came and told me to come to his house, and said a man had thrown a stone at his wife, and had hit her.
    116 There was the prisoner, he said he did not intend it for her, but to affright his own wife; but she opening the door, it happened to hit her.
    117 Prisoner's Defence.
    118 I happening to go into a publick house, my wife came in and called me all the names she could, and up with a stone to break the window, but I desired she'd desist till I had paid for my two pints of beer.
    119 I took the stone out of her hand, and jerk'd it away, and it happened to hit this poor woman.
    120 For the Prisoner.
    121 James Smith .
    122 About eight days after this accident I went to the deceased's house in behalf of the prisoner, and the surgeon was there, who said she was in a fair way of recovery.
    123 She said she was better, but in pain, and was very willing to let the man out of jail, but her husband was against it, and wanted five shillings per week till she could work, and to pay all charges.
    124 She said to him, let him out by all means, as it was an accident, and was not flung with an intent to hurt her, and his abiding in jail will be no satisfaction to her.
    125 Q. to Norman.
    126 Do you remember your wife saying these words?
    127 Norman.
    128 No, I remember no such words.
    129 Isaac Campion .
    130 The prisoner was my servant five or six years, and I never knew him behave amiss during that time, nor ever heard to the contrary but that he is an honest man.
    131 John Burnham .
    132 I have known the prisoner about seven years, and take him to be a very honest man.
    133 Guilty of manslaughter .
    134 [Imprisonment.
    135 See summary.]



```python
sentences[49]
```




    She had not opened it above a minute and half before she said she was almost killed.




```python
for token in sentences[1]:
    
    # Print the token and its dependency tag
    print(token, 
          token.dep_)
```

    He nsubj
    also advmod
    stood ROOT
    charged advcl
    on prep
    the det
    coroner poss
    's case
    inquisition pobj
    with prep
    manslaughter pobj
    , punct
    Nov. npadvmod
    12 nummod
    . punct



```python
spacy.explain('pobj')
```




    'object of preposition'




```python
!pip install -q pytextrank
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m26.0[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


More phrase methods - Let's try finding phrases in our text. There is a simple mathematical tool called "Text Rank" which (long story short) is able to find phrases very efficiently and has a simple implementation in spacy


```python
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import pytextrank
```

    /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages



```python
nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")
```




    <pytextrank.base.BaseTextRankFactory at 0x7cd6f5129640>




```python
doc = nlp(text)


for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)
```

    Mrs. Norman
    0.07170439113417262 4
    [Mrs. Norman, Mrs. Norman, Mrs. Norman, Mrs. Norman]
    Elizabeth wife
    0.06747236124675021 1
    [Elizabeth wife]
    Mr. Norman
    0.06598258860587485 3
    [Mr. Norman, Mr. Norman, Mr. Norman]
    Elizabeth Norman
    0.062464203043643286 2
    [Elizabeth Norman, Elizabeth Norman]
    George Norman
    0.060364541524874235 4
    [George Norman, George Norman, George Norman, George Norman]
    Norman
    0.06019355692656814 18
    [Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman, Norman]
    prisoner
    0.05833977794372302 2
    [prisoner, prisoner]
    brick bat
    0.05647064542038944 1
    [brick bat]
    Mrs. Willis
    0.0516538013463117 1
    [Mrs. Willis]
    Mrs. Normans door
    0.04912001013627757 2
    [Mrs. Norman's door, Mrs. Norman's door]
    bad symptoms
    0.04838770375338422 1
    [bad symptoms]
    Charles Lane
    0.04807621583071045 9
    [Charles Lane, Charles Lane, Charles Lane, Charles Lane, Charles Lane, Charles Lane, Charles Lane, Charles Lane, Charles Lane]
    Mr. Normans house
    0.04343095408408718 1
    [Mr. Norman's house]
    pain
    0.041259915658322356 1
    [pain]
    week
    0.04093651699782863 1
    [week]
    good health
    0.03937249338185243 1
    [good health]
    Isaac Campion
    0.03834331587720725 2
    [Isaac Campion, Isaac Campion]
    James Smith
    0.03834331587720725 2
    [James Smith, James Smith]
    John Burnham
    0.03834331587720725 2
    [John Burnham, John Burnham]
    Richard Riley
    0.03834331587720725 2
    [Richard Riley, Richard Riley]
    Robert Taylor
    0.03834331587720725 2
    [Robert Taylor, Robert Taylor]
    Elizabeth Willis
    0.03779381096792588 2
    [Elizabeth Willis, Elizabeth Willis]
    next day
    0.037309827696225935 1
    [next day]
    Lane
    0.036087008934028116 16
    [Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane, Lane]
    Q.
    0.03608244653022945 12
    [Q., Q., Q., Q., Q., Q., Q., Q., Q., Q., Q., Q.]
    husband
    0.03571508669086562 1
    [husband]
    his own wife
    0.03543526933299692 2
    [his own wife, his own wife]
    Justice Withers
    0.03446719715685401 1
    [Justice Withers]
    Samuel Portman
    0.033717453999861126 1
    [Samuel Portman]
    Jail
    0.03307804691229592 1
    [Jail]
    jail
    0.03307804691229592 2
    [jail, jail]
    matter
    0.0328544701076546 1
    [matter]
    court
    0.03284660398352266 1
    [court]
    blood
    0.03269028853714516 2
    [blood, blood]
    perfect health
    0.032161136004831566 1
    [perfect health]
    the prisoners wife
    0.0319647536653794 1
    [the prisoner's wife]
    accident
    0.03139527159385695 1
    [accident]
    opinion
    0.030588745852660044 1
    [opinion]
    His wife
    0.030481819804286044 2
    [His wife, His wife]
    his wife
    0.030481819804286044 8
    [his wife, his wife, his wife, his wife, his wife, his wife, his wife, his wife]
    my wife
    0.030481819804286044 3
    [my wife, my wife, my wife]
    your wife
    0.030481819804286044 1
    [your wife]
    the brick bat
    0.02994808223605382 4
    [the brick bat, the brick bat, the brick bat, the brick bat]
    first
    0.029220548806347613 2
    [first, first]
    Justice
    0.029114540120668008 1
    [Justice]
    December
    0.029037036821994763 3
    [December, December, December]
    Willis
    0.028994554950439323 9
    [Willis, Willis, Willis, Willis, Willis, Willis, Willis, Willis, Willis]
    April
    0.028757486907905437 1
    [April]
    Riley
    0.028757486907905437 2
    [Riley, Riley]
    November
    0.02841285959311881 3
    [November, November, November]
    manslaughter
    0.028146034423618637 2
    [manslaughter, manslaughter]
    Elizabeth
    0.02768094017803918 1
    [Elizabeth]
    her door
    0.027137953312513525 2
    [her door, her door]
    the door
    0.027137953312513525 8
    [the door, the door, the door, the door, the door, the door, the door, the door]
    The prisoner
    0.02694598920021718 2
    [The prisoner, The prisoner]
    the prisoner
    0.02694598920021718 10
    [the prisoner, the prisoner, the prisoner, the prisoner, the prisoner, the prisoner, the prisoner, the prisoner, the prisoner, the prisoner]
    the same time
    0.026580865860492835 1
    [the same time]
    a publick house
    0.026204750831393613 1
    [a publick house]
    Gloucester
    0.02619765455120352 1
    [Gloucester]
    much talk
    0.026164302791296945 1
    [much talk]
    his mothers door
    0.026143597986656584 1
    [his mother's door]
    + Charles Lane
    0.025496263670762555 1
    [+ Charles Lane]
    a great noise
    0.025460214156039735 2
    [a great noise, a great noise]
    half
    0.02532034219422203 4
    [half, half, half, half]
    an honest man
    0.02487122852568942 1
    [an honest man]
    ink
    0.0247896382966877 1
    [ink]
    thro
    0.024758788714340067 1
    [thro]
    that poor woman
    0.024592870667686586 1
    [that poor woman]
    the poor woman
    0.024592870667686586 1
    [the poor woman]
    this poor woman
    0.024592870667686586 1
    [this poor woman]
    hopes
    0.024049019729098956 1
    [hopes]
    her head
    0.02399269085896329 5
    [her head, her head, her head, her head, her head]
    the head
    0.02399269085896329 1
    [the head]
    the right side
    0.023904637048456464 1
    [the right side]
    and such aggravating words
    0.023771810498548998 1
    [and such aggravating words]
    White-cross street
    0.023758777461560965 1
    [White-cross street]
    a pain shooting thro her head
    0.023669838439991653 1
    [a pain shooting thro' her head]
    danger
    0.023644459188060525 1
    [danger]
    business
    0.023444687917117103 1
    [business]
    Prisoner
    0.023328632337848817 4
    [Prisoner, Prisoner, Prisoner, Prisoner]
    my first husband
    0.02307275389870155 1
    [my first husband]
    the first fortnight
    0.022691577889749064 1
    [the first fortnight]
    behalf
    0.022373314807301037 1
    [behalf]
    his house
    0.022260889497257963 1
    [his house]
    my house
    0.022260889497257963 1
    [my house]
    the house
    0.022260889497257963 2
    [the house, the house]
    the same side
    0.021931628825625753 1
    [the same side]
    the wound
    0.02190935260611988 4
    [the wound, the wound, the wound, the wound]
    this wound
    0.02190935260611988 1
    [this wound]
    no such words
    0.021654844638061683 1
    [no such words]
    Gods sake
    0.02131654550063419 2
    [God's sake, God's sake]
    hold
    0.021304198339058653 1
    [hold]
    that time
    0.02118936906536467 2
    [that time, that time]
    the time
    0.02118936906536467 1
    [the time]
    Gloucester-court
    0.02100723980699607 1
    [Gloucester-court]
    beer
    0.02091074509589875 1
    [beer]
    the other side
    0.020812283102453723 1
    [the other side]
    the deceaseds house
    0.02066159977618391 1
    [the deceased's house]
    the brain
    0.020629712974801165 5
    [the brain, the brain, the brain, the brain, the brain]
    a very bad one
    0.020147876433639873 1
    [a very bad one]
    White-cross
    0.019894681007750124 1
    [White-cross]
    a brick
    0.019780630176522936 1
    [a brick]
    My man
    0.019773080524749034 1
    [My man]
    a man
    0.019773080524749034 1
    [a man]
    the man
    0.019773080524749034 2
    [the man, the man]
    any coagulated blood
    0.019660089580719794 1
    [any coagulated blood]
    a very great pain
    0.019623044146963687 1
    [a very great pain]
    a good night
    0.019540675004318244 2
    [a good night, a good night]
    my mother
    0.01925568979346773 3
    [my mother, my mother, my mother]
    your mother
    0.01925568979346773 1
    [your mother]
    a small vessel
    0.019254514824876164 1
    [a small vessel]
    The next morning
    0.019204021816822223 1
    [The next morning]
    her death
    0.019077581130877826 5
    [her death, her death, her death, her death, her death]
    the death
    0.019077581130877826 1
    [the death]
    a pain
    0.019057138729658935 1
    [a pain]
    none
    0.018819477239709906 1
    [none]
    a stone
    0.01866039555281823 2
    [a stone, a stone]
    that stone
    0.01866039555281823 1
    [that stone]
    the stone
    0.01866039555281823 2
    [the stone, the stone]
    the right eye-brow
    0.01865238032555397 1
    [the right eye-brow]
    fracture
    0.01860244722202111 1
    [fracture]
    the wilful murder
    0.01859352816407591 1
    [the wilful murder]
    the upper part
    0.018441602733662705 1
    [the upper part]
    recovery
    0.01841887742332758 1
    [recovery]
    a large effusion
    0.018417565190540344 1
    [a large effusion]
    a fair way
    0.01821819810412653 1
    [a fair way]
    a very honest man
    0.018050909381953365 1
    [a very honest man]
    the left globe
    0.017996860921131855 1
    [the left globe]
    daughter
    0.01797755664082626 1
    [daughter]
    care
    0.0179670129145061 1
    [care]
    her hand
    0.017935474988346134 2
    [her hand, her hand]
    her hands
    0.017935474988346134 1
    [her hands]
    his hand
    0.017935474988346134 1
    [his hand]
    calld
    0.01764101776811295 1
    [call'd]
    quarreling
    0.01753357054045607 1
    [quarreling]
    any words
    0.01741858559361231 1
    [any words]
    his words
    0.01741858559361231 1
    [his words]
    these words
    0.01741858559361231 2
    [these words, these words]
    a noise
    0.01710324389124167 2
    [a noise, a noise]
    the noise
    0.01710324389124167 1
    [the noise]
    a woman
    0.017054915461100256 1
    [a woman]
    the deceaseds husband
    0.0169995390356296 1
    [the deceased's husband]
    an accidental thing
    0.01682751611864454 1
    [an accidental thing]
    the dura mater
    0.01676196420194101 1
    [the dura mater]
    her husband
    0.016496091932081673 1
    [her husband]
    a fortnight
    0.016038555918117158 1
    [a fortnight]
    a yard
    0.016018946946233948 1
    [a yard]
    a violent weight
    0.01595820907114954 1
    [a violent weight]
    My son
    0.015955496483228495 1
    [My son]
    his son
    0.015955496483228495 1
    [his son]
    my son
    0.015955496483228495 2
    [my son, my son]
    her forehead
    0.015890116365132864 2
    [her forehead, her forehead]
    the forehead
    0.015890116365132864 1
    [the forehead]
    (M.) Samuel Portman
    0.01583892474515373 1
    [(M.) Samuel Portman]
    a fever
    0.015722735754722446 1
    [a fever]
    no fever
    0.015722735754722446 1
    [no fever]
    that fever
    0.015722735754722446 1
    [that fever]
    a surgeon
    0.01548251004902082 1
    [a surgeon]
    the surgeon
    0.01548251004902082 1
    [the surgeon]
    its natural state
    0.015175360450725926 1
    [its natural state]
    the matter
    0.015174829728589516 2
    [the matter, the matter]
    a pretty large one
    0.015171420609611143 1
    [a pretty large one]
    Prisoners Defence
    0.015001394584303274 1
    [Prisoner's Defence]
    the coroners inquisition
    0.014650601708804561 1
    [the coroner's inquisition]
    an accident
    0.01450085480479578 1
    [an accident]
    this accident
    0.01450085480479578 1
    [this accident]
    that day
    0.014302799819368495 2
    [that day, that day]
    two days
    0.014302799819368495 2
    [two days, two days]
    her opinion
    0.014128336521765062 1
    [her opinion]
    my opinion
    0.014128336521765062 1
    [my opinion]
    your opinion
    0.014128336521765062 1
    [your opinion]
    the window
    0.014050239549039984 2
    [the window, the window]
    the skull
    0.014018934254626789 2
    [the skull, the skull]
    the 12th of November
    0.013887271672269417 3
    [the 12th of November, the 12th of November, the 12th of November]
    about a week
    0.013730525215383064 4
    [about a week, about a week, about a week, about a week]
    summary
    0.013644049471422913 1
    [summary]
    the morning
    0.013608420375485812 2
    [the morning, the morning]
    the kennel
    0.013499574637823015 3
    [the kennel, the kennel, the kennel]
    the Justice
    0.013447430060799868 1
    [the Justice]
    the way
    0.013423088893452853 1
    [the way]
    A fissure
    0.01339158964144923 1
    [A fissure]
    no fissure
    0.01339158964144923 1
    [no fissure]
    the brickbat
    0.013282514245997074 1
    [the brickbat]
    the fifth of December
    0.013231533222337836 2
    [the fifth of December, the fifth of December]
    her face
    0.013124421245148211 1
    [her face]
    the face
    0.013124421245148211 1
    [the face]
    the coroners jury
    0.013114263730468296 1
    [the coroner's jury]
    a minute
    0.012903855420875963 1
    [a minute]
    the concussion
    0.01285394462129627 1
    [the concussion]
    a crack
    0.012803514884571439 1
    [a crack]
    a hair
    0.012803514884571439 1
    [a hair]
    the cause
    0.012388226389557618 2
    [the cause, the cause]
    the 5th of December
    0.012378911695915235 1
    [the 5th of December]
    the blow
    0.01232316278447461 1
    [the blow]
    fourteen the 19th of April
    0.011982286211627266 1
    [fourteen the 19th of April]
    the 12th
    0.011840829231967713 3
    [the 12th, the 12th, the 12th]
    the member
    0.011698069482428136 1
    [the member]
    the occasion
    0.011653814799012159 2
    [the occasion, the occasion]
    about a fortnight
    0.01164694874401578 1
    [about a fortnight]
    Not a yard
    0.011632709014977045 1
    [Not a yard]
    about a yard
    0.011632709014977045 1
    [about a yard]
    about two yards
    0.011632709014977045 1
    [about two yards]
    any thing
    0.011598924784473205 1
    [any thing]
    five shillings
    0.01158958406927075 1
    [five shillings]
    no satisfaction
    0.011293628974201943 2
    [no satisfaction, no satisfaction]
    a rupture
    0.011227344530948948 1
    [a rupture]
    a collection
    0.011195715882338338 1
    [a collection]
    her illness
    0.011190073826902226 1
    [her illness]
    an account
    0.01111628962977657 1
    [an account]
    an intent
    0.010883246165844525 1
    [an intent]
    the fireside
    0.010823957813520774 1
    [the fireside]
    the Prisoner
    0.01077503372806626 1
    [the Prisoner]
    a piece
    0.010604217536000285 1
    [a piece]
    the place
    0.010572017534016355 1
    [the place]
    any body
    0.01039617425610677 1
    [any body]
    About eight days
    0.010386469782103619 1
    [About eight days]
    a design
    0.010368394085886481 1
    [a design]
    his abiding
    0.01023705408647371 1
    [his abiding]
    all probability
    0.010117063539624216 1
    [all probability]
    her bed
    0.009992895570791423 1
    [her bed]
    all means
    0.009982035179985265 1
    [all means]
    her stomach
    0.009973068904868293 1
    [her stomach]
    your tongue
    0.009964902558656738 1
    [your tongue]
    Nov. 12
    0.009936875218019637 1
    [Nov. 12]
    an apothecary
    0.009849928749809543 1
    [an apothecary]
    her scalp
    0.009838719487895178 1
    [her scalp]
    the contrary
    0.009831509611469804 1
    [the contrary]
    her pulse
    0.0096281733343578 1
    [her pulse]
    a neighbour
    0.009535214284448878 1
    [a neighbour]
    a minute and
    0.009370578215047668 1
    [a minute and]
    about a minute
    0.009370578215047668 2
    [about a minute, about a minute]
    Such a concussion
    0.009334333772128662 1
    [Such a concussion]
    all charges
    0.009303344565872625 1
    [all charges]
    the whole
    0.009239193727337077 1
    [the whole]
    my servant
    0.00922899386601579 1
    [my servant]
    about seven years
    0.009149669205400746 1
    [about seven years]
    the forepart
    0.009107433633335472 1
    [the forepart]
    the alley
    0.00904551828397098 1
    [the alley]
    her mouth
    0.008966303463988757 1
    [her mouth]
    a washing
    0.008490828287490282 1
    [a washing]
    The witness
    0.008388118076117513 1
    [The witness]
    the 5th
    0.008268783575806755 1
    [the 5th]
    About four oclock
    0.008128254819795858 1
    [About four o'clock]
    the fifth
    0.008077721447306526 1
    [the fifth]
    my two pints
    0.007794230232985349 1
    [my two pints]
    all the names
    0.007777304325882977 1
    [all the names]
    that eruption
    0.007744831894536296 1
    [that eruption]
    five or six years
    0.00749855210502102 2
    [five or six years, five or six years]
    four and five oclock
    0.006661458564332647 1
    [four and five o'clock]
    a quarter
    0.006349683265769255 2
    [a quarter, a quarter]
    [Imprisonment
    0.005144295647050943 1
    [[Imprisonment]
    91
    0.0 1
    [91]
    About nine or
    0.0 1
    [About nine or]
    He
    0.0 3
    [He, He, He]
    I
    0.0 38
    [I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I]
    It
    0.0 4
    [It, It, It, It]
    She
    0.0 11
    [She, She, She, She, She, She, She, She, She, She, She]
    They
    0.0 1
    [They]
    This
    0.0 4
    [This, This, This, This]
    We
    0.0 1
    [We]
    What
    0.0 2
    [What, What]
    Who
    0.0 2
    [Who, Who]
    You
    0.0 1
    [You]
    five
    0.0 1
    [five]
    he
    0.0 18
    [he, he, he, he, he, he, he, he, he, he, he, he, he, he, he, he, he, he]
    her
    0.0 13
    [her, her, her, her, her, her, her, her, her, her, her, her, her]
    herself
    0.0 1
    [herself]
    him
    0.0 12
    [him, him, him, him, him, him, him, him, him, him, him, him]
    it
    0.0 26
    [it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it, it]
    me
    0.0 7
    [me, me, me, me, me, me, me]
    she
    0.0 32
    [she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she, she]
    that
    0.0 2
    [that, that]
    they
    0.0 2
    [they, they]
    this
    0.0 4
    [this, this, this, this]
    two
    0.0 1
    [two]
    we
    0.0 6
    [we, we, we, we, we, we]
    what
    0.0 3
    [what, what, what]
    which
    0.0 6
    [which, which, which, which, which, which]
    who
    0.0 2
    [who, who]
    you
    0.0 9
    [you, you, you, you, you, you, you, you, you]


Let's try this for another  case


```python
text = df['text'][3]
text
```




    "82. (M.) Sarah Williams , spinster , was indicted for stealing one brass kettle, value 10 s. the property of Joseph Smithson , Dec. 28 . ++ Ann Smithson . I am wife to Joseph Smithson , we live in Round Court, in the Strand , he is a broker . On the 27th of December last I was informed a woman had taken a large pot from the door; I followed the woman into Chandos-street, where she was running along with it, but seeing me come after her dropped it, and one Hawkins took her; I took the pot up, and returned home, having left no body in the shop. Mary Busbey . I saw a woman take a pot away from the prosecutor's door, and run away with it, but cannot say the prisoner is the woman; I did not see her face. I went and told Ann Smithson , she went after her, and the prisoner was brought back. Burley Hawkins. I was at the prosecutor's shop when the last witness informed them of a woman that had taken a pot away. I went with Ann Smithson , and saw the prisoner drop the pot. I took her, and brought her back to Mr. Smithson's shop; Mrs. Smithson took the pot, which was delivered to the constable, who is not here. Prisoner's Defence. I have not been near Round-Court these three months, nor never saw the pot till after I was taken up. To her character. Mr. Welch. I keep the Bell in Fleet-street, I have known the prisoner about nine months; I never heard any thing ill of her. She lived with me about six months, and behaved well all that time. Guilty . [Transportation. See summary.]"




```python
doc = nlp(text)


for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)
```

    Joseph Smithson
    0.11712577562831375 4
    [Joseph Smithson, Joseph Smithson, Joseph Smithson, Joseph Smithson]
    Ann Smithson
    0.11343869149169401 5
    [Ann Smithson, Ann Smithson, Ann Smithson, Ann Smithson, Ann Smithson]
    Mrs. Smithson
    0.10909232840659061 1
    [Mrs. Smithson]
    Smithson
    0.10519153008093728 2
    [Smithson, Smithson]
    Round Court
    0.08566195138081686 2
    [Round Court, Round Court]
    Mary Busbey
    0.07832304187224903 2
    [Mary Busbey, Mary Busbey]
    Sarah Williams
    0.07355932921376608 1
    [Sarah Williams]
    Mr. Smithsons shop
    0.07095986478098902 1
    [Mr. Smithson's shop]
    spinster
    0.06187613066025379 1
    [spinster]
    Burley Hawkins
    0.06143364656383225 2
    [Burley Hawkins, Burley Hawkins]
    Mr. Welch
    0.060455869859694734 1
    [Mr. Welch]
    Prisoner
    0.058742281404186775 1
    [Prisoner]
    Hawkins
    0.057659161170482996 1
    [Hawkins]
    a large pot
    0.05430736731332438 1
    [a large pot]
    last
    0.04963511519809828 1
    [last]
    Bell
    0.04775423425943192 1
    [Bell]
    Strand
    0.047090919041653606 1
    [Strand]
    December
    0.04592290254501361 1
    [December]
    Round-Court
    0.045429110033285966 2
    [Round-Court, Round-Court]
    a pot
    0.04442524254932972 2
    [a pot, a pot]
    the pot
    0.04442524254932972 4
    [the pot, the pot, the pot, the pot]
    Fleet-street
    0.04437875997420309 1
    [Fleet-street]
    Chandos-street
    0.044167802983248426 2
    [Chandos-street, Chandos-street]
    ++ Ann Smithson
    0.043662572712832304 1
    [++ Ann Smithson]
    one brass kettle
    0.0415909064829841 1
    [one brass kettle]
    Prisoners Defence
    0.0415370655232689 1
    [Prisoner's Defence]
    a woman
    0.04153054349213569 3
    [a woman, a woman, a woman]
    the woman
    0.04153054349213569 2
    [the woman, the woman]
    the prisoner
    0.04138718226537616 4
    [the prisoner, the prisoner, the prisoner, the prisoner]
    wife
    0.041203468464656876 1
    [wife]
    the last witness
    0.034926977142469125 1
    [the last witness]
    (M.) Sarah Williams
    0.03443534122775805 1
    [(M.) Sarah Williams]
    the prosecutors shop
    0.03383516108544899 1
    [the prosecutor's shop]
    the prosecutors door
    0.03296490592275533 1
    [the prosecutor's door]
    the shop
    0.032862731809362576 1
    [the shop]
    the door
    0.031397189600646934 1
    [the door]
    Welch
    0.03104151521840712 1
    [Welch]
    summary
    0.02922412344210867 1
    [summary]
    about nine months
    0.026671961324029624 1
    [about nine months]
    about six months
    0.026671961324029624 1
    [about six months]
    these three months
    0.026671961324029624 1
    [these three months]
    one Hawkins
    0.026631625778421097 1
    [one Hawkins]
    10 s.
    0.026042308099535644 1
    [10 s.]
    the property
    0.02601970047061776 1
    [the property]
    Transportation
    0.022750787759664506 1
    [Transportation]
    the 27th of December
    0.022506176213671254 1
    [the 27th of December]
    the Bell
    0.022056736003701977 1
    [the Bell]
    the Strand
    0.02175036382806847 1
    [the Strand]
    any thing
    0.021714180026476203 1
    [any thing]
    no body
    0.020540842245252502 1
    [no body]
    a broker
    0.01942367585545845 1
    [a broker]
    the 27th
    0.019252892573116995 1
    [the 27th]
    Dec. 28
    0.01920081761402509 1
    [Dec. 28]
    the constable
    0.018412018168583132 1
    [the constable]
    her face
    0.01802221770821832 1
    [her face]
    [Transportation
    0.010508138749854679 1
    [[Transportation]
    her character
    0.010508138749854679 1
    [her character]
    10
    0.0 1
    [10]
    82
    0.0 1
    [82]
    I
    0.0 15
    [I, I, I, I, I, I, I, I, I, I, I, I, I, I, I]
    She
    0.0 1
    [She]
    he
    0.0 1
    [he]
    her
    0.0 6
    [her, her, her, her, her, her]
    it
    0.0 3
    [it, it, it]
    me
    0.0 2
    [me, me]
    one
    0.0 2
    [one, one]
    she
    0.0 2
    [she, she]
    that
    0.0 1
    [that]
    them
    0.0 1
    [them]
    we
    0.0 1
    [we]
    which
    0.0 1
    [which]
    who
    0.0 1
    [who]


Let's go back to our named entity recognition code. I want to save the people involved in the cases as a seperate column.


```python
df_text = df[[ "id", "text", "offenceCategory", "offenceSubcategory", "punishmentCategory", "punishmentSubcategory"]]
df_text
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
      <th>offenceCategory</th>
      <th>offenceSubcategory</th>
      <th>punishmentCategory</th>
      <th>punishmentSubcategory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f17540116-1</td>
      <td>THE PROCEEDINGS ON THE King's Commissions of t...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>t17540116-1</td>
      <td>80. Hannah Ash , spinster , was indicted for s...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>2</th>
      <td>t17540116-2</td>
      <td>81. (M.) Peter Foreman and Mary his wife were ...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>miscPunish</td>
      <td>branding</td>
    </tr>
    <tr>
      <th>3</th>
      <td>t17540116-3</td>
      <td>82. (M.) Sarah Williams , spinster , was indic...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>4</th>
      <td>t17540116-4</td>
      <td>83. (M.) Elizabeth wife of Joseph Kempster , w...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>5</th>
      <td>t17540116-5</td>
      <td>84. (M.) John Allen was indicted for stealing ...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>t17540116-6</td>
      <td>85. (M.) William Derter was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>t17540116-7</td>
      <td>86. (M.) William Ford was indicted for stealin...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>8</th>
      <td>t17540116-8</td>
      <td>87. (L.) Anne Beezley , spinster , was indicte...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>9</th>
      <td>t17540116-9</td>
      <td>88. Robert Barber was indicted for that he, to...</td>
      <td>deception</td>
      <td>forgery</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>10</th>
      <td>t17540116-10</td>
      <td>89, 90. (M.) Elizabeth Eaton and Catherine Dav...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>11</th>
      <td>t17540116-11</td>
      <td>91. (M.) Samuel Portman was indicted for the w...</td>
      <td>kill</td>
      <td>murder</td>
      <td>imprison</td>
      <td>newgate</td>
    </tr>
    <tr>
      <th>12</th>
      <td>t17540116-12</td>
      <td>92. (L.) Anne Ashley , widow , was indicted fo...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>13</th>
      <td>t17540116-13</td>
      <td>93. (M.) Sarah, wife of Charles Griffice , was...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>corporal</td>
      <td>whipping</td>
    </tr>
    <tr>
      <th>14</th>
      <td>t17540116-14</td>
      <td>91. (M) Elizabeth Pettit , was indicted for st...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>corporal</td>
      <td>whipping</td>
    </tr>
    <tr>
      <th>15</th>
      <td>t17540116-15</td>
      <td>95. (M.) Sarah Barefoot , spinster , was indic...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>t17540116-16</td>
      <td>96, 97, 98, 99, 100. (M ) Thomas Radborn , oth...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>17</th>
      <td>t17540116-17</td>
      <td>101. (L.) Thomas Waters was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>18</th>
      <td>t17540116-18</td>
      <td>102. (L.) Michael Riley was indicted for steal...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>19</th>
      <td>t17540116-19</td>
      <td>103. (M.) Edward Allen was indicted for steali...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>20</th>
      <td>t17540116-20</td>
      <td>104. (L.) John Skelt was indicted for stealing...</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>21</th>
      <td>t17540116-21</td>
      <td>105. (L.) Anne M'Cormeck , spinster , was indi...</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>22</th>
      <td>t17540116-22</td>
      <td>106, 107. (M.) Elizabeth Humphrys and Catharin...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>23</th>
      <td>t17540116-23</td>
      <td>108. (M.) Henry Champness , was indicted for s...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>miscPunish</td>
      <td>branding</td>
    </tr>
    <tr>
      <th>24</th>
      <td>t17540116-24</td>
      <td>109. (M) Thomas Cooke , was indicted for steal...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>25</th>
      <td>t17540116-25</td>
      <td>110. (M.) Elizabeth Hore , widow , was indicte...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>26</th>
      <td>t17540116-26</td>
      <td>111. (M.) Martha Mingest , spinster , was indi...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>27</th>
      <td>t17540116-27</td>
      <td>112, 113. (M.) Anne Purvise , widow , and Fran...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>28</th>
      <td>t17540116-28</td>
      <td>114. (M.) Sarah Conyers , widow , was indicted...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>29</th>
      <td>t17540116-29</td>
      <td>115. (L.) William James was indicted for steal...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>30</th>
      <td>t17540116-30</td>
      <td>116. (L.) George Butler was indicted for steal...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>31</th>
      <td>t17540116-31</td>
      <td>117, 118. (L.) Grace Bunn , widow , and Cathar...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>t17540116-32</td>
      <td>119. (L.) Benjamin Ditto was indicted for that...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>t17540116-33</td>
      <td>120. (M.) George Cole was indicted for stealin...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>34</th>
      <td>t17540116-34</td>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>35</th>
      <td>t17540116-34</td>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>36</th>
      <td>t17540116-35</td>
      <td>122. (M.) Daniel Wood , butcher , was indicted...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>37</th>
      <td>t17540116-36</td>
      <td>123. (M.) Anne Jones , widow , was indicted fo...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>38</th>
      <td>t17540116-37</td>
      <td>124. (M.) Grace Riley , spinster , was indicte...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>39</th>
      <td>t17540116-38</td>
      <td>125, 126. (M.) William Irons , otherwise Isles...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>40</th>
      <td>t17540116-39</td>
      <td>127. (M.) Mary Jones , spinster , was indicted...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>miscPunish</td>
      <td>branding</td>
    </tr>
    <tr>
      <th>41</th>
      <td>t17540116-40</td>
      <td>128. (L.) John Hudson , was indicted for the w...</td>
      <td>kill</td>
      <td>murder</td>
      <td>imprison</td>
      <td>newgate</td>
    </tr>
    <tr>
      <th>42</th>
      <td>t17540116-41</td>
      <td>129. (M.) Joshua Kidden was indicted for that ...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>43</th>
      <td>t17540116-42</td>
      <td>130. (L.) Richard Gandy was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>t17540116-43</td>
      <td>131. (L.) Samuel Witham was indicted for break...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
    </tr>
    <tr>
      <th>45</th>
      <td>t17540116-44</td>
      <td>132. (M.) John Watson , was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>46</th>
      <td>t17540116-45</td>
      <td>133. (M.) Mary, wife of Joseph Durant , was in...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>47</th>
      <td>t17540116-46</td>
      <td>134. (L.) John Stewart , was indicted for stea...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>48</th>
      <td>t17540116-47</td>
      <td>135. (L.) Isaac Angel , was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
    <tr>
      <th>49</th>
      <td>t17540116-48</td>
      <td>136. (L.) Matth.ew Minott , was indicted for s...</td>
      <td>theft</td>
      <td>pettyLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_text['persons_involved'] = df_text['text'].apply(lambda x: [person.text for person in persons(x)])

```

    /tmp/ipykernel_1131978/3373888748.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_text['persons_involved'] = df_text['text'].apply(lambda x: [person.text for person in persons(x)])



```python
df_text['text'][3]
```




    "82. (M.) Sarah Williams , spinster , was indicted for stealing one brass kettle, value 10 s. the property of Joseph Smithson , Dec. 28 . ++ Ann Smithson . I am wife to Joseph Smithson , we live in Round Court, in the Strand , he is a broker . On the 27th of December last I was informed a woman had taken a large pot from the door; I followed the woman into Chandos-street, where she was running along with it, but seeing me come after her dropped it, and one Hawkins took her; I took the pot up, and returned home, having left no body in the shop. Mary Busbey . I saw a woman take a pot away from the prosecutor's door, and run away with it, but cannot say the prisoner is the woman; I did not see her face. I went and told Ann Smithson , she went after her, and the prisoner was brought back. Burley Hawkins. I was at the prosecutor's shop when the last witness informed them of a woman that had taken a pot away. I went with Ann Smithson , and saw the prisoner drop the pot. I took her, and brought her back to Mr. Smithson's shop; Mrs. Smithson took the pot, which was delivered to the constable, who is not here. Prisoner's Defence. I have not been near Round-Court these three months, nor never saw the pot till after I was taken up. To her character. Mr. Welch. I keep the Bell in Fleet-street, I have known the prisoner about nine months; I never heard any thing ill of her. She lived with me about six months, and behaved well all that time. Guilty . [Transportation. See summary.]"




```python
df_text
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
      <th>offenceCategory</th>
      <th>offenceSubcategory</th>
      <th>punishmentCategory</th>
      <th>punishmentSubcategory</th>
      <th>persons_involved</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f17540116-1</td>
      <td>THE PROCEEDINGS ON THE King's Commissions of t...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[Oyer, Gaol Delivery, Thomas Rawlinson, M. COO...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>t17540116-1</td>
      <td>80. Hannah Ash , spinster , was indicted for s...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Hannah Ash]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>t17540116-2</td>
      <td>81. (M.) Peter Foreman and Mary his wife were ...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>miscPunish</td>
      <td>branding</td>
      <td>[Peter Foreman, Mary, Joseph Sheers, Guilty]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>t17540116-3</td>
      <td>82. (M.) Sarah Williams , spinster , was indic...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Sarah Williams, Joseph Smithson, Ann Smithson...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>t17540116-4</td>
      <td>83. (M.) Elizabeth wife of Joseph Kempster , w...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Elizabeth, Joseph Kempster, Mary Kennedy, Mar...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>t17540116-5</td>
      <td>84. (M.) John Allen was indicted for stealing ...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[John Allen, Thomas Fazakerley]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>t17540116-6</td>
      <td>85. (M.) William Derter was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[William Derter, Thomas Wetworth]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>t17540116-7</td>
      <td>86. (M.) William Ford was indicted for stealin...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[William Ford, Nicholas Healing, Abraham Sande...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>t17540116-8</td>
      <td>87. (L.) Anne Beezley , spinster , was indicte...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[L., Anne Beezley, John Jervas, John Jervas, s...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>t17540116-9</td>
      <td>88. Robert Barber was indicted for that he, to...</td>
      <td>deception</td>
      <td>forgery</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Robert Barber, John Thorp, Abraham Julian, Gu...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>t17540116-10</td>
      <td>89, 90. (M.) Elizabeth Eaton and Catherine Dav...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Elizabeth Eaton, Catherine Davis, Mark Verit,...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>t17540116-11</td>
      <td>91. (M.) Samuel Portman was indicted for the w...</td>
      <td>kill</td>
      <td>murder</td>
      <td>imprison</td>
      <td>newgate</td>
      <td>[Samuel Portman, Elizabeth, George Norman, Cha...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>t17540116-12</td>
      <td>92. (L.) Anne Ashley , widow , was indicted fo...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Anne Ashley, John Smith, John Smith, Smith, J...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>t17540116-13</td>
      <td>93. (M.) Sarah, wife of Charles Griffice , was...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>corporal</td>
      <td>whipping</td>
      <td>[Sarah, Charles Griffice, George Cole, Mary Co...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>t17540116-14</td>
      <td>91. (M) Elizabeth Pettit , was indicted for st...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>corporal</td>
      <td>whipping</td>
      <td>[Elizabeth Pettit, Rebecca Smith, Rebecca Smit...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>t17540116-15</td>
      <td>95. (M.) Sarah Barefoot , spinster , was indic...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[Sarah Barefoot, Richard Tomley, Richard Tomle...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>t17540116-16</td>
      <td>96, 97, 98, 99, 100. (M ) Thomas Radborn , oth...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[John Bell, Daniel Pugh, John Radborn, Anne Br...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>t17540116-17</td>
      <td>101. (L.) Thomas Waters was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Thomas Waters, Thomas Jeffery, Thomas Jeffery...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>t17540116-18</td>
      <td>102. (L.) Michael Riley was indicted for steal...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[L., Michael Riley, John Randall, John Randall...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>t17540116-19</td>
      <td>103. (M.) Edward Allen was indicted for steali...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[Edward Allen, Thomas Rant, William Banes, Tho...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>t17540116-20</td>
      <td>104. (L.) John Skelt was indicted for stealing...</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[L., John Skelt, John Porter, John Porter, Gui...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>t17540116-21</td>
      <td>105. (L.) Anne M'Cormeck , spinster , was indi...</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[L., Anne M'Cormeck, James Bulkley, James Bulk...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>t17540116-22</td>
      <td>106, 107. (M.) Elizabeth Humphrys and Catharin...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Elizabeth Humphrys, Catharine Brown, William ...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>t17540116-23</td>
      <td>108. (M.) Henry Champness , was indicted for s...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>miscPunish</td>
      <td>branding</td>
      <td>[Henry Champness, John Tempest]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>t17540116-24</td>
      <td>109. (M) Thomas Cooke , was indicted for steal...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Thomas Cooke, Godard Williams, Godard William...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>t17540116-25</td>
      <td>110. (M.) Elizabeth Hore , widow , was indicte...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Elizabeth Hore, John Nichols, John Nichols]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>t17540116-26</td>
      <td>111. (M.) Martha Mingest , spinster , was indi...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Martha Mingest, George Cellea, James Bull, Ce...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>t17540116-27</td>
      <td>112, 113. (M.) Anne Purvise , widow , and Fran...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Anne Purvise, Frances Hayes, George Carlow, E...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>t17540116-28</td>
      <td>114. (M.) Sarah Conyers , widow , was indicted...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Sarah Conyers, Isaac Bullock, Isaac Bullock, ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>t17540116-29</td>
      <td>115. (L.) William James was indicted for steal...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[L., William James, hose, John Roberts, John, ...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>t17540116-30</td>
      <td>116. (L.) George Butler was indicted for steal...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[George Butler, Richard Osbourne, Thomas Goodw...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>t17540116-31</td>
      <td>117, 118. (L.) Grace Bunn , widow , and Cathar...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[Grace Bunn, John Bunn, John Howel, John Barne...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>t17540116-32</td>
      <td>119. (L.) Benjamin Ditto was indicted for that...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[L., Benjamin Ditto, Sarah Holt, Mary Merrit, ...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>t17540116-33</td>
      <td>120. (M.) George Cole was indicted for stealin...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[George Cole, Thomas Minett, Thomas Minett, Hi...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>t17540116-34</td>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[Thomas Barnard, Barnett, Boyce Tree, Boyce Tr...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>t17540116-34</td>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[Thomas Barnard, Barnett, Boyce Tree, Boyce Tr...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>t17540116-35</td>
      <td>122. (M.) Daniel Wood , butcher , was indicted...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[John Marsh, John Marsh, thro, I. M., Thomas B...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>t17540116-36</td>
      <td>123. (M.) Anne Jones , widow , was indicted fo...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Anne Jones, Kethurah Vincent, Kethurah Vincen...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>t17540116-37</td>
      <td>124. (M.) Grace Riley , spinster , was indicte...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[Grace Riley, Samuel Collins, Samuel Collins, ...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>t17540116-38</td>
      <td>125, 126. (M.) William Irons , otherwise Isles...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[William Irons, Benjamin Richford, William Bri...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>t17540116-39</td>
      <td>127. (M.) Mary Jones , spinster , was indicted...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>miscPunish</td>
      <td>branding</td>
      <td>[Mary Jones, Mary Casanover]</td>
    </tr>
    <tr>
      <th>41</th>
      <td>t17540116-40</td>
      <td>128. (L.) John Hudson , was indicted for the w...</td>
      <td>kill</td>
      <td>murder</td>
      <td>imprison</td>
      <td>newgate</td>
      <td>[L., John Hudson, Thomas Moss, John Johnson, T...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>t17540116-41</td>
      <td>129. (M.) Joshua Kidden was indicted for that ...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[Joshua Kidden, Mary Jones, Mary Jones, Burry,...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>t17540116-42</td>
      <td>130. (L.) Richard Gandy was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[L., Richard Gandy, Samuel Hall, Samuel Hall, ...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>t17540116-43</td>
      <td>131. (L.) Samuel Witham was indicted for break...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>death</td>
      <td>deathNoDetail</td>
      <td>[L., Samuel Witham, Thomas Upton, Thomas Upton...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>t17540116-44</td>
      <td>132. (M.) John Watson , was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[John Watson, John Campbel, John Campbel, Ratc...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>t17540116-45</td>
      <td>133. (M.) Mary, wife of Joseph Durant , was in...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[Joseph Durant, Francis Platt, Francis Platt, ...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>t17540116-46</td>
      <td>134. (L.) John Stewart , was indicted for stea...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[L., John Stewart, Thomas Mason, Thomas Mason,...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>t17540116-47</td>
      <td>135. (L.) Isaac Angel , was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[L., Isaac Angel, Benjamin Vaughan, Guilty]</td>
    </tr>
    <tr>
      <th>49</th>
      <td>t17540116-48</td>
      <td>136. (L.) Matth.ew Minott , was indicted for s...</td>
      <td>theft</td>
      <td>pettyLarceny</td>
      <td>transport</td>
      <td>transportNoDetail</td>
      <td>[William Whiteman, Guilty]</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
df_text["persons_involved"][3]
```




    ['Sarah Williams',
     'Joseph Smithson',
     'Ann Smithson',
     'Joseph Smithson',
     'Chandos-street',
     'Mary Busbey',
     'Ann Smithson',
     'Burley Hawkins',
     'Ann Smithson',
     'Smithson',
     'Smithson',
     'Prisoner',
     'Welch']



How would I do this for organizations?



# Part III - Introduction to representing and feautirizing text


Counting words is one of the most foundational tasks in NLP. Let us recall how we counted words (using the counter function)


```python
from collections import Counter

# Function to count words in a spacy Doc object and return as a Counter
def word_counter(doc):
    # Get the words from the Doc (ignoring punctuation and spaces)
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    return Counter(words)

word_freq = word_counter(doc)
word_freq

```




    Counter({'the': 20,
             'I': 15,
             'her': 8,
             'and': 8,
             'was': 7,
             'Smithson': 7,
             'a': 7,
             'pot': 7,
             'woman': 5,
             'of': 4,
             'in': 4,
             'with': 4,
             'took': 4,
             "'s": 4,
             'not': 4,
             'prisoner': 4,
             'Ann': 3,
             'to': 3,
             'is': 3,
             'taken': 3,
             'it': 3,
             'after': 3,
             'shop': 3,
             'saw': 3,
             'away': 3,
             'went': 3,
             'months': 3,
             'one': 2,
             'Joseph': 2,
             '+': 2,
             'Round': 2,
             'Court': 2,
             'last': 2,
             'informed': 2,
             'had': 2,
             'from': 2,
             'door': 2,
             'street': 2,
             'she': 2,
             'but': 2,
             'me': 2,
             'Hawkins': 2,
             'up': 2,
             'prosecutor': 2,
             'brought': 2,
             'back': 2,
             'that': 2,
             'Mr.': 2,
             'have': 2,
             'never': 2,
             'about': 2,
             '82': 1,
             'M.': 1,
             'Sarah': 1,
             'Williams': 1,
             'spinster': 1,
             'indicted': 1,
             'for': 1,
             'stealing': 1,
             'brass': 1,
             'kettle': 1,
             'value': 1,
             '10': 1,
             's.': 1,
             'property': 1,
             'Dec.': 1,
             '28': 1,
             'am': 1,
             'wife': 1,
             'we': 1,
             'live': 1,
             'Strand': 1,
             'he': 1,
             'broker': 1,
             'On': 1,
             '27th': 1,
             'December': 1,
             'large': 1,
             'followed': 1,
             'into': 1,
             'Chandos': 1,
             'where': 1,
             'running': 1,
             'along': 1,
             'seeing': 1,
             'come': 1,
             'dropped': 1,
             'returned': 1,
             'home': 1,
             'having': 1,
             'left': 1,
             'no': 1,
             'body': 1,
             'Mary': 1,
             'Busbey': 1,
             'take': 1,
             'run': 1,
             'can': 1,
             'say': 1,
             'did': 1,
             'see': 1,
             'face': 1,
             'told': 1,
             'Burley': 1,
             'at': 1,
             'when': 1,
             'witness': 1,
             'them': 1,
             'drop': 1,
             'Mrs.': 1,
             'which': 1,
             'delivered': 1,
             'constable': 1,
             'who': 1,
             'here': 1,
             'Prisoner': 1,
             'Defence': 1,
             'been': 1,
             'near': 1,
             'these': 1,
             'three': 1,
             'nor': 1,
             'till': 1,
             'To': 1,
             'character': 1,
             'Welch': 1,
             'keep': 1,
             'Bell': 1,
             'Fleet': 1,
             'known': 1,
             'nine': 1,
             'heard': 1,
             'any': 1,
             'thing': 1,
             'ill': 1,
             'She': 1,
             'lived': 1,
             'six': 1,
             'behaved': 1,
             'well': 1,
             'all': 1,
             'time': 1,
             'Guilty': 1,
             'Transportation': 1,
             'See': 1,
             'summary': 1})



Let's reduce our dataset even further to see how text can be "Featurized" or represented using the "CountVectorizer" function (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)


```python
new_df = df_text[["text", "offenceCategory", "offenceSubcategory"]]
```


```python
new_df.head(10)
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
      <th>offenceCategory</th>
      <th>offenceSubcategory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>THE PROCEEDINGS ON THE King's Commissions of t...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80. Hannah Ash , spinster , was indicted for s...</td>
      <td>theft</td>
      <td>grandLarceny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>81. (M.) Peter Foreman and Mary his wife were ...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
    </tr>
    <tr>
      <th>3</th>
      <td>82. (M.) Sarah Williams , spinster , was indic...</td>
      <td>theft</td>
      <td>grandLarceny</td>
    </tr>
    <tr>
      <th>4</th>
      <td>83. (M.) Elizabeth wife of Joseph Kempster , w...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
    </tr>
    <tr>
      <th>5</th>
      <td>84. (M.) John Allen was indicted for stealing ...</td>
      <td>theft</td>
      <td>grandLarceny</td>
    </tr>
    <tr>
      <th>6</th>
      <td>85. (M.) William Derter was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
    </tr>
    <tr>
      <th>7</th>
      <td>86. (M.) William Ford was indicted for stealin...</td>
      <td>theft</td>
      <td>animalTheft</td>
    </tr>
    <tr>
      <th>8</th>
      <td>87. (L.) Anne Beezley , spinster , was indicte...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
    </tr>
    <tr>
      <th>9</th>
      <td>88. Robert Barber was indicted for that he, to...</td>
      <td>deception</td>
      <td>forgery</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer, exclude numbers with a regex pattern
vectorizer = CountVectorizer(ngram_range = (1,2), 
                             token_pattern=r'\b[a-zA-Z]+\b')

# Fit the vectorizer to the text data and transform it into a document-term matrix
X = vectorizer.fit_transform(new_df['text'])

# Convert the document-term matrix to a DataFrame with words as columns
word_counts = pd.DataFrame(X.toarray(), 
                           columns=vectorizer.get_feature_names_out())

# Concatenate the word counts DataFrame with the original DataFrame
df = pd.concat([new_df, word_counts], axis=1)

```


```python
df.shape
```




    (50, 15433)




```python
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
      <th>text</th>
      <th>offenceCategory</th>
      <th>offenceSubcategory</th>
      <th>a</th>
      <th>a a</th>
      <th>a back</th>
      <th>a bad</th>
      <th>a ball</th>
      <th>a band</th>
      <th>a bargain</th>
      <th>...</th>
      <th>your possession</th>
      <th>your tongue</th>
      <th>your watch</th>
      <th>your wife</th>
      <th>your word</th>
      <th>yours</th>
      <th>yours he</th>
      <th>yours stop</th>
      <th>yours thomas</th>
      <th>yours williams</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>THE PROCEEDINGS ON THE King's Commissions of t...</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>1</th>
      <td>80. Hannah Ash , spinster , was indicted for s...</td>
      <td>theft</td>
      <td>grandLarceny</td>
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
      <th>2</th>
      <td>81. (M.) Peter Foreman and Mary his wife were ...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>1</td>
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
      <th>3</th>
      <td>82. (M.) Sarah Williams , spinster , was indic...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>7</td>
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
      <td>83. (M.) Elizabeth wife of Joseph Kempster , w...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>3</td>
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
      <th>5</th>
      <td>84. (M.) John Allen was indicted for stealing ...</td>
      <td>theft</td>
      <td>grandLarceny</td>
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
      <th>6</th>
      <td>85. (M.) William Derter was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
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
      <th>7</th>
      <td>86. (M.) William Ford was indicted for stealin...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>13</td>
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
      <th>8</th>
      <td>87. (L.) Anne Beezley , spinster , was indicte...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>4</td>
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
      <th>9</th>
      <td>88. Robert Barber was indicted for that he, to...</td>
      <td>deception</td>
      <td>forgery</td>
      <td>3</td>
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
      <th>10</th>
      <td>89, 90. (M.) Elizabeth Eaton and Catherine Dav...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>6</td>
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
      <th>11</th>
      <td>91. (M.) Samuel Portman was indicted for the w...</td>
      <td>kill</td>
      <td>murder</td>
      <td>45</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>92. (L.) Anne Ashley , widow , was indicted fo...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>6</td>
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
      <th>13</th>
      <td>93. (M.) Sarah, wife of Charles Griffice , was...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>5</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>91. (M) Elizabeth Pettit , was indicted for st...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>4</td>
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
      <th>15</th>
      <td>95. (M.) Sarah Barefoot , spinster , was indic...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>12</td>
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
      <th>16</th>
      <td>96, 97, 98, 99, 100. (M ) Thomas Radborn , oth...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>9</td>
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
      <th>17</th>
      <td>101. (L.) Thomas Waters was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>5</td>
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
      <th>18</th>
      <td>102. (L.) Michael Riley was indicted for steal...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>11</td>
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
      <th>19</th>
      <td>103. (M.) Edward Allen was indicted for steali...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>13</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>104. (L.) John Skelt was indicted for stealing...</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>1</td>
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
      <th>21</th>
      <td>105. (L.) Anne M'Cormeck , spinster , was indi...</td>
      <td>theft</td>
      <td>theftOther</td>
      <td>2</td>
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
      <th>22</th>
      <td>106, 107. (M.) Elizabeth Humphrys and Catharin...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>17</td>
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
      <th>23</th>
      <td>108. (M.) Henry Champness , was indicted for s...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>3</td>
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
      <th>24</th>
      <td>109. (M) Thomas Cooke , was indicted for steal...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>22</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>110. (M.) Elizabeth Hore , widow , was indicte...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>3</td>
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
      <th>26</th>
      <td>111. (M.) Martha Mingest , spinster , was indi...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>3</td>
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
      <th>27</th>
      <td>112, 113. (M.) Anne Purvise , widow , and Fran...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>2</td>
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
      <th>28</th>
      <td>114. (M.) Sarah Conyers , widow , was indicted...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
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
      <th>29</th>
      <td>115. (L.) William James was indicted for steal...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>15</td>
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
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>116. (L.) George Butler was indicted for steal...</td>
      <td>theft</td>
      <td>theftFromPlace</td>
      <td>3</td>
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
      <th>31</th>
      <td>117, 118. (L.) Grace Bunn , widow , and Cathar...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>23</td>
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
      <th>32</th>
      <td>119. (L.) Benjamin Ditto was indicted for that...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>30</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>120. (M.) George Cole was indicted for stealin...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>35</th>
      <td>121. (L.) Thomas Barnard , otherwise Barnett ,...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>36</th>
      <td>122. (M.) Daniel Wood , butcher , was indicted...</td>
      <td>theft</td>
      <td>animalTheft</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
      <th>37</th>
      <td>123. (M.) Anne Jones , widow , was indicted fo...</td>
      <td>theft</td>
      <td>shoplifting</td>
      <td>16</td>
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
      <th>38</th>
      <td>124. (M.) Grace Riley , spinster , was indicte...</td>
      <td>theft</td>
      <td>pocketpicking</td>
      <td>11</td>
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
    <tr>
      <th>39</th>
      <td>125, 126. (M.) William Irons , otherwise Isles...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>9</td>
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
      <th>40</th>
      <td>127. (M.) Mary Jones , spinster , was indicted...</td>
      <td>theft</td>
      <td>grandLarceny</td>
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
      <th>41</th>
      <td>128. (L.) John Hudson , was indicted for the w...</td>
      <td>kill</td>
      <td>murder</td>
      <td>54</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>129. (M.) Joshua Kidden was indicted for that ...</td>
      <td>violentTheft</td>
      <td>highwayRobbery</td>
      <td>50</td>
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
      <th>43</th>
      <td>130. (L.) Richard Gandy was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>7</td>
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
      <th>44</th>
      <td>131. (L.) Samuel Witham was indicted for break...</td>
      <td>theft</td>
      <td>burglary</td>
      <td>30</td>
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
      <th>45</th>
      <td>132. (M.) John Watson , was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>8</td>
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
      <th>46</th>
      <td>133. (M.) Mary, wife of Joseph Durant , was in...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>8</td>
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
      <th>47</th>
      <td>134. (L.) John Stewart , was indicted for stea...</td>
      <td>theft</td>
      <td>grandLarceny</td>
      <td>7</td>
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
      <th>48</th>
      <td>135. (L.) Isaac Angel , was indicted for steal...</td>
      <td>theft</td>
      <td>grandLarceny</td>
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
      <th>49</th>
      <td>136. (L.) Matth.ew Minott , was indicted for s...</td>
      <td>theft</td>
      <td>pettyLarceny</td>
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
  </tbody>
</table>
<p>50 rows × 15433 columns</p>
</div>



This is known as a "Document Term Matrix" (more on this next week). Each row represents a document, and each column represents a word (ngram, bigram, etc).

Document Term Matrices (DTMs) are an important foundation to Machine Learning based applications of NLP. 

Notice how we simply counted words `count` vectorizer - but now we have a very convenient way of mathematically examining documents and corpora very efficiently.

Notice also that changing the ngram_range considerably changes the `dimensionality` or `features` of the documents.




```python
df.shape
```




    (50, 15433)



We now magically transformed our words into numbers. It's that simple - crazy!

Let's look at two columns, the words "the" and "pocket"


```python
df[['the', 'pocket']]
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
      <th>the</th>
      <th>pocket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>104</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>26</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>45</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>52</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>34</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>85</td>
      <td>3</td>
    </tr>
    <tr>
      <th>33</th>
      <td>70</td>
      <td>2</td>
    </tr>
    <tr>
      <th>34</th>
      <td>264</td>
      <td>3</td>
    </tr>
    <tr>
      <th>35</th>
      <td>264</td>
      <td>3</td>
    </tr>
    <tr>
      <th>36</th>
      <td>80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>44</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>153</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>68</td>
      <td>3</td>
    </tr>
    <tr>
      <th>43</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>44</td>
      <td>3</td>
    </tr>
    <tr>
      <th>45</th>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>32</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The word "pocket" seems to be used a lot in the document number 21. Let's read it.


```python
df['text'][21]
```




    "105. (L.) Anne M'Cormeck , spinster , was indicted for stealing one linen handkerchief , the property of James Bulkley , Dec. 12 . ++ James Bulkley . I was coming along Fleet-street about the 12th of December last, and a gentleman's servant asking me if I had lost my handkerchief, I answered, that I had. He shewed me the person, who is the prisoner at the bar. I took her before Justice Fielding, where she was searched, and the handkerchief found under her arm. Produced in court and deposed to. She said she did not know whether it was mine or not. Prisoner's Defence. I picked it off the ground, tho' the footman said I took it out of his pocket. Q. Did you feel any body pick your pocket? Bulkley. I did not feel any hand in my pocket. I was looking in at a print shop; the footman declared he saw her pick my pocket, and was told that he need not appear here, because the handkerchief was found upon her. Guilty . [Transportation. See summary.]"




```python

```

The words "the" and "pocket" are actually no longer simple words. They can be thought of as "vectors" - because if we look at the column it becomes a set of numbers. Plotting this in a space can actually give us some geometric intuitions... 

* See Jurafsky Martin Chapter 6 for more details. https://web.stanford.edu/~jurafsky/slp3/6.pdf
* We will cover more of this next class, but for now, just start thinking of text as matrices.

For example, we could compare documents where words would be vector coordinates identifying documents?


```python
subset_df = df.loc[21:22, ["text", "the", "pocket"]]
subset_df
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
      <th>the</th>
      <th>pocket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>105. (L.) Anne M'Cormeck , spinster , was indi...</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>106, 107. (M.) Elizabeth Humphrys and Catharin...</td>
      <td>52</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
!pip install -q bokeh
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m26.0[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, PointDrawTool, LabelSet, CustomJS

# 1. Setup Data with TWO default vectors
source = ColumnDataSource(data=dict(
    x=[0, 0], 
    y=[0, 0], 
    xe=[52, 10], 
    ye=[2, 4], 
    text=["(52.0, 2.0)", "(10.0, 4.0)"]
))

# 2. Zoomed-out plot range
p = figure(
    x_range=(-10, 60), 
    y_range=(-10, 20),
    width=700, 
    height=500
)

# 3. Draw Vectors
p.segment(
    x0='x', y0='y', 
    x1='xe', y1='ye', 
    source=source, 
    line_width=4, 
    line_alpha=0.6
)

# 4. Permanent Labels
p.add_layout(LabelSet(
    x='xe', 
    y='ye', 
    text='text', 
    source=source, 
    x_offset=10
))

# 5. Draggable Handles
render = p.scatter(
    'xe', 'ye', 
    source=source, 
    size=15, 
    color="red", 
    marker="circle"
)

# 6. JS Callback
source.js_on_change('data', CustomJS(args=dict(s=source), code="""
    const d = s.data;
    for (let i = 0; i < d['xe'].length; i++) {
        d['x'][i] = 0;
        d['y'][i] = 0;
        d['text'][i] = `(${d['xe'][i].toFixed(1)}, ${d['ye'][i].toFixed(1)})`;
    }
    s.change.emit();
"""))

# 7. Draw Tool
draw_tool = PointDrawTool(renderers=[render])
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool

show(p)

```

If we add more words (for example 3 words), we would have a richer space to compare?


```python
subset_df = df.loc[21:22, ["text", "the", "pocket", "spinster"]]
subset_df
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
      <th>the</th>
      <th>pocket</th>
      <th>spinster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>105. (L.) Anne M'Cormeck , spinster , was indi...</td>
      <td>10</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>106, 107. (M.) Elizabeth Humphrys and Catharin...</td>
      <td>52</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
!pip install -q ipympl
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m26.0[0m[39;49m -> [0m[32;49m26.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
# %matplotlib ipympl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 1. Setup Figure
plt.close('all')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 2. Define vectors
vectors = np.array([
    [10, 4, 1],
    [52, 2, 0]
])

# 3. Draw the axes
ax.quiver(-10, 0, 0, 20, 0, 0, color='red', arrow_length_ratio=0.05, alpha=0.3)   # X
ax.quiver(0, -10, 0, 0, 20, 0, color='green', arrow_length_ratio=0.05, alpha=0.3) # Y
ax.quiver(0, 0, -10, 0, 0, 20, color='blue', arrow_length_ratio=0.05, alpha=0.3)  # Z

# 4. Draw vectors
for vec in vectors:
    x, y, z = vec
    ax.quiver(0, 0, 0, x, y, z, color='black', linewidth=2, arrow_length_ratio=0.05)
    ax.text(x, y, z, f"({x}, {y}, {z})", fontsize=12, fontweight='bold')

# 5. Set limits
ax.set_xlim(-5, 60)
ax.set_ylim(-5, 20)
ax.set_zlim(-5, 5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Vectors: (10,4,1) and (52,2,0)")

plt.show()

```

## Bibliography
 - Based on Applied Language Technology course github page from University of Helsinki: https://github.com/Applied-Language-Technology

## XML part 
 - Based on LS1
 - 23 XML parsing lab: https://github.com/ds-modules/Legalst-123/tree/master/labs
 - All files from Old Bailey API - https://www.oldbaileyonline.org/obapi/
 - ElementTree information adapted from Driscoll, Mike. (2013, April). Python 101 – Intro to XML Parsing with ElementTree.
 https://www.blog.pythonlibrary.org/2013/04/30/python-101-intro-to-xml-parsing-with-elementtree/

 - Web Scraping code adapted from MEDST-250 Notebook developed by Tejas Priyadarshan.
 https://github.com/ds-modules/MEDST-250/tree/master/04%20-%20XML_Day_1
 
 - Image source from https://www.researchgate.net/publication/257631377_Efficient_XML_Path_Filtering_Using_GPUs

Notebook developed by: Jason Jiang, Iland Leigh, Violet Yao, and Wilson Berkow; adjustment to new Old Bailey Online API by Jon Marshall 2024. 

Data Science Modules: http://data.berkeley.edu/education/modules
