# Lab 3 

 __case.law API, word frequencies, regex, concordances and collocations, mult-word phrases, introduction to NLP tasks (part of speech tagging)__

In this lab, we'll be going over how to access the case.law data using an API. We'll also examine the dataset, introduce Regular Expressions, and go over concordances and collocations and other means of finding multi-word expressions.


# Introduction to the large datasets Datasets

We'll be working with parts of the **[case.law](https://case.law/)** dataset, particularly the Illinois data, since it's easy to access it. Case law is  a database of 360 years of United States caselaw. 

* The overwhelming amounts of data has made **APIs and API KEYS** an important means of accessing data. For example, there's the **[Twitter API](https://developer.twitter.com/en/docs/twitter-api)** if you want to study tweets. Or **[NYTimes API](https://developer.nytimes.com/apis)** if you want to study the New York Times Archive.
(API is an acronym for **Application Programming Interface.**  If you ask me, that's seems like a pretty vague and general term (unless you are a CS person who can explain what this means). The term itself most likely comes from the early days of computing. )


* Furthermore, tools like **[Common Crawler](https://commoncrawl.org/)** are used throughout NLP research to collect data off the internet. 

The API for case.law is very well-documented and you can find examples of how to use the API by following the various **[jupyter notebooks they provided by the case.law team](https://github.com/harvard-lil/cap-examples)**. These notebooks are excellent - __so check them out!__

# Getting the data and putting it into a pandas dataframe

Go to the files on Canvas and download the zip file. For this lab we will examine cases for Illinois between 1771 - 2011.

The dataset is stored as a [jsonl](https://jsonlines.org/) file in a zip file - which stands for "json lines." A .JSON is itself a dictionary, and "lines" stands for the fact that each entry in the data is stored as a line. 

The above code below sure that we can load the json file.


```python
import pandas as pd
import zipfile
import lzma
from pathlib import Path
from itertools import islice
import json

rows = []
with zipfile.ZipFile("data/text.data.jsonl.xz.zip") as z:
    xz_name = z.namelist()[0]
    with z.open(xz_name) as xz_fp, lzma.open(xz_fp, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 1000:
                break
            rows.append(json.loads(line))

df = pd.DataFrame(rows)
```


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
      <th>id</th>
      <th>name</th>
      <th>name_abbreviation</th>
      <th>decision_date</th>
      <th>docket_number</th>
      <th>first_page</th>
      <th>last_page</th>
      <th>citations</th>
      <th>volume</th>
      <th>reporter</th>
      <th>court</th>
      <th>jurisdiction</th>
      <th>casebody</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2747110</td>
      <td>The People of the State of Illinois, Plaintiff...</td>
      <td>People v. Tobin</td>
      <td>1771-10-12</td>
      <td>No. 70-17</td>
      <td>538</td>
      <td>543</td>
      <td>[{'type': 'official', 'cite': '2 Ill. App. 3d ...</td>
      <td>{'volume_number': '2'}</td>
      <td>{'full_name': 'Illinois Appellate Court Report...</td>
      <td>{'id': 8837, 'name': 'Illinois Appellate Court...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>435537</td>
      <td>James A. Whitesides and others, Plaintiffs in ...</td>
      <td>Whitesides v. People</td>
      <td>1819-12</td>
      <td></td>
      <td>21</td>
      <td>22</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 21'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>435638</td>
      <td>Amos Chipps, Appellant, v. Thomas Yancey, Appe...</td>
      <td>Chipps v. Yancey</td>
      <td>1819-12</td>
      <td></td>
      <td>19</td>
      <td>19</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 19'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>435690</td>
      <td>Jonathan Taylor, Appellant, v. Michael Sprinkl...</td>
      <td>Taylor v. Sprinkle</td>
      <td>1819-12</td>
      <td></td>
      <td>17</td>
      <td>18</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 17'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>435710</td>
      <td>François Coleen and Abraham Claypole, Appellan...</td>
      <td>Coleen v. Figgins</td>
      <td>1819-12</td>
      <td></td>
      <td>19</td>
      <td>20</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 19'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
    </tr>
  </tbody>
</table>
</div>



We now have a __pandas dataframe__ of all the cases. 

Note, however, that we don't really see the "text" of the decisions. The actual **text** is contained within a dictionary in the column named **casebody.** So it's a dictionary within a dictionary - these data structures can get pretty complicated!

# Getting to the text

Let's examine the data structure of **"casebody"** a little bit further.

This is a good opportunity to go over the basics of data structures - namely lists and dictionaries.


```python
df['casebody'][0]
```




    {'status': 'ok',
     'data': {'opinions': [{'type': 'majority',
        'text': 'Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered the opinion of the court:\nDefendant Tobin was convicted by a jury of the crime of burglary. The court entered judgment upon the verdict and sentenced the defendant to a fifteen to twenty-five year term in the Illinois State Penitentiary. The judgment of the court further provided that the mittimus was to be effective upon release by federal authorities.\nThe defendant has appealed from that judgment and raised the following issues: (1) The State failed to prove lack of authority to enter the premises; (2) The State failed to prove intent to commit a theft; (3) The court erred in allowing testimony concerning the arrest of Sherri Tobin, her possession of a firearm and evidence concerning defendant’s possession of a firearm; (4) The sentence was excessive.\nThe facts giving rise to this case are as follows: On the night of February 9, 1969, at about 11:00 P.M. the defendant, in the company of Sherri Tobin, Daniel Stout, Michael Hume and Eddie Dunn was in an automobile driven by defendant in the vicinity of the Oliver C. Joseph Automobile Agency in Belleville. Their behavior while driving was observed by James Muir who resided nearby. He stated the car stopped by the agency and the driver, identified as Tobin, jumped out and ran across the street and kicked the agency door. The car in the meantime circled the block and picked Tobin up. The car drove away and Muir next observed four men walking up the street to the agency. Muir recognized one of the four as the man who kicked the door. The four men then entered the agency building by the same door previously kicked. Muir then called the police.\nThe police arrived and Officers Rettle and Wobbe observed four men walking through the building. Two other policemen arrived. Rettle observed two of the suspects at the back door. He identified himself and ordered them out. They disappeared back inside the building. The police entered the building and found one suspect lying under a car, After turning on the lights they searched the building. The other suspects were found in the basement. The defendant was hiding behind an air compressor when discovered.\nAfter apprehending all four men the police with Mr. Muir’s help located the car a few blocks away. Sherri Tobin was found in the car asleep. She was carrying a “.38 caliber snub-nosed revolver, fully loaded, with the serial numbers filed off” in the waist band of her slacks. One of the defendants, Hume, testified for the State and said defendant stated earlier in the evening that they would go to Belleville and “make some money”. He also stated that defendant brought a gun into the building.\nThe evidence also showed that the door jamb of the door kicked by defendant was splintered and the door opened by force.\nAdditionally, Mr. Oliver P. Joseph testified that the building was owned by and in the possession of Oliver C. Joseph, Inc., a corporation, engaged in the selling of automobiles.\nAs to the authority to be in the premises, “* * * the law presumes that the presence in a public building for a purpose inconsistent with the purposes for which the building is open to the public is without authority”. (People v. Urban (1971), (Ill.App.2d), 266 N.E.2d 112, 114. Also see People v. Weaver (1968), 41 Ill.2d 434, 243 N.E.2d 245 cert. den. 395 U.S. 959, 89 S.Ct. 2100, 23 L.Ed.2d 746.) Here the defendant (a) had to break open a door to gain admission, (b) at 11:00 P.M., (c) when the buffding was unlit, and (d) hid in the basement upon arrival of the police. Under these circumstances there was sufficient evidence for the jury to believe that his presence was without authority.\nIn regard to the question of intent there is also sufficient circumstantial evidence for the jury to believe that Tobin intended to commit a theft in the building. Intent must ordinarily be proved circumstantially, by inferences drawn from conduct appraised in its factual environment.” (People v. Johnson (1963), 28 Ill.2d 441, 192 N.E.2d 864, 866.) The Court in Johnson went on to say:\n“* * * We are of the opinion that in the absence of inconsistent circumstances, proof of unlawful breaking and entry into a building which contains personal property that could be the subject of larceny gives rise to an inference that will sustain a conviction of burglary. Like other inferences, this one is grounded in human experience, which justifies the assumption that the unlawful entry was not purposeless, and, in the absence of other proof, indicates theft is the most likely purpose.”\nThe circumstances of the entry coupled with Hume’s testimony that Tobin intended to “make some money” in BeUeviHe is sufficient evidence of intent to commit a theft.\nThe third aUeged error relates to Sherri Tobin’s arrest and the question of firearms. It is contended that the evidence of Sherri’s arrest is not only irrelevant but introduced solely for the purpose of bringing in evidence of the .38 cafibre gun to prejudice the jury. In regard to Tobin’s possession of a firearm the only evidence of this is Hume’s testimony. No gun was introduced into evidence.\nWhile the evidence of Sherri’s arrest may not have been essential to the conviction of Tobin, it apprised the finder of fact of the total circumstances surrounding the event and at worst, it is harmless error. The defendant has aHeged prejudice but a careful search of the record discloses none. Such evidence did not prove an element of the crime not estabfished by other properly admitted evidence. People v. Landgham (1970), 122 Ill.App.2d 9, 275 N.E.2d 484; People v. Jones, (1970), 125 Ill.App.2d 30, 259 N.E.2d 585.\nDefendant argues that his sentence was excessive because he received a heavier sentence than his co-defendants who pleaded guilty to the same offense. The record shows that Eddie Dunn and Michael Hume were each placed on probation for a period of five years. Daniel Stout was sentenced to not less than five nor more than ten years. Defendant asserts that he was penalized for having exercised his constitutional right to a trial by jury.\nThe basic principles regarding sentencing are set forth in People v. Jones, 118 Ill.App.2d 189, 254 N.E.2d 843, 847:\n“We recognize that not every offense in a like category calls for an identical punishment. There may be a proper variation in sentences as between different offenders, depending upon the circumstances of the individual case. As a general rule, where the punishment for the offense is fixed by statute, that imposed in the sentence must conform thereto, and a sentence which conforms to statutory regulation is proper. Before an AppeUate Court wiU interfere, it must be manifest from the record that the sentence is excessive and not justified by any reasonable view which might be taken of the record. People v. Hobbs, 58 Ill.App.2d 93, 99, 205 N.E.2d 503 (1965). Disparity of sentences between defendants does not, of itself, warrant the use of the power to reduce a punishment imposed by the trial court. People v. Thompson, 36 Ill.2d 478, 482, 224 N.E.2d 264 (1967).”\nHume had no prior criminal record. The records of the other defendants do not appear in the record on this appeal. Defendant Tobin had a prior burglary conviction as a juvenile on which he served a year, and contrary to the provision of his pretrial bail, he left the State. While gone he became involved with federal authorities as evidenced by the fact that his presence at trial was secured by virtue of a Writ of Habeas Corpus Ad Prosequendum. Tobin was also the apparent ringleader of the burglary. He drove the car, planned the burglary, selected the site and broke open the door. Under the circumstances, we find the factual situation to differ from People v. Jones, supra, on which defendant relies, and a penalty greater than that imposed upon defendant’s accomplices is approved.\nHowever, we do not consider the possibility of rehabilitation to be so remote as to justify a sentence of 15 to 25 years, which as a practical matter leaves little or no room for rehabilitation; nor does such sentence provide for an exercise of the discretion of parole authorities at a time when such discretion may contribute most to rehabilitation. As a result we would consider that the sentence should be modified to provide a minimum of seven years and a maximum of 20 years.\nLastly, defendant claims the court erred in making his sentence consecutive to a possible future federal sentence. We agree. However, the remedy is not necessarily to make the sentence concurrent with a possible future federal sentence as argued by defendant. See Ill. Rev. Stat. ch. 38, pars. 119 — 1 and 119 — 2.\nThe language “Mittimus to be effective upon release by federal authorities” is too broad and does not clearly define what sentence the imposed sentence is to follow. See Ill. Rev. Stat. ch. 38, par. 7 — 1. We would particularly call attention to the fact that ch. 38, par. 7 — l(n) makes provision only for a concurrent sentence with a “previous and unexpired sentence” imposed by a Federal District Court. During the oral argument in June 1971, this Court was advised that the defendant’s involvement with federal authorities to which we have referred is that depicted by United States v. Tobin, 426 Fed.2d 1279, in which defendant’s conviction was reversed and the cause remanded for a new trial in May 1970, and that defendant had not been reprosecuted in that cause.\n“A sentence should be so complete as not to require construction by the court to ascertain its import, and so complete that it will not be necessary for a nonjudicial or ministerial officer to supplement the written words to ascertain its meaning.” People v. Walton (1969), 118 Ill.App.2d 324, 254 N.E.2d 190, 194 — 5.\nThe judgment of conviction is affirmed, the sentence is modified to provide a minimum sentence of seven years and a maximum sentence of twenty years;\nJudgment with sentence so modified is affirmed.\nMORAN and CREES, JJ., concur.',
        'author': 'Mr. PRESIDING JUSTICE EBERSPACHER'}],
      'parties': ['The People of the State of Illinois, Plaintiff-Appellee, v. Danny Tobin, Defendant-Appellant.'],
      'attorneys': ['John D. Shulleriberger, Morton Zwick, Director of Defender Project, of Chicago, (Matthew J. Moran, and Norman W. Fishman, of Defender Project, of counsel,) for appellant.',
       'Robert H. Rice, State’s Attorney, of Belleville, for the People.'],
      'head_matter': 'Fifth District\n(No. 70-17;\nThe People of the State of Illinois, Plaintiff-Appellee, v. Danny Tobin, Defendant-Appellant.\n— October 12, 1771.\nRehearing denied December 9,1971.\nJohn D. Shulleriberger, Morton Zwick, Director of Defender Project, of Chicago, (Matthew J. Moran, and Norman W. Fishman, of Defender Project, of counsel,) for appellant.\nRobert H. Rice, State’s Attorney, of Belleville, for the People.',
      'judges': []}}



We can write for loop to extract the text in the casebody column:


```python
opinion_texts = [] ## create an emtpy list for just the text

for i in range(len(df)):
    if df['casebody'][i]['data']['opinions']:
        text = df['casebody'][i]['data']['opinions'][0]['text'] # you can add .lower() to lowercase
        opinion_texts.append(text)
    else:
        opinion_texts.append("No Text Found") ## If no text is found, have a "NAN" entry - eg. df.loc[df['text'] == 'No Text Found']
    
```

How would you ask ChatGPT to write this for loop for you? 


```python
print(opinion_texts[4])
```

    Opinion of the Court.
    †
    It appears from the record in this cause, that the writ issued by the Madison circuit court, on the 31st day of March, 1819, and made returnable to May term following, and that the act creating circuit courts, passed on the same day the writ issued. Although it appears, that the act establishing circuit courts, passed on the 31st day of March, yet the court are clearly of opinion, that it did not take effect until the first day of April, and that the process is therefore void, as the clerk had no authority to issue the writ, and make it returnable to a court not in existence, at the time the writ issued. No appearance could make the writ good. The court below was bound to have quashed it, it differing materially, from process that is voidable merely, where appearing and pleading might cure the defect.
    It is unnecessary for the court, to notice any other error assigned, as the point already decided, determines the case.
    The judgment of the court is reversed,
    (a)
    (1)
    Judgment reversed.
    † Justice Beynolds having decided this cause in the court below, gave no opinion.
    (a) An appearance of the defendant by attorney, cures any antecedent irregularity of process. Knox et al. v. Summers et al., 3 Cranch, 496.
    Process returnable out of term is void, and can not be amended. Cramer v. Van Alstyne, 9 Johns., 386.
    (1) It can hardly admit of a doubt that an appearance cures all defects as to the manner in which a party is brought into court. If a party, without process, pleads to an action, it is too late for him then to say that no process was issued or served on him. He is then in court, and it is immaterial whether he appears in compliance with the mandates of the law, or whether he waives a right which he might have insisted on, and voluntarily places himself in a position in which he is required to make bis defense. The decisions on this question are uniform. In Easton et al. v. Altum, 1 Scam., 250, the court said: “ The authorities are numerous and explicit, that irregularity of process, whether the process be void or voidable, is cured by appearance without objection.” And in Mitchell v. Jacobs et al., 17 Ills. Rep., 236 : “A defendant appearing without objection waives all objections thereto, although the process may be void, or there may have been no service.” To the same effect is Mineral Point R. R. Co. v. Keep, 22 Ills. Rep., 9. The following cases have also been passed upon by the Supreme Court of this State, in each of which this question arose, and received substantially the same solution. Pearce et al. v. Swan, 1 Scam., 269. Vance et al. v. Funk, 2 Scam., 263. Beecher et al. v. James et al. id., 463. Palmer v. Logan, 3 Scam., 57. Bowles’ heirs v. Rouse, adm’r., 3 Gilm., 409. Whittaker et al. v. Murray et al., 15 Ills. R., 294.
    Although a general appearance will cure all irregularities as to the issuing or service of process, yet an appearance for the purpose of objecting to such process or service will not have that effect. Mitchell v. Jacobs et al., 17 Ills. R., 236. Anglin v. Nott, 1 Scam., 395. Little v. Carlisle et al., 2 Scam., 376.


Let's reinsert the "opinion_texts" list into the dataframe under the column __'text'__


```python
df['text'] = opinion_texts
```


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
      <th>id</th>
      <th>name</th>
      <th>name_abbreviation</th>
      <th>decision_date</th>
      <th>docket_number</th>
      <th>first_page</th>
      <th>last_page</th>
      <th>citations</th>
      <th>volume</th>
      <th>reporter</th>
      <th>court</th>
      <th>jurisdiction</th>
      <th>casebody</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2747110</td>
      <td>The People of the State of Illinois, Plaintiff...</td>
      <td>People v. Tobin</td>
      <td>1771-10-12</td>
      <td>No. 70-17</td>
      <td>538</td>
      <td>543</td>
      <td>[{'type': 'official', 'cite': '2 Ill. App. 3d ...</td>
      <td>{'volume_number': '2'}</td>
      <td>{'full_name': 'Illinois Appellate Court Report...</td>
      <td>{'id': 8837, 'name': 'Illinois Appellate Court...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>435537</td>
      <td>James A. Whitesides and others, Plaintiffs in ...</td>
      <td>Whitesides v. People</td>
      <td>1819-12</td>
      <td></td>
      <td>21</td>
      <td>22</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 21'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>435638</td>
      <td>Amos Chipps, Appellant, v. Thomas Yancey, Appe...</td>
      <td>Chipps v. Yancey</td>
      <td>1819-12</td>
      <td></td>
      <td>19</td>
      <td>19</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 19'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>435690</td>
      <td>Jonathan Taylor, Appellant, v. Michael Sprinkl...</td>
      <td>Taylor v. Sprinkle</td>
      <td>1819-12</td>
      <td></td>
      <td>17</td>
      <td>18</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 17'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>435710</td>
      <td>François Coleen and Abraham Claypole, Appellan...</td>
      <td>Coleen v. Figgins</td>
      <td>1819-12</td>
      <td></td>
      <td>19</td>
      <td>20</td>
      <td>[{'type': 'official', 'cite': '1 Ill. 19'}]</td>
      <td>{'volume_number': '1'}</td>
      <td>{'full_name': 'Illinois Reports'}</td>
      <td>{'id': 8853, 'name': 'Illinois Supreme Court',...</td>
      <td>{'id': 29, 'slug': 'ill', 'name': 'Ill.', 'nam...</td>
      <td>{'status': 'ok', 'data': {'opinions': [{'type'...</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
    </tr>
  </tbody>
</table>
</div>



Let's drop the columns which contain the **metadata** - ie data about data (like for example "last page") - and don't seem important at the moment (since we care only about text).


```python
df_cleaned = df[['decision_date', 'name_abbreviation', 'text']] ## keep only these columns
df_cleaned
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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1845-12</td>
      <td>Thomas v. Negus</td>
      <td>"The Opinion of the Court was delivered'by\nTr...</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1845-12</td>
      <td>Owen v. Barnum</td>
      <td>The Opinion of the Court was delivered by\nPur...</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1845-12</td>
      <td>Murphy v. Summerville</td>
      <td>The Opinion of the Court was delivered by\nSca...</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1845-12</td>
      <td>Anderson v. Semple</td>
      <td>The Opinion of the Court was delivered by\nSca...</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1845-12</td>
      <td>Scott v. Crow</td>
      <td>The Opinion of the Court was delivered by\nTre...</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 3 columns</p>
</div>



# Word frequency over time

Below, we'll be using code from the [case.law API example codes](https://github.com/harvard-lil/cap-examples/blob/develop/ngrams/ngrams.ipynb) on n-grams.

Now that we have the data - let's try looking at simple word frequencies over time. If you are familiar with "google ngram viewer" this will look be easily understandable. 

First, let's convert our "Decision Date" column into **datetime format**. 

"Datetime" Format allows us to work with "numbers" as datetime objects - ie, their months and years and days. This is convenient because dates are not like normal numbers - usually months end at 30, and restart to 1. So it's obvious that it needs its own heuristic.


```python
df_cleaned['decision_date'] = pd.to_datetime(df_cleaned["decision_date"],
                                            format = "mixed")
```

    /tmp/ipykernel_29263/1604096991.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_cleaned['decision_date'] = pd.to_datetime(df_cleaned["decision_date"],


Now let's extract the year from our newly converted datetime column. Looking at "words over year" is a good simple way of seeing __general trends__ in the law and legal language.


```python
df_cleaned['year'] = df_cleaned['decision_date'].dt.year
```

    /tmp/ipykernel_29263/1144164058.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_cleaned['year'] = df_cleaned['decision_date'].dt.year



```python
df_cleaned.head()
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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
    </tr>
  </tbody>
</table>
</div>



Let's define a function called "search_ngram" which counts all the **occurrences** (or frequencies) of a given word over a given year. Thus, for example, if we cared about the word "robbery", how many times did it appear over time in the Illinois dataset, and what were the trends of this word over time?


```python

```


```python
def search_ngram(ngram):
    pairs = []
    for year in df_cleaned["year"].unique():                           ## list all unique years
        temp = df_cleaned[df_cleaned["year"] == year]["text"].tolist() ## extract all the text for a given year
        temp = " ".join(temp).lower()                                  ## make into a string via .join and lowercase
        total_number_of_words = len(temp.split(" "))                   ## count the tokenized words - use for relative frequency
        ngram_count = temp.count(ngram.lower())
        pairs.append((year, 
                      ngram_count/total_number_of_words))              ## normalize ngram count by total word count
   
    return pd.DataFrame(pairs, columns=['Year', 'Normalized Frequency'])



```

Let's see what the function does:


```python
robbery = search_ngram("robbery")
robbery
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
      <th>Year</th>
      <th>Normalized Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1820</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1822</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1823</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1824</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1825</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1826</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1827</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1828</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1829</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1830</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1831</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1832</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1833</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1834</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1835</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1836</td>
      <td>0.000044</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1837</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1838</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1839</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1840</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1841</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1842</td>
      <td>0.000132</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1843</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1844</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1845</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Plotting

Now let's plot our results:


```python
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize = (15,8))
sns.lineplot(data = robbery, 
             x = "Year", 
             y = "Normalized Frequency")
```




    <Axes: xlabel='Year', ylabel='Normalized Frequency'>



Let's try something else. Since the output of the search_ngram() function is a Pandas Dataframe, we can explicitly put it in plotting code below. 

A good example is the word "computer". The word itself never existed before computers were invented, so if we don't see its occurrence in the past, we can conclude that the code is capturing trends in the text data. This  can be thought of as a __"sanity check"__ - ie a simple hypothesis that we all know to be true that is captured by the code.


```python
plt.figure(figsize = (15,8))
sns.lineplot(data = search_ngram('Computer'), 
             x = "Year", 
             y = "Normalized Frequency")
```




    <Axes: xlabel='Year', ylabel='Normalized Frequency'>



You can explore the data further here if you'd like or draw some more plots. 

**Note:** Again - I encourage you to check out the case.law **[example notebooks](https://github.com/harvard-lil/cap-examples)**. For instance, the **[Cartwright notebook - which shows who was Illinois' most prolific judge](https://github.com/harvard-lil/cap-examples/blob/develop/bulk_exploration/cartwright.ipynb)** is very good!

#   Introduction to  Regular Expressions (ReGex)

As we saw above, despite their simplicity, word frequencies are a very powerful tool for examining trends in text data. This simplicity has the added benefit of being __intuitive in interpretation__

Nevertheless, there are other methods we can use to examine text. The most (in)famous way of searching for patterns in text is known as REGEX, or regular expressions. 

This can be especially useful if we want to find or even remove some "bad patterns" - boilerplate text, uninformative headers or footers, etc. 

**Sets, Quantifiers, and Special Characters**

One of the best ways to learn Regex is by using Regex 101 to practice matching words in a body of text.

[RegexR](https://regexr.com/)

[Regex101](https://regex101.com/)

[Regex Reference Sheet](http://www.rexegg.com/regex-quickstart.html#ref)

For example, say we had a text and we wanted to find every instance of a word within that text.


```python
import regex as re
text = "Samuel and I went down to the river yesterday! Samuel isn't a very good swimmer, though. Good thing our friend Sally was there to help."

# the findall() function finds every instance of a specified word pattern within a text
re.findall(r'Samuel', text)
```




    ['Samuel', 'Samuel']



Let's say that instead of only wanting to find Samuel, we wanted to find every word in the text starting with 'Sa' (case sensetive). What would we do? Use regular expressions!


```python
re.findall(r'Sa[a-z]*', text)
```




    ['Samuel', 'Samuel', 'Sally']



You may be wondering what the [a-z] in the Sa[a-z] pattern means. This is called a **set** in regex. When characters are within a set, such as  [abcde], any one character will match. However, regex has a special rule where [a-z] means the same thing as [abcde...xyz].

Here are some more:
~~~ 
[0-9]        any numeric character
[a-z]        any lowercase alphabetic character
[A-Z]        any uppercase alphabetic character
[aeiou]      any vowel (i.e. any character within the brackets)
[0-9a-z]     to combine sets, list them one after another 
[^...]       exclude specific characters
~~~


You still may be wondering how the entirety of Samuel was able to be matched if only one character within [a-z] would match. The answer is something called a **quantifier**!

Rules:
~~~ 
*        0 or more of the preceding character/expression
+        1 or more of the preceding character/expression
?        0 or 1 of the preceding character/expression
{n}      n copies of the preceding character/expression 
{n,m}    n to m copies of the preceding character/expression 
~~~

Say that now, you only wanted to return Samuel when the name was mentioned at the beginning of the text.


```python
re.findall(r'^Samuel', text)
```




    ['Samuel']



**Special characters**, such as the *^* which was just used in the pattern above, match strings that have a specific placement in a sentence. For example, *^* matches the subsequent pattern only if it is at the beginning of the string. This is why only a single 'Samuel' was returned.

Rules:
~~~ 
.         any single character except newline character
^         start of string
$         end of entire string
\n        new line
\r        carriage return
\t        tab

~~~

**Python RegEx Methods**

* `re.findall(pattern, string)`: Returns all phrases that match your pattern in the string.

* `re.sub(pattern, replacement, string)`: Return the string after replacing the leftmost non-overlapping occurrences of the pattern in string with replacement

* `re.split(pattern, string)`: Split string by the occurrences of pattern. If capturing parentheses are used in pattern, then the text of all groups in the pattern are also returned as part of the resulting list. 

**Pandas RegEx Methods**

Pandas also has its own [__built in methods__](https://kanoki.org/2019/11/12/how-to-use-regex-in-pandas/) of working with regex (without calling a separate "re" library)

* `df['column'].str.extract(pattern)`: Extract pattern into a new column

* `df['column'].str.replace(pattern, replace):` Replace pattern with something else (usually a "space" if you want to remove


------------------

## Practical Regex cleaning example

Let's go back to our case.law dataset and try to "clean" it of some repetitions in the data. 

For example - the __"OPINION OF THE COURT"__ in the beginning of every decision is not really useful information. 

**How would we use Regex to clean it up?**



```python
df = df_cleaned[:1000]
```


```python
df.tail()
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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>995</th>
      <td>1845-12-01</td>
      <td>Thomas v. Negus</td>
      <td>"The Opinion of the Court was delivered'by\nTr...</td>
      <td>1845</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1845-12-01</td>
      <td>Owen v. Barnum</td>
      <td>The Opinion of the Court was delivered by\nPur...</td>
      <td>1845</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1845-12-01</td>
      <td>Murphy v. Summerville</td>
      <td>The Opinion of the Court was delivered by\nSca...</td>
      <td>1845</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1845-12-01</td>
      <td>Anderson v. Semple</td>
      <td>The Opinion of the Court was delivered by\nSca...</td>
      <td>1845</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1845-12-01</td>
      <td>Scott v. Crow</td>
      <td>The Opinion of the Court was delivered by\nTre...</td>
      <td>1845</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.text[200][:1000]) 
print("\n")
print(df.text[300][:1000])
print("\n")
print(df.text[0][:1000])
```

    Wilson, Chief Justice,
    delivered the opinion of the Court:
    (1)
    The bill of exceptions, or rather demurrer to evidence, in this case, presents this state of facts.
    The plaintiff below made an improvement on the land of the United States which the defendant afterwards purchased of the government, and, after the purchase, promised the plaintiff to pay him the value of his improvements. It further appears from the evidence, that the plaintiff had, prior to to the commencement of this suit, instituted an action before a justice of the peace, upon another demand, without having joined this one with itj though it was at the time a subsisting demand. The first suit was never tried, but was compromised by the parties, and dismissed.
    Upon this evidence, the Court below gave judgment in favor of the plaintiff, for the value of the improvements.
    The first error assigned to reverse this decision, is, that the first suit commenced by the plaintiff, is a bar to this action. To support this assignment
    
    
    Lockwood, Justice,
    delivered the opinion of the Court:
    This was an action of ejectment brought in the Jo Daviess Circuit Court, to recover the possession of a lot of ground in the town of Galena. The cause was tried by the Court, by consent of parties, without a jury, and it was agreed by the parties, “ That both or either party should have the same right to except, as if this cause were tried by a jury.”
    A bill of exceptions was taken by the plaintiff on the trial, by which it appears that testimony was given by both parties on the question raised on the trial, whether a deed purporting to have been executed by Spraggins to Ballingall, had been duly delivered. The Court was of opinion that there was not sufficient proof-of the delivery of the deed, and non-suited the plaintiff. This decision the plaintiff assigns for error.
    The point presented in this case for our decision, is whether a bill of exceptions will lie to the opinion of the Court, where the Court hears the testimony on bot
    
    
    Mr. PRESIDING JUSTICE EBERSPACHER
    delivered the opinion of the court:
    Defendant Tobin was convicted by a jury of the crime of burglary. The court entered judgment upon the verdict and sentenced the defendant to a fifteen to twenty-five year term in the Illinois State Penitentiary. The judgment of the court further provided that the mittimus was to be effective upon release by federal authorities.
    The defendant has appealed from that judgment and raised the following issues: (1) The State failed to prove lack of authority to enter the premises; (2) The State failed to prove intent to commit a theft; (3) The court erred in allowing testimony concerning the arrest of Sherri Tobin, her possession of a firearm and evidence concerning defendant’s possession of a firearm; (4) The sentence was excessive.
    The facts giving rise to this case are as follows: On the night of February 9, 1969, at about 11:00 P.M. the defendant, in the company of Sherri Tobin, Daniel Stout, Michael Hume and Eddie Dun


As you can see, mostly decisions are written with "OPINION __OF__ THE COURT".

In my experience, the easiest way to work with these patterns is to copy them to [__RegexR__](https://regexr.com/), and work there on a small sample (like the one above). 

The important thing to note is that you must **search for patterns** in the text that will enable you to clean up the data. But __be very careful__ because REGEX can be very powerful - making an incorrect pattern could remove text that you didn't want to remove. 

### First Try - an obvious pattern

After testing out some patterns, the simplest pattern we could use is as follows:


```python
pattern = r'(Opinion of the Court|opinion of the Court|opinion of the court)' # remember, in python we need the paranthesis
 
df['pattern'] = df['text'].str.extract(pattern, 
                                       expand = True) # str.extract returns the FIRST match
                                                        
```

    /tmp/ipykernel_29263/3403993640.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['pattern'] = df['text'].str.extract(pattern,



```python
df.head(10)
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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
      <th>pattern</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
      <td>opinion of the court</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1819-12-01</td>
      <td>Smith v. Bridges</td>
      <td>Opinion of the Court.\n*\nThe plaintiff below,...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1820-07-01</td>
      <td>Mason v. Buckmaster</td>
      <td>Opinion of the Court. It is necessary by the c...</td>
      <td>1820</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1820-07-01</td>
      <td>Cox v. McFerron</td>
      <td>Opinion of the Court. It appears, that by the ...</td>
      <td>1820</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1820-07-01</td>
      <td>Beaumont v. Yantz</td>
      <td>Opinion of the Court. The cases cited by the a...</td>
      <td>1820</td>
      <td>Opinion of the Court</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1820-07-01</td>
      <td>Cornelius v. Vanorsdall</td>
      <td>Opinion of the Court. In this case there was a...</td>
      <td>1820</td>
      <td>Opinion of the Court</td>
    </tr>
  </tbody>
</table>
</div>



And actually... it's pretty good!

We can conditionally check the ones that "weren't captured" - ie **NaN** - by subsetting the dataframe. 

We do that by creating a new dataframe which has the condition of having only the rows that contain **NaN** in the "cleaned_text" column


```python
df_test = df[df['pattern'].isna()]
df_test
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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
      <th>pattern</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>1820-12-01</td>
      <td>Naught v. Oneal</td>
      <td>Per curiam.\nIf the cause of action accrued on...</td>
      <td>1820</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1823-11-01</td>
      <td>Ackless v. Seekright</td>
      <td>Opinion of the Gowrt by\nChief Justice Reynold...</td>
      <td>1823</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1823-11-01</td>
      <td>White v. Stafford</td>
      <td>Opinion' of the Court by\nJustice John Reynold...</td>
      <td>1823</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1825-12-01</td>
      <td>Conley v. Good</td>
      <td>Opinion of the. Court by\nJustice Lockwood.\n*...</td>
      <td>1825</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>65</th>
      <td>1825-12-01</td>
      <td>Cornelius v. Wash</td>
      <td>Opinion-of the Court by\nJustice Lockwood.\nTw...</td>
      <td>1825</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>869</th>
      <td>1844-12-01</td>
      <td>Sellers v. People</td>
      <td>Per Curiam.\n*\nThe case before us appears to ...</td>
      <td>1844</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>873</th>
      <td>1844-12-01</td>
      <td>Wren v. Moss</td>
      <td>Per Curiam.\nThe motion is allowed, with an or...</td>
      <td>1844</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>903</th>
      <td>1844-12-01</td>
      <td>Hedges v. County of Madison</td>
      <td>Per Curiam.\nThe motion must be denied. On the...</td>
      <td>1844</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>910</th>
      <td>1844-12-01</td>
      <td>Frazier v. Laughlin</td>
      <td>Motion allowed — Lockwood, J. dissenting.\nMot...</td>
      <td>1844</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>952</th>
      <td>1845-12-01</td>
      <td>Harback v. Gear</td>
      <td>Per Curiam.\nThis is the same case as the one ...</td>
      <td>1845</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>94 rows × 5 columns</p>
</div>



As we can see, it's not really capturing all the "useless" text.



One of the problems is that "opinion of the court" also appears in the body of the opinion and not necessarily just in the beginning. 

We can see 


```python
matches = df['text'].str.findall(pattern).apply(pd.Series) ## find all the "opinions of the court" - One column per match
```


```python
matches.head(20)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>opinion of the court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Opinion of the Court</td>
      <td>opinion of the court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>opinion of the court</td>
      <td>opinion of the court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
rows_no_nans = matches[matches.notna().sum(axis=1) == 3] # Rows with exactly three non-NaNs
```


```python
rows_no_nans
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>96</th>
      <td>Opinion of the Court</td>
      <td>opinion of the court</td>
      <td>opinion of the court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>191</th>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>198</th>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>206</th>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>272</th>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>302</th>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>315</th>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>317</th>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>752</th>
      <td>opinion of the court</td>
      <td>opinion of the court</td>
      <td>opinion of the court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>848</th>
      <td>opinion of the court</td>
      <td>opinion of the court</td>
      <td>opinion of the court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>888</th>
      <td>Opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>918</th>
      <td>Opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>923</th>
      <td>Opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>960</th>
      <td>Opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>985</th>
      <td>Opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>opinion of the Court</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Second try 

Another pattern that emerges is that the first line of the text is usually useless. The first line is seperated by a "/n" symbol. We could try removing anythign before the first newline character /n. 


```python
pattern = r"(^.*?\n)"

df['pattern2'] = df['text'].str.extract(pattern)
```

    /tmp/ipykernel_29263/2497637567.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['pattern2'] = df['text'].str.extract(pattern)



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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
      <th>pattern</th>
      <th>pattern2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
      <td>opinion of the court</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1845-12-01</td>
      <td>Thomas v. Negus</td>
      <td>"The Opinion of the Court was delivered'by\nTr...</td>
      <td>1845</td>
      <td>Opinion of the Court</td>
      <td>"The Opinion of the Court was delivered'by\n</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1845-12-01</td>
      <td>Owen v. Barnum</td>
      <td>The Opinion of the Court was delivered by\nPur...</td>
      <td>1845</td>
      <td>Opinion of the Court</td>
      <td>The Opinion of the Court was delivered by\n</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1845-12-01</td>
      <td>Murphy v. Summerville</td>
      <td>The Opinion of the Court was delivered by\nSca...</td>
      <td>1845</td>
      <td>Opinion of the Court</td>
      <td>The Opinion of the Court was delivered by\n</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1845-12-01</td>
      <td>Anderson v. Semple</td>
      <td>The Opinion of the Court was delivered by\nSca...</td>
      <td>1845</td>
      <td>Opinion of the Court</td>
      <td>The Opinion of the Court was delivered by\n</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1845-12-01</td>
      <td>Scott v. Crow</td>
      <td>The Opinion of the Court was delivered by\nTre...</td>
      <td>1845</td>
      <td>Opinion of the Court</td>
      <td>The Opinion of the Court was delivered by\n</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 6 columns</p>
</div>



Again, not a perfect pattern. We are losing some information by capturing the names of Judges in the pattern - and thus we could remove that. 

But this is just an example. Now that we tested the pattern using `.str.extract()`, we can proceed to remove the pattern using `.str.replace()` by replacing the pattern with a space.


```python
df['cleaned_text'] = df['text'].str.replace(pattern, " ", 
                                            regex = True)
```

    /tmp/ipykernel_29263/394696829.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['cleaned_text'] = df['text'].str.replace(pattern, " ",



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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
      <th>pattern</th>
      <th>pattern2</th>
      <th>cleaned_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
      <td>opinion of the court</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\n</td>
      <td>delivered the opinion of the court:\nDefendan...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1. Uncertainty in the indictment, in not aver...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
      <td>*\nThis was an action of debt on a judgment r...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
      <td>*\nThis was an action of covenant. The fifth ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
      <td>†\nIt appears from the record in this cause, ...</td>
    </tr>
  </tbody>
</table>
</div>



The reality of regex in my experience is that you have to try different patterns depending on your data and what you want to remove or explore. Seemingly irrelevant information like "newline" becomes important in the context of regex. 

# Concordances and Collocations

## Concordances
To continue this examination of simple NLP methods, let us now explore concordances.

A "concordance" lists every instance of a given word (string), together with some of its context. Concordances are important if we want to understand the meaning of a word in a context.

For example, we can look up the word "petition" in context


In order to perform concordance analysis, we'll import __NLTK__, which is a good simple library for doing NLP tasks in Python. 

You are probably noticing that we are  importing a lot of packages - this is because there are numerous libraries which can do a lot of different things in python, usually in a couple of lines.  


```python
import nltk
nltk.download("punkt")

from nltk.text import Text
from nltk.tokenize import word_tokenize # import tokenizer function from nltk - more on this in the following labs

```

    [nltk_data] Downloading package punkt to /home/leondgarse/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.



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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
      <th>pattern</th>
      <th>pattern2</th>
      <th>cleaned_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
      <td>opinion of the court</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\n</td>
      <td>delivered the opinion of the court:\nDefendan...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1. Uncertainty in the indictment, in not aver...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
      <td>*\nThis was an action of debt on a judgment r...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
      <td>*\nThis was an action of covenant. The fifth ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
      <td>Opinion of the Court</td>
      <td>Opinion of the Court.\n</td>
      <td>†\nIt appears from the record in this cause, ...</td>
    </tr>
  </tbody>
</table>
</div>



NLTK’s concordance works on a single list of tokens, so we first have to merge everything.


```python
cases = df['cleaned_text'][:100]  ## Make a list of cases out of the first 100 cases from the dataframe
```


```python
cases[0]
```




    ' delivered the opinion of the court:\nDefendant Tobin was convicted by a jury of the crime of burglary. The court entered judgment upon the verdict and sentenced the defendant to a fifteen to twenty-five year term in the Illinois State Penitentiary. The judgment of the court further provided that the mittimus was to be effective upon release by federal authorities.\nThe defendant has appealed from that judgment and raised the following issues: (1) The State failed to prove lack of authority to enter the premises; (2) The State failed to prove intent to commit a theft; (3) The court erred in allowing testimony concerning the arrest of Sherri Tobin, her possession of a firearm and evidence concerning defendant’s possession of a firearm; (4) The sentence was excessive.\nThe facts giving rise to this case are as follows: On the night of February 9, 1969, at about 11:00 P.M. the defendant, in the company of Sherri Tobin, Daniel Stout, Michael Hume and Eddie Dunn was in an automobile driven by defendant in the vicinity of the Oliver C. Joseph Automobile Agency in Belleville. Their behavior while driving was observed by James Muir who resided nearby. He stated the car stopped by the agency and the driver, identified as Tobin, jumped out and ran across the street and kicked the agency door. The car in the meantime circled the block and picked Tobin up. The car drove away and Muir next observed four men walking up the street to the agency. Muir recognized one of the four as the man who kicked the door. The four men then entered the agency building by the same door previously kicked. Muir then called the police.\nThe police arrived and Officers Rettle and Wobbe observed four men walking through the building. Two other policemen arrived. Rettle observed two of the suspects at the back door. He identified himself and ordered them out. They disappeared back inside the building. The police entered the building and found one suspect lying under a car, After turning on the lights they searched the building. The other suspects were found in the basement. The defendant was hiding behind an air compressor when discovered.\nAfter apprehending all four men the police with Mr. Muir’s help located the car a few blocks away. Sherri Tobin was found in the car asleep. She was carrying a “.38 caliber snub-nosed revolver, fully loaded, with the serial numbers filed off” in the waist band of her slacks. One of the defendants, Hume, testified for the State and said defendant stated earlier in the evening that they would go to Belleville and “make some money”. He also stated that defendant brought a gun into the building.\nThe evidence also showed that the door jamb of the door kicked by defendant was splintered and the door opened by force.\nAdditionally, Mr. Oliver P. Joseph testified that the building was owned by and in the possession of Oliver C. Joseph, Inc., a corporation, engaged in the selling of automobiles.\nAs to the authority to be in the premises, “* * * the law presumes that the presence in a public building for a purpose inconsistent with the purposes for which the building is open to the public is without authority”. (People v. Urban (1971), (Ill.App.2d), 266 N.E.2d 112, 114. Also see People v. Weaver (1968), 41 Ill.2d 434, 243 N.E.2d 245 cert. den. 395 U.S. 959, 89 S.Ct. 2100, 23 L.Ed.2d 746.) Here the defendant (a) had to break open a door to gain admission, (b) at 11:00 P.M., (c) when the buffding was unlit, and (d) hid in the basement upon arrival of the police. Under these circumstances there was sufficient evidence for the jury to believe that his presence was without authority.\nIn regard to the question of intent there is also sufficient circumstantial evidence for the jury to believe that Tobin intended to commit a theft in the building. Intent must ordinarily be proved circumstantially, by inferences drawn from conduct appraised in its factual environment.” (People v. Johnson (1963), 28 Ill.2d 441, 192 N.E.2d 864, 866.) The Court in Johnson went on to say:\n“* * * We are of the opinion that in the absence of inconsistent circumstances, proof of unlawful breaking and entry into a building which contains personal property that could be the subject of larceny gives rise to an inference that will sustain a conviction of burglary. Like other inferences, this one is grounded in human experience, which justifies the assumption that the unlawful entry was not purposeless, and, in the absence of other proof, indicates theft is the most likely purpose.”\nThe circumstances of the entry coupled with Hume’s testimony that Tobin intended to “make some money” in BeUeviHe is sufficient evidence of intent to commit a theft.\nThe third aUeged error relates to Sherri Tobin’s arrest and the question of firearms. It is contended that the evidence of Sherri’s arrest is not only irrelevant but introduced solely for the purpose of bringing in evidence of the .38 cafibre gun to prejudice the jury. In regard to Tobin’s possession of a firearm the only evidence of this is Hume’s testimony. No gun was introduced into evidence.\nWhile the evidence of Sherri’s arrest may not have been essential to the conviction of Tobin, it apprised the finder of fact of the total circumstances surrounding the event and at worst, it is harmless error. The defendant has aHeged prejudice but a careful search of the record discloses none. Such evidence did not prove an element of the crime not estabfished by other properly admitted evidence. People v. Landgham (1970), 122 Ill.App.2d 9, 275 N.E.2d 484; People v. Jones, (1970), 125 Ill.App.2d 30, 259 N.E.2d 585.\nDefendant argues that his sentence was excessive because he received a heavier sentence than his co-defendants who pleaded guilty to the same offense. The record shows that Eddie Dunn and Michael Hume were each placed on probation for a period of five years. Daniel Stout was sentenced to not less than five nor more than ten years. Defendant asserts that he was penalized for having exercised his constitutional right to a trial by jury.\nThe basic principles regarding sentencing are set forth in People v. Jones, 118 Ill.App.2d 189, 254 N.E.2d 843, 847:\n“We recognize that not every offense in a like category calls for an identical punishment. There may be a proper variation in sentences as between different offenders, depending upon the circumstances of the individual case. As a general rule, where the punishment for the offense is fixed by statute, that imposed in the sentence must conform thereto, and a sentence which conforms to statutory regulation is proper. Before an AppeUate Court wiU interfere, it must be manifest from the record that the sentence is excessive and not justified by any reasonable view which might be taken of the record. People v. Hobbs, 58 Ill.App.2d 93, 99, 205 N.E.2d 503 (1965). Disparity of sentences between defendants does not, of itself, warrant the use of the power to reduce a punishment imposed by the trial court. People v. Thompson, 36 Ill.2d 478, 482, 224 N.E.2d 264 (1967).”\nHume had no prior criminal record. The records of the other defendants do not appear in the record on this appeal. Defendant Tobin had a prior burglary conviction as a juvenile on which he served a year, and contrary to the provision of his pretrial bail, he left the State. While gone he became involved with federal authorities as evidenced by the fact that his presence at trial was secured by virtue of a Writ of Habeas Corpus Ad Prosequendum. Tobin was also the apparent ringleader of the burglary. He drove the car, planned the burglary, selected the site and broke open the door. Under the circumstances, we find the factual situation to differ from People v. Jones, supra, on which defendant relies, and a penalty greater than that imposed upon defendant’s accomplices is approved.\nHowever, we do not consider the possibility of rehabilitation to be so remote as to justify a sentence of 15 to 25 years, which as a practical matter leaves little or no room for rehabilitation; nor does such sentence provide for an exercise of the discretion of parole authorities at a time when such discretion may contribute most to rehabilitation. As a result we would consider that the sentence should be modified to provide a minimum of seven years and a maximum of 20 years.\nLastly, defendant claims the court erred in making his sentence consecutive to a possible future federal sentence. We agree. However, the remedy is not necessarily to make the sentence concurrent with a possible future federal sentence as argued by defendant. See Ill. Rev. Stat. ch. 38, pars. 119 — 1 and 119 — 2.\nThe language “Mittimus to be effective upon release by federal authorities” is too broad and does not clearly define what sentence the imposed sentence is to follow. See Ill. Rev. Stat. ch. 38, par. 7 — 1. We would particularly call attention to the fact that ch. 38, par. 7 — l(n) makes provision only for a concurrent sentence with a “previous and unexpired sentence” imposed by a Federal District Court. During the oral argument in June 1971, this Court was advised that the defendant’s involvement with federal authorities to which we have referred is that depicted by United States v. Tobin, 426 Fed.2d 1279, in which defendant’s conviction was reversed and the cause remanded for a new trial in May 1970, and that defendant had not been reprosecuted in that cause.\n“A sentence should be so complete as not to require construction by the court to ascertain its import, and so complete that it will not be necessary for a nonjudicial or ministerial officer to supplement the written words to ascertain its meaning.” People v. Walton (1969), 118 Ill.App.2d 324, 254 N.E.2d 190, 194 — 5.\nThe judgment of conviction is affirmed, the sentence is modified to provide a minimum sentence of seven years and a maximum sentence of twenty years;\nJudgment with sentence so modified is affirmed.\nMORAN and CREES, JJ., concur.'



Now, we must convert these cases into a single string


```python
case_corpus = " ".join(cases).lower()  # join every case into a single string, and lowercase
```


```python
print(case_corpus[:1000])
```

     delivered the opinion of the court:
    defendant tobin was convicted by a jury of the crime of burglary. the court entered judgment upon the verdict and sentenced the defendant to a fifteen to twenty-five year term in the illinois state penitentiary. the judgment of the court further provided that the mittimus was to be effective upon release by federal authorities.
    the defendant has appealed from that judgment and raised the following issues: (1) the state failed to prove lack of authority to enter the premises; (2) the state failed to prove intent to commit a theft; (3) the court erred in allowing testimony concerning the arrest of sherri tobin, her possession of a firearm and evidence concerning defendant’s possession of a firearm; (4) the sentence was excessive.
    the facts giving rise to this case are as follows: on the night of february 9, 1969, at about 11:00 p.m. the defendant, in the company of sherri tobin, daniel stout, michael hume and eddie dunn was in an automobile driven by 



```python
import nltk
nltk.download('punkt_tab')
```

    [nltk_data] Downloading package punkt_tab to
    [nltk_data]     /home/leondgarse/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt_tab.zip.





    True



Now we tokenize - ie, convert each word into a token.


```python
case_corpus_tokenized = word_tokenize(case_corpus)
text = Text(case_corpus_tokenized)
```


```python
text.concordance('fraud')
```

    Displaying 18 of 18 matches:
    tes , page 773 , provides , “ if any fraud or circumvention be used , in obtain
    oresaid , ( notes and bonds , ) such fraud or circumvention may be pleaded in b
    brought by the party committing such fraud or circumvention , or any assignee o
    t was held to apply only to cases of fraud in making or obtaining the instrumen
    ; but the court held that that was a fraud in the consideration and not in the 
     , or was prevented from doing it by fraud or accident , or the act of the oppo
    dgment complained of was obtained by fraud , or resulted from inevitable accide
    ch award or umpirage was obtained by fraud , corruption or other undue means : 
     that the award had been produced by fraud , corruption or other undue means , 
     , 16 ill. , 34. if there is neither fraud or misconduct on the part of the arb
    on law . that case required that the fraud should be of such a nature as would 
     which is necessary to be shown in a fraud on a private individual . the act of
    t specie . there is no allegation of fraud on the part of m. duncan , or notice
    iable instrument , where there is no fraud , be impeached , either at law or in
     of them with any knowledge that any fraud or misrepresentation had been used i
     turcotte or his agent practised any fraud or deception . turcotte was delayed 
     , or was prevented from doing it by fraud or accident , or the act of the oppo
    imself at law , but was prevented by fraud or accident , unmixed with any fault



Let's examine other words.


```python
text.concordance("accident", 
                 width=110) ## width determines how many characters before and after we want to examine
```

    Displaying 4 of 4 matches:
    mself , or was prevented from doing it by fraud or accident , or the act of the opposite party , unmixed with 
    as obtained by fraud , or resulted from inevitable accident , and that the courts of law can not grant adequat
    mself , or was prevented from doing it by fraud or accident , or the act of the opposite party , unmixed with 
    led himself at law , but was prevented by fraud or accident , unmixed with any fault or negligence in himself 



```python
## if we want phrases we need to put them in a list
text.concordance(["good", "faith"],  
                 width=110)
```

    Displaying 3 of 3 matches:
    the contract appears to have been entered info in good faith , and in the ordinary manner of making such contr
     court feel constrained to say , that justice and good faith require that the plaintiffs should recover the di
    asonable indulgence , and if they are governed by good faith , and act within their jurisdiction , they ought 


**Write your own code to explore the occurrence of other words of interest**


```python

```

## Collocations - attempts at informative multi-word phrases
Collocations are expressions of multiple words which commonly co-occur. These collocations are measured using **Pointwise Mutual Information** - which gives the probability of "two events co-occuring" - in this case, two words co-occuring. This can give us measures of associations between words - things like phrases, or co-occuring words can be revealed.

Here's [a pretty good explanation of PMI](https://stats.stackexchange.com/a/522504) is given on **stackexchange.**

Note: stackexchange (and google generally) is a wonderful resource for all things relating to NLP and statistics. Although for this course, I do not emphasize concrete statistical knowledge or mathematical formulas, you should still get some **intuitive** understanding of what these measures like PMI do. 



```python
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
fourgram_measures = nltk.collocations.QuadgramAssocMeasures()
```


```python
bigram_finder = BigramCollocationFinder.from_words(case_corpus_tokenized, 
                                                   window_size = 5)

bigram_finder.apply_freq_filter(5)  # appear at least N times - here 5
bigram_finder.nbest(bigram_measures.pmi, 20) # show top 20 PMI scoring
```




    [('crouch', 'hall'),
     ('dane', 'dig.'),
     ('nul', 'tiel'),
     ('laid', 'down'),
     ('taylor', 'sprinkle'),
     ('george', 'lunceford'),
     ('brothers', 'sisters'),
     ('jonathan', 'mayo'),
     ('expire', 'end'),
     ('ill.app.2d', 'n.e.2d'),
     ('forcible', 'detainer'),
     ('goods', 'chattels'),
     ('scire', 'facias'),
     ('18th', 'july'),
     ('st.', 'clair'),
     ('nil', 'debet'),
     ('thompson', 'armstrong'),
     ('ex', 'facto'),
     ('bryan', 'morrison'),
     ('heirs', 'assigns')]



Do you see any interesting patterns that emerge from the dataset?

We can also examine the **score of these bigrams as they are in PMI**. 

The [mathematics](https://en.wikipedia.org/wiki/Pointwise_mutual_information) are not that important for the purposes of the class - as we care more about intuition behind these measures.

It suffices to say that:
* A bigger **positive PMI** score implies that a word1 (event1) tends to co-occur more with word2 (event2). 
* A **PMI score of 0** means that the two words (events) are independent. 
* A **negative PMI** score can mean that the two words are uninformative. In practice, usually, [Positive PMI (or __PPMI__)](https://stats.stackexchange.com/a/284573) is used (where the negative values are not included).

To learn more about this, see generally Jurafsky and Martin reference text.


```python
for i in bigram_finder.score_ngrams(bigram_measures.pmi):
    print(i)
```

    (('crouch', 'hall'), 11.707143401980908)
    (('dane', 'dig.'), 11.707143401980908)
    (('nul', 'tiel'), 11.707143401980908)
    (('laid', 'down'), 11.444108996147115)
    (('taylor', 'sprinkle'), 11.292105902702065)
    (('george', 'lunceford'), 11.122180901259753)
    (('brothers', 'sisters'), 11.12218090125975)
    (('jonathan', 'mayo'), 11.029071496868271)
    (('expire', 'end'), 10.970177807814704)
    (('ill.app.2d', 'n.e.2d'), 10.859146495425959)
    (('forcible', 'detainer'), 10.832674284064769)
    (('goods', 'chattels'), 10.707143401980908)
    (('scire', 'facias'), 10.707143401980908)
    (('18th', 'july'), 10.614033997589427)
    (('st.', 'clair'), 10.591666184560971)
    (('nil', 'debet'), 10.48475098064446)
    (('thompson', 'armstrong'), 10.48475098064446)
    (('ex', 'facto'), 10.466135302477113)
    (('bryan', 'morrison'), 10.385215307093546)
    (('heirs', 'assigns'), 10.385215307093546)
    (('six', 'cent'), 10.229096105176263)
    (('mason', 'wash'), 10.122180901259751)
    (('affix', 'seal'), 10.044178389258478)
    (('31st', 'october'), 10.006703683839815)
    (('per', 'cent'), 10.006703683839815)
    (('six', 'per'), 9.952255899817441)
    (('scates', 'comp.'), 9.822620619400844)
    (('become', 'vacant'), 9.777532729872306)
    (('t.', 'r.'), 9.768543946645053)
    (('ten', 'days'), 9.719216234281483)
    (('davidson', 'wife'), 9.640029206122371)
    (('false', 'imprisonment'), 9.552566364872032)
    (('r.', 'co.'), 9.505509540811259)
    (('sec', 'scates'), 9.48475098064446)
    (('during', 'recess'), 9.417636784785925)
    (('happen', 'recess'), 9.318101111235007)
    (('next', 'kin'), 9.262358559308012)
    (('8', 'johns.'), 9.221716574810667)
    (('happen', 'during'), 9.180597587485073)
    (('purple', 'statutes'), 9.169709271342338)
    (('regulating', 'practice'), 9.162822885757098)
    (('edward', 'coles'), 9.162822885757096)
    (('constrained', 'say'), 9.122180901259753)
    (('six', 'damages'), 9.063287212206182)
    (('hundred', 'dollars'), 9.047101784465244)
    (('award', 'umpirage'), 9.04417838925848)
    (('afford', 'remedy'), 8.99932415347422)
    (('dollars', 'cents'), 8.94869808040425)
    (('replevy', 'three'), 8.919551734744733)
    (('certificate', 'register'), 8.882714966564363)
    (('november', '1825'), 8.814058605897419)
    (('march', '1819'), 8.806679075531823)
    (('ills', 'r.'), 8.768543946645053)
    (('cent', 'damages'), 8.755164916843851)
    (('end', 'next'), 8.747785386478252)
    (('johns', 'cas.'), 8.737517051024426)
    (('replevy', 'years'), 8.707143401980908)
    (('october', '1825'), 8.698581388477484)
    (('entry', 'detainer'), 8.695170760314832)
    (('setting', 'aside'), 8.648249712927338)
    (('motion', 'dismiss'), 8.602806742166173)
    (('issuing', 'warrant'), 8.586128001019542)
    (('persons', 'interested'), 8.55965578965835)
    (('chief', 'reynolds'), 8.549846008866346)
    (('post', 'facto'), 8.52271883084348)
    (('avail', 'himself'), 8.49624661948229)
    (('chief', 'wilson'), 8.492130511010057)
    (('7', 'cranch'), 8.470104204680057)
    (('22d', '1819'), 8.417636784785923)
    (('ex', 'post'), 8.407241613423546)
    (('countersign', 'commission'), 8.385215307093546)
    (('per', 'damages'), 8.28567963354263)
    (('&', 'c.'), 8.275503935364933)
    (('recess', 'senate'), 8.253970773815292)
    (('mason', 'ante'), 8.234655630518166)
    (('availed', 'himself'), 8.233212213648494)
    (('john', 'reynolds'), 8.229937081615635)
    (('general', 'assembly'), 8.183581445923895)
    (('lieutenant', 'governor'), 8.183581445923895)
    (('sims', 'post'), 8.162822885757098)
    (('tract', 'land'), 8.122180901259751)
    (('set', 'aside'), 8.114817512846267)
    (('new', 'york'), 8.082652537073114)
    (('first', 'taker'), 8.044178389258478)
    (('secretary', 'countersign'), 8.044178389258478)
    (('set', 'forth'), 8.025319362007163)
    (('section', 'article'), 7.954236263945681)
    (('discharge', 'thereof'), 7.899788479923304)
    (('united', 'states'), 7.899788479923304)
    (('reason', 'why'), 7.882714966564363)
    (('i', 'am'), 7.860553316640205)
    (('errors', 'assigned'), 7.858669492597711)
    (('johns', 'rep.'), 7.854874001662587)
    (('due', 'diligence'), 7.844646925730844)
    (('became', 'due'), 7.800252806372388)
    (('discharge', 'duties'), 7.800252806372388)
    (('unnecessary', 'decide'), 7.762284956173369)
    (('justices', 'peace'), 7.760724442185753)
    (('ever', 'since'), 7.747785386478252)
    (('why', 'mandamus'), 7.719216234281484)
    (('less', 'than'), 7.716188541583916)
    (('set', 'off'), 7.662749282622455)
    (('three', 'years'), 7.6565173289109385)
    (('third', 'article'), 7.636754074089511)
    (('st.', 'county'), 7.628192060586086)
    (('cent', 'interest'), 7.591666184560973)
    (('clair', 'county'), 7.591666184560973)
    (('fourth', 'section'), 7.591666184560971)
    (('real', 'estate'), 7.587971701003232)
    (('r.', 'r.'), 7.566910085475401)
    (('variance', 'between'), 7.56191884116692)
    (('counsel', 'relator'), 7.561465946785274)
    (('distinction', 'between'), 7.5398925348369215)
    (('county', 'commissioners'), 7.521276856669575)
    (('debet', 'good'), 7.521276856669575)
    (('&', 'co.'), 7.505509540811259)
    (('property', 'gatewood'), 7.505509540811259)
    (('failure', 'consideration'), 7.48475098064446)
    (('bill', 'exceptions'), 7.475122279446683)
    (('ante', 'page'), 7.471217824990768)
    (('opposite', 'party'), 7.4576348527178435)
    (('more', 'than'), 7.434902431193899)
    (('et', 'al.'), 7.434124907574493)
    (('sum', 'dollars'), 7.434124907574493)
    (('et', 'al'), 7.434124907574491)
    (('laws', '1819'), 7.42537243356553)
    (('well', 'settled'), 7.4161471666454375)
    (('grant', 'mandamus'), 7.414361652753064)
    (('nul', 'record'), 7.409462853340223)
    (('tiel', 'record'), 7.409462853340223)
    (('register', 'land'), 7.3972881393941226)
    (('vested', 'interest'), 7.385215307093546)
    (('p.', 'sec'), 7.385215307093544)
    (('incumbent', 'office'), 7.3804143208297255)
    (('viz', ':'), 7.373242665427469)
    (('its', 'face'), 7.368141793734603)
    (('public', 'lands'), 7.356646154896776)
    (('neither', 'nor'), 7.355467963699493)
    (('article', 'constitution'), 7.292105902702064)
    (('5', 'gilm.'), 7.288353767840956)
    (('intention', 'legislature'), 7.241479829632095)
    (('office', 'paymaster-general'), 7.187769242887329)
    (('follows', ':'), 7.138777411790448)
    (('promise', 'pay'), 7.134253733560328)
    (('claimed', 'due'), 7.122180901259753)
    (('per', 'interest'), 7.122180901259753)
    (('popular', 'action'), 7.112196812687129)
    (('wit', ':'), 7.110208259593676)
    (('commencement', 'suit'), 7.077786781901297)
    (('20', 'ill.'), 7.069713481365616)
    (('legislature', 'pass'), 7.0556132843207635)
    (('11', 'ill.'), 7.051097803198269)
    (('14', 'ill.'), 7.029071496868271)
    (('ante', 'p.'), 7.008651955629338)
    (('statute', 'limitations'), 6.990936367981499)
    (('1819', 'page'), 6.970177807814702)
    (('21', 'ill.'), 6.970177807814702)
    (('warrant', 'issued'), 6.948698080404251)
    (('dane', 's'), 6.943377748470986)
    (('acting', 'governor'), 6.920547040090101)
    (('provision', 'constitution'), 6.9135942794483345)
    (('so', 'far'), 6.912727535630802)
    (('true', 'construction'), 6.8784779736778905)
    (('supreme', 'united'), 6.863047933108287)
    (('22', 'ill.'), 6.851533311316082)
    (('cause', 'remanded'), 6.8436454018857695)
    ((':', '1.'), 6.813815256813452)
    (('inconsistent', 'opinion'), 6.8127747266270635)
    (('comp.', 'p.'), 6.8002528063723915)
    (('register', 'office'), 6.778378306749628)
    (('12', 'ill.'), 6.777532729872306)
    (('we', 'consider'), 6.752947091594033)
    (('15', 'ill.'), 6.744618107399571)
    (('set', 'out'), 6.715565981065904)
    (('expire', 'at'), 6.714677074666316)
    (('power', 'fill'), 6.711658883103402)
    (('13', 'ill.'), 6.7071434019809075)
    (('personal', 'estate'), 6.698154618753653)
    (('personal', 'property'), 6.698154618753653)
    (('purple', 's'), 6.695450235027403)
    (('s', 'dig.'), 6.680343342637194)
    (('like', 'other'), 6.64824971292734)
    (('dane', '’'), 6.6410542115231355)
    (('’', 's'), 6.632176060176684)
    (('recover', 'costs'), 6.628192060586088)
    (('chief', 'justice'), 6.628192060586086)
    (('justice', 'reynolds'), 6.599886691365942)
    (('dispense', 'with'), 6.598618945202739)
    (('attended', 'with'), 6.5986189452027375)
    (('compliance', 'with'), 6.5986189452027375)
    (('dispensed', 'with'), 6.5986189452027375)
    (('statute', 'frauds'), 6.5986189452027375)
    (('inquire', 'whether'), 6.586128001019542)
    (('al.', '22'), 6.584746770621182)
    (('several', 'assigned'), 6.572146733841548)
    (('laws', 'page'), 6.5565498787905305)
    (('19', 'ill.'), 6.555140308535858)
    (('make', 'appointment'), 6.5523252929288045)
    (('due', 'philips'), 6.537218400538595)
    (('statutes', 'p.'), 6.510746189177404)
    (('exercise', 'power'), 6.4955254522386365)
    (('departure', 'from'), 6.491206002781761)
    (('person', 'whom'), 6.474482645190633)
    (('perfect', 'right'), 6.4719269402868775)
    (('’', 'comp.'), 6.471129210080825)
    (('inconsistent', 'with'), 6.461115421452805)
    (('subject', 'matter'), 6.459215888537324)
    (('*', '1819'), 6.44775677316226)
    (('17', 'ill.'), 6.444108996147115)
    (('does', 'appear'), 6.440356861286007)
    (('action', 'ejectment'), 6.434124907574493)
    (('promissory', 'note'), 6.434124907574493)
    (('demurrer', 'overruled'), 6.425857291590891)
    (('plea', 'nil'), 6.425857291590891)
    (('justice', 'browne'), 6.421741183118661)
    (('justice', 'lockwood'), 6.404667669759718)
    (('wilson', '*'), 6.403362653803805)
    (('purple', '’'), 6.393126698079552)
    (('i', 'think'), 6.387066102781487)
    (('’', 'dig.'), 6.378019805689343)
    (('passage', 'act'), 6.335971788073696)
    (('exceptions', 'taken'), 6.302853337536073)
    (('gave', 'opinion'), 6.294926421764444)
    (('scates', 'p.'), 6.292105902702065)
    (('16', 'ill.'), 6.292105902702064)
    (('new', 'trial'), 6.291239158884533)
    (('grant', 'new'), 6.275297615015509)
    (('debt', 'due'), 6.247711783343611)
    (('circuit', 'st.'), 6.241168937476839)
    (('record', 'shows'), 6.2395378518979125)
    (('action', 'assumpsit'), 6.237727694770989)
    (('granting', 'new'), 6.234655630518166)
    (('says', '“'), 6.234114179942889)
    (('delivered', 'opinion'), 6.227812225905906)
    (('4', 'gilm.'), 6.213065640536719)
    (('follows', '“'), 6.197588303917774)
    (('plea', 'debet'), 6.162822885757098)
    (('would', 'seem'), 6.160248942093272)
    (('3', 'scam.'), 6.1470555698987805)
    (('averment', 'declaration'), 6.137287793649961)
    (('offered', 'evidence'), 6.137287793649959)
    (('4', 'scam.'), 6.133828410726796)
    (('all', 'interested'), 6.1322345659236746)
    (('sufficient', 'answer'), 6.1322345659236746)
    (('justice', 'peace'), 6.119178413098229)
    (('legislature', 'intended'), 6.112196812687131)
    (('execution', 'issued'), 6.109711549297536)
    (('one', 'hundred'), 6.1077815221991365)
    (('writ', 'issued'), 6.106239357390731)
    (('sheriff', 'deed'), 6.105248835524915)
    (('scates', '’'), 6.100485830160434)
    (('justice', 'wilson'), 6.099813088231297)
    (('view', 'taken'), 6.09931994345094)
    (('remanded', 'new'), 6.082652537073116)
    (('act', 'regulating'), 6.072937382239903)
    (('constitution', 'united'), 6.0594451459117895)
    (('remanded', 'proceedings'), 6.055066705401213)
    (('any', 'thing'), 6.0253193620071634)
    (('action', 'replevin'), 5.996719595267193)
    (('under', 'circumstances'), 5.982856941885448)
    (('at', 'end'), 5.977711480500108)
    (('copy', 'note'), 5.974693288937196)
    (('duties', 'office'), 5.95895055239145)
    (('am', 'opinion'), 5.950278250376998)
    (('*', 'laws'), 5.941019439746606)
    (('now', 'settled'), 5.939841729443744)
    (('facto', 'law'), 5.938959077203982)
    (('madison', 'county'), 5.936314355948419)
    (('without', 'authority'), 5.926330267375796)
    (('second', 'point'), 5.924036026169981)
    (('want', 'consideration'), 5.899788479923304)
    (('al.', 'gilm.'), 5.899311477095058)
    (('s', 'statutes'), 5.898983629112534)
    (('said', 'davidson'), 5.8918363565329095)
    (('3', 'gilm.'), 5.8892578124311346)
    (('justice', 'john'), 5.877420666894849)
    (('chief', '*'), 5.872847937105025)
    (('action', 'commenced'), 5.871188713183335)
    (('b', ')'), 5.859216071517258)
    (('why', 'should'), 5.858868224609994)
    (('common', 'law'), 5.8458496728125)
    (('necessary', 'decide'), 5.841719423667774)
    (('bar', 'action'), 5.832088893494394)
    (('verdict', 'found'), 5.826131438197995)
    (('see', 'also'), 5.824500352619067)
    (('following', ':'), 5.823045582866989)
    (('good', 'plea'), 5.82227911270239)
    (('madison', 'circuit'), 5.812325638672963)
    (('king', 's'), 5.805874224721052)
    (('below', 'erred'), 5.80025280637239)
    (('rule', 'granted'), 5.80025280637239)
    (('we', 'think'), 5.790421797012696)
    (('persons', 'who'), 5.790268717799769)
    (('had', 'inconsistent'), 5.782048210994798)
    (('1819', 'p.'), 5.777532729872306)
    (('shall', 'deemed'), 5.774042927049541)
    (('other', 'means'), 5.773780595011198)
    (('section', 'act'), 5.764815086877572)
    (('will', 'consider'), 5.764628896641669)
    (('following', 'cases'), 5.764628896641668)
    (('called', 'upon'), 5.760724442185753)
    (('office', 'governor'), 5.757316691221799)
    (('charged', 'with'), 5.750622038647789)
    (('with', 'difficulties'), 5.750622038647789)
    (('entered', 'into'), 5.750363930580029)
    (('(', 'b'), 5.745211442814428)
    (('section', 'third'), 5.743669278006022)
    (('state', 'paper'), 5.734110449581179)
    (('c', ')'), 5.7336851894334)
    (('jurisdiction', 'matter'), 5.722250294371115)
    (('action', 'trespass'), 5.719879389908371)
    (('*', 'lockwood'), 5.708217235332224)
    (('lockwood', 'having'), 5.7021427209225415)
    (('between', 'parties'), 5.695543405297116)
    (('see', 'note'), 5.688985381962587)
    (('al.', 'scam.'), 5.687623951261484)
    (('causes', 'error'), 5.656206436755683)
    (('at', 'time'), 5.655783385612748)
    (('action', 'debt'), 5.6527651940498345)
    (('state', 'illinois'), 5.6312996439974015)
    (('ex', 'law'), 5.63083678184165)
    (('general', 'issue'), 5.625585992803009)
    (('remanded', 'circuit'), 5.61968056073057)
    (('further', 'proceedings'), 5.615443567844098)
    (('(', ')'), 5.613055484247859)
    (('would', 'lie'), 5.603855593568886)
    (('’', 'statutes'), 5.596660092164683)
    (('3', 'johns'), 5.585513957579378)
    (('sec', '’'), 5.570664883631737)
    ((':', '2.'), 5.565887743369867)
    (('al', 'et'), 5.551481858212652)
    (('our', 'statute'), 5.546151525308604)
    ((')', 'taylor'), 5.541040111491004)
    (('or', 'umpirage'), 5.530554670257585)
    (('2', 'gilm.'), 5.527234311965973)
    (('”', 'purple'), 5.5206164332014644)
    (('at', 'term'), 5.513043213496665)
    (('et', '22'), 5.508125489018269)
    ((':', '“'), 5.507543757139061)
    (('king', '’'), 5.503550687773201)
    (('action', 'brought'), 5.503387570011606)
    (('enter', 'upon'), 5.497690036351958)
    (('having', 'counsel'), 5.491925013532196)
    (('gave', 'no'), 5.490397543785603)
    (('secretary', 'state'), 5.486182936137594)
    (('(', 'c'), 5.482177036980634)
    (('have', 'availed'), 5.476188967141036)
    (('did', 'appear'), 5.470650783599607)
    (('brought', 'into'), 5.468283769055688)
    (('same', 'effect'), 5.454162660811038)
    (('cause', 'gave'), 5.4513279791070115)
    (('demurrer', 'sustained'), 5.440356861286007)
    (('relied', 'upon'), 5.438796347298389)
    (('suit', 'brought'), 5.432451663200661)
    (('people', 'gilm.'), 5.423283347927066)
    (('writ', 'error'), 5.421741183118659)
    (('grant', 'trial'), 5.407935383593628)
    (('land', 'office'), 5.3804143208297255)
    (('against', 'administrator'), 5.376824036834405)
    (('sheriff', 'officer'), 5.368283241358707)
    (('legal', 'right'), 5.367590280472141)
    (('justice', 'smith'), 5.362847494065091)
    (('inasmuch', 'as'), 5.361368565139179)
    (('fraud', 'or'), 5.360629668815275)
    (('so', 'much'), 5.360186512602024)
    (('reversed', '*'), 5.353609618606706)
    (('will', 'lie'), 5.3495913973628255)
    (('entitled', 'act'), 5.335971788073698)
    (('his', 'own'), 5.330173943535584)
    (('evidence', 'title'), 5.28929088709501)
    (('suit', 'commenced'), 5.284237659368726)
    (('founded', 'upon'), 5.275297615015511)
    (('negligence', 'or'), 5.267520264423792)
    (('let', 'judgment'), 5.266274234370039)
    (('subsequent', 'one'), 5.259784615644186)
    (('appeal', 'circuit'), 5.25711048134586)
    (('?', '2.'), 5.252090223854186)
    (('land', 'united'), 5.252090223854186)
    (('question', 'presented'), 5.2518161816763484)
    (('amend', 'his'), 5.247711783343611)
    (('provisions', 'statute'), 5.2449819905880375)
    (('2', 'scam.'), 5.232491046404844)
    (('rendered', 'against'), 5.224820943389355)
    (('debt', 'brought'), 5.223865041389244)
    (('constitution', 'states'), 5.221716574810667)
    (('him', 'remedy'), 5.221716574810667)
    (('with', 'cent'), 5.22010732194901)
    (('all', 'persons'), 5.21911201660364)
    (('(', 'vide'), 5.214696726115649)
    (('have', 'examined'), 5.213154561307244)
    (('al.', 'ill.'), 5.211185907318496)
    (('new', 'proceedings'), 5.208183419156974)
    (('p.', 'lockwood'), 5.2046430614517245)
    (('people', '3'), 5.2046430614517245)
    ((')', 'vide'), 5.203170472734621)
    (('first', 'day'), 5.202876135277537)
    (('law', 'requires'), 5.201993483037777)
    (('al.', '3'), 5.188818094290044)
    (('(', '1'), 5.1839002101579155)
    (('general', 'rule'), 5.183581445923895)
    (('for', 'purpose'), 5.17518440140849)
    (('reversal', 'judgment'), 5.173164829978557)
    (('act', 'entitled'), 5.1660467866313855)
    (('jurisdiction', 'subject'), 5.162822885757096)
    (('third', 'person'), 5.15255455030327)
    (('his', 'heirs'), 5.148176109792695)
    (('against', 'maker'), 5.142358783197384)
    (('as', 'regards'), 5.138976143802729)
    (('crouch', 'v.'), 5.1322345659236746)
    (('sawyer', 'v.'), 5.1322345659236746)
    (('sims', 'v.'), 5.1322345659236746)
    (('v.', 'al.'), 5.1322345659236746)
    (('v.', 'campbell'), 5.1322345659236746)
    (('v.', 'sprinkle'), 5.1322345659236746)
    (('as', 'follows'), 5.126903311502156)
    (('on', 'face'), 5.117734996228561)
    (('et', 'gilm.'), 5.112196812687131)
    (('plea', 'filed'), 5.112196812687131)
    (('second', 'section'), 5.106239357390731)
    (('2', 'johns'), 5.101928477233301)
    (('error', 'assigned'), 5.099813088231297)
    (('1', ')'), 5.090946357433044)
    (('3', 'rep.'), 5.087534758452861)
    (('proper', 'parties'), 5.062325735968738)
    (('no', 'doubt'), 5.060713268542356)
    (('an', 'improvement'), 5.044178389258478)
    (('give', 'evidence'), 5.037752120099045)
    (('reversed', 'costs'), 5.030951230086496)
    (('entitled', '“'), 5.027663302475464)
    (('affirmed', '('), 5.022745418343339)
    (('laws', 'p.'), 5.015981497427827)
    (('rendering', 'judgment'), 5.003239828536243)
    (('people', 'scam.'), 5.003009200282074)
    (('act', 'practice'), 5.002548054348505)
    (('any', 'offense'), 4.996173016347646)
    (('judgment', 'reversed'), 4.990961498757809)
    (('words', '“'), 4.987021317978115)
    (('an', 'amendment'), 4.985284700204911)
    (('those', 'cases'), 4.984677377509817)
    (('thompson', 'v.'), 4.980231472478625)
    (('motion', 'new'), 4.97831587725838)
    (('must', 'reversed'), 4.974834625703499)
    (('one', 'dollars'), 4.974693288937194)
    (('first', 'point'), 4.968890261954241)
    (('circuit', 'courts'), 4.967603864150874)
    (('2', 'rep.'), 4.966519357491496)
    (('assigned', 'error'), 4.962309564481362)
    (('judgment', 'rendered'), 4.960171106644358)
    (('at', 'subsequent'), 4.9491423283033384)
    (('tried', 'at'), 4.9491423283033384)
    (('an', 'facto'), 4.9446427157075625)
    (('power', 'make'), 4.942271811244817)
    (('taylor', 'v.'), 4.939589487981278)
    (('pay', 'costs'), 4.936314355948419)
    (('ante', 'justice'), 4.934753841960802)
    (('al', 'v.'), 4.922016858533327)
    (('by', 'virtue'), 4.921418495894846)
    (('other', 'than'), 4.920329258364141)
    (('are', 'satisfied'), 4.919240842589478)
    (('et', 'scam.'), 4.9105629515174805)
    (('reversed', 'lockwood'), 4.906962512811038)
    (('appeal', 'from'), 4.906243502060603)
    (('seal', 'state'), 4.901220435416436)
    (('person', 'who'), 4.9010157833073045)
    (('*', 'having'), 4.900862313274622)
    (('with', 'costs'), 4.898179227061647)
    (('v.', '11'), 4.89122646641988)
    (('for', 'purposes'), 4.890616834025053)
    (('”', 'c.'), 4.883186512586175)
    (('v.', 'ante'), 4.8769775106815985)
    (('mandamus', 'should'), 4.875941737968937)
    (('he', 'believed'), 4.869200160089882)
    (('v.', 'hall'), 4.869200160089882)
    (('defendant', 'pleaded'), 4.865841147999966)
    (('will', 'grant'), 4.8641645701925835)
    (('state', 'bank'), 4.863745729997772)
    (('statute', 'passed'), 4.861653351036534)
    (('without', 'notice'), 4.8583118439471455)
    (('1', 'scam.'), 4.849680709370459)
    (('an', 'assumpsit'), 4.8477811764549745)
    (('judgment', 'affirmed'), 4.841488758539631)
    (('up', 'opinion'), 4.8392469379882534)
    (('prayed', 'for'), 4.838149414130918)
    (('an', 'ex'), 4.829165498287628)
    (('appellant', 's'), 4.82790053105105)
    (('on', 'account'), 4.82227911270239)
    (('*', 'smith'), 4.818400153082649)
    (('affirmed', '*'), 4.818400153082649)
    (('this', 'respect'), 4.818400153082649)
    (('v.', 'id.'), 4.799659226836804)
    (('decree', 'reversed'), 4.7947530092250155)
    (('circuit', 'county'), 4.793709960505618)
    (('v.', '15'), 4.791197648088611)
    (('go', 'at'), 4.788677656110092)
    (('v.', '20'), 4.784311262503369)
    (('clearly', 'opinion'), 4.780353248934684)
    (('far', 'as'), 4.776406064418021)
    (('mason', 'v.'), 4.769664486538966)
    (('remanded', 'with'), 4.768543946645051)
    (('may', 'during'), 4.762284956173369)
    (('”', '&'), 4.759555163417797)
    (('with', 'per'), 4.750622038647789)
    (('plaintiff', 'recover'), 4.749315841147165)
    (('affirmed', ')'), 4.748184759128515)
    (('reversed', 'remanded'), 4.731390948227586)
    (('”', 'statutes'), 4.724149827286597)
    (('court', 'overruling'), 4.718458715208742)
    (('v.', 'miller'), 4.717197066644831)
    (('decide', 'whether'), 4.711658883103402)
    (('their', 'verdict'), 4.702142720922543)
    (('has', 'left'), 4.701144661359466)
    (('costs', 'suit'), 4.69927515864757)
    (('counsel', 'cause'), 4.697967946690136)
    (('v.', 'ill.'), 4.68682341760131)
    (('be', 'punished'), 4.678623361970857)
    (('other', 'officer'), 4.678623361970857)
    (('v.', '13'), 4.662749282622455)
    (('v.', '17'), 4.662749282622455)
    (('1', 'see'), 4.660524617507942)
    (('there', 'reason'), 4.657294852530347)
    (('has', 'been'), 4.651296111908902)
    (('1', 'affirmed'), 4.65077664095638)
    (('reverse', 'judgment'), 4.649602873921543)
    (('scott', 'v.'), 4.646807738753434)
    (('v.', '16'), 4.646807738753434)
    (('v.', 'armstrong'), 4.646807738753434)
    (('williams', 'v.'), 4.646807738753434)
    (('gave', '('), 4.638296238897915)
    (('sheriff', 'county'), 4.637469874174096)
    (('have', 'been'), 4.633625946636716)
    ((')', 'affirmed'), 4.627890525410804)
    (('v.', 'post'), 4.617661393093918)
    (('v.', '12'), 4.617661393093915)
    (('inquiry', 'before'), 4.614738610566016)
    (('received', 'evidence'), 4.60677307695118)
    (('circuit', 'court'), 4.600622224914883)
    (('cases', 'where'), 4.595004675986621)
    (('page', '1'), 4.591882951902811)
    (('judgment', 'obtained'), 4.588202329257399)
    (('an', 'action'), 4.586735323714638)
    (('had', 'vested'), 4.586128001019542)
    (('provide', 'for'), 4.575115008297125)
    (('sheriff', 's'), 4.574143938805268)
    (('there', 'no'), 4.569832011280006)
    (('id.', 'v.'), 4.568333680730349)
    (('’', 'p.'), 4.567053630079359)
    (('from', 'view'), 4.565206584225537)
    (('at', 'common'), 4.562673981221266)
    (('s', 'deed'), 4.558713898235663)
    (('an', 'appeal'), 4.5523252929288045)
    (('affirmed', 'costs'), 4.54727206520252)
    (('cornelius', 'v.'), 4.54727206520252)
    (('recover', 'his'), 4.5472720652025185)
    (('v.', '21'), 4.5472720652025185)
    (('right', 'appointment'), 4.545927521730654)
    (('(', 'affirmed'), 4.542119577436916)
    (('relied', 'on'), 4.532772495507405)
    (('no', 'way'), 4.532217719480229)
    (('supreme', 'court'), 4.532045590977862)
    (('or', 'replevy'), 4.530554670257585)
    (('inconsistent', 'this'), 4.528893535887665)
    (('appear', 'from'), 4.527731878806874)
    (('executed', 'note'), 4.527234311965973)
    (('note', 'executed'), 4.527234311965973)
    (('ill.', 'lockwood'), 4.526571156339088)
    (('set', 'declaration'), 4.525853081567613)
    (('appellant', '’'), 4.525576994103199)
    (('v.', 'scam.'), 4.515563205475182)
    (('changed', 'by'), 4.506380996616002)
    (('decide', 'question'), 4.492824281180143)
    (('*', '*'), 4.491899328405461)
    (('we', 'say'), 4.48991268576024)
    (('last', 'case'), 4.4863620305504455)
    (('must', 'costs'), 4.478966051811613)
    (('having', 'been'), 4.476722606888526)
    (('v.', '22'), 4.469269553201245)
    (('being', 'governor'), 4.4647631984679474)
    (('against', 'him'), 4.464286878084748)
    (('governor', 'state'), 4.455660991360697)
    (('court', 'erred'), 4.4554243093749495)
    (('reversed', 'laws'), 4.45526654295335)
    (('collins', 'v.'), 4.454162660811038)
    (('real', 'question'), 4.449755559288256)
    (('below', 'gave'), 4.440356861286007)
    (('below', 'pleaded'), 4.440356861286007)
    (('et', '3'), 4.434124907574493)
    (('vide', 'v.'), 4.431794847782584)
    (('page', '('), 4.423283347927066)
    (('on', '31st'), 4.421741183118661)
    (('be', 'construed'), 4.415588956137064)
    (('officer', 'who'), 4.415588956137064)
    (('purporting', 'be'), 4.415588956137064)
    (('said', 'hubbard'), 4.4146300361676385)
    (('by', 'parol'), 4.406845323065088)
    (('for', 'relator'), 4.405190006854813)
    (('an', 'exception'), 4.400322199483753)
    (('assault', 'and'), 4.398804371841502)
    (('1', 'reversed'), 4.39749021167415)
    (('appears', 'from'), 4.3960487697414194)
    (('moore', 'v.'), 4.395268971757467)
    (('v.', '19'), 4.395268971757467)
    (('does', 'not'), 4.389475692283822)
    (('act', 'legislature'), 4.3848813885546445)
    (('s', 'counsel'), 4.3826627939965075)
    (('gave', 'defendant'), 4.3804143208297255)
    (('has', 'provided'), 4.379216566472103)
    (('has', 'ever'), 4.379216566472101)
    (('general', 'demurrer'), 4.376226523866292)
    (('people', 'ill.'), 4.374568062894038)
    (('decisions', 'are'), 4.371753047286983)
    (('scam.', 'et'), 4.369994570154777)
    (('below', 'reversed'), 4.36506873398177)
    (('regarded', 'as'), 4.361368565139179)
    (('v.', '14'), 4.354626987260122)
    (('sale', 'made'), 4.346954731918197)
    (('not', 'inconsistent'), 4.344387802755286)
    (('this', 'allegation'), 4.332973325912407)
    (('show', 'cause'), 4.332683482608392)
    (('an', 'trespass'), 4.329932871592357)
    (('merits', 'case'), 4.322863298267567)
    (('law', 'relation'), 4.322287716755488)
    (('appointment', 'made'), 4.31625793469528)
    (('v.', 'id'), 4.312806811565496)
    (('v.', 'gilm.'), 4.3050711627858895)
    (('are', 'called'), 4.297752465843207)
    (('below', 'affirmed'), 4.296427068376641)
    (('v.', 'people'), 4.2926992381169224)
    ((')', '*'), 4.2878854363838705)
    (('he', 'became'), 4.284237659368726)
    (('sheriff', 'other'), 4.2790159032616195)
    (('sheriff', '’'), 4.271820401857417)
    (('right', 'property'), 4.270293079117227)
    (('v.', '3'), 4.269738089673609)
    (('at', 'law'), 4.268420844776752)
    (('johns.', '('), 4.259784615644186)
    (('(', 'reversed'), 4.254885816231347)
    (('(', 'see'), 4.254885816231347)
    (('replevy', 'for'), 4.253186913409763)
    (('v.', 'et'), 4.2515334942960195)
    (('can', 'doubt'), 4.247711783343609)
    (('can', 'sustained'), 4.247711783343609)
    (('persons', 'are'), 4.247126392773238)
    (('under', 'section'), 4.24589134771924)
    ((')', 'reversed'), 4.243359562850319)
    ((')', 'see'), 4.243359562850319)
    (('by', 'default'), 4.24334659078221)
    (('disclosed', 'by'), 4.24334659078221)
    (('et', 'ill.'), 4.241479829632095)
    (('therefore', 'reversed'), 4.239537851897911)
    (('s', 'rep.'), 4.230659700551458)
    (('ought', 'have'), 4.228261453697453)
    (('created', 'by'), 4.220978777753755)
    (('court', 'below'), 4.210117496090501)
    (('gilm.', 'v.'), 4.208855847526589)
    (('s', 'p.'), 4.206412154304779)
    (('v.', '4'), 4.203787826948707)
    (('give', 'opinion'), 4.195390748213528)
    (('must', 'considered'), 4.194512662342909)
    (('any', 'other'), 4.188818094290044)
    (('had', 'notice'), 4.186197394130907)
    (('consideration', 'note'), 4.1793110085456675)
    (('is', 'presumed'), 4.1783637366528765)
    (('appears', 'record'), 4.176802096549949)
    (('do', 'so'), 4.175761941464595)
    (('an', 'debt'), 4.16970927134234)
    (('with', 'great'), 4.165659537926633)
    (('on', 'day'), 4.165401429858875)
    (('plaintiff', 'error'), 4.164353340426009)
    (('second', 'plea'), 4.162822885757098)
    (('good', 'action'), 4.148722688712244)
    (('not', 'noticed'), 4.148467592780028)
    (('costs', 'cause'), 4.143205683744679)
    (('might', 'have'), 4.139153979863467)
    (('under', 'consideration'), 4.138976143802731)
    (('note', 'assigned'), 4.138192021220073)
    (('see', 'et'), 4.136444358933806)
    (('affirmed', '1'), 4.13620346812662)
    (('v.', '7'), 4.132234565923676)
    (('post', 'v.'), 4.1322345659236746)
    (('v.', 'johns.'), 4.1322345659236746)
    (('v.', 'wash'), 4.1322345659236746)
    (('parties', 'their'), 4.123254734611068)
    (('only', 'question'), 4.121314157442219)
    (('plaintiff', 'take'), 4.108718641453191)
    (('it', 'contended'), 4.108274666979494)
    (('are', 'opinion'), 4.107927906963191)
    (('those', 'are'), 4.105107387900809)
    (('courts', 'state'), 4.104753829501568)
    (('errors', 'been'), 4.104753829501567)
    (('lockwood', 'an'), 4.094219071758074)
    (('it', 'perceived'), 4.089165844031788)
    (('present', 'case'), 4.088398044630544)
    (('’', 'deed'), 4.0864653598455)
    (('remark', 'that'), 4.08509158252453)
    (('defendants', 'below'), 4.084481434475306)
    (('’', 'counsel'), 4.0803392570486565)
    (('’', 'rep.'), 4.0803392570486565)
    (('they', 'are'), 4.06948347817009)
    (('without', 'being'), 4.065493015047421)
    (('grounds', 'which'), 4.063287212206182)
    (('governor', 'power'), 4.06256604496253)
    (('under', 'general'), 4.060199030418612)
    (('(', '2'), 4.0571554491291675)
    (('cause', 'new'), 4.049229535535666)
    (('an', 'examination'), 4.044178389258478)
    (('can', 'considered'), 4.044178389258478)
    (('governor', 'shall'), 4.0440300935251035)
    (('we', 'are'), 4.043995873597423)
    (('court', 'presume'), 4.040386810096106)
    (('judgment', '*'), 4.039309083243765)
    (('reversed', '('), 4.0324933948948996)
    (('no', 'than'), 4.029263629917304)
    (('cause', 'proceedings'), 4.021643703863763)
    (('according', 'law'), 4.021421237395954)
    (('it', 'doubtful'), 4.018776516140392)
    (('it', 'unnecessary'), 4.018776516140392)
    (('errors', 'have'), 4.016757348503738)
    (('justice', 'jurisdiction'), 4.014924190644784)
    (('question', 'here'), 4.013656444481583)
    (('note', 'due'), 4.012661139136215)
    (('on', 'november'), 4.006703683839817)
    (('why', 'not'), 4.005863197337227)
    (('had', 'jurisdiction'), 4.005281608770618)
    (('et', 'v.'), 4.002174025359302)
    (('v.', '24'), 3.99473104217374)
    (('sued', 'for'), 3.990152507575969)
    (('given', 'evidence'), 3.985284700204909)
    (('as', 'man'), 3.982856941885448)
    (('never', 'been'), 3.9792229474177088)
    (('ever', 'been'), 3.979222947417707)
    (('are', 'bound'), 3.975824370955845)
    (('authorities', 'are'), 3.975824370955845)
    (('proceedings', 'had'), 3.9746932889371944)
    (('as', 'well'), 3.9743454420299305)
    (('which', 'refused'), 3.970177807814702)
    (('v.', 'johns'), 3.9699631370247985)
    (('said', 'coles'), 3.958950552391448)
    (('an', 'appearance'), 3.9567155480081393)
    (('under', 'issue'), 3.9553762054633417)
    (('estate', 'shall'), 3.9509206891336213)
    (('a', 'future'), 3.949660790474189)
    (('on', 'ground'), 3.9478099947862475)
    (('p.', 'justice'), 3.9478099947862475)
    (('neither', 'or'), 3.945592169536429)
    (('reversed', ')'), 3.9385049813218984)
    (('not', 'appear'), 3.9334547018091772)
    (('must', 'show'), 3.9314782565091164)
    (('or', 'other'), 3.9310925998413158)
    (('had', 'been'), 3.9285968743477397)
    (('must', 'affirmed'), 3.928560773126822)
    (('id', 'v.'), 3.9257836884562476)
    (('plaintiffs', 'below'), 3.9257836884562476)
    (('we', 'might'), 3.9228720930363448)
    (('diligence', 'by'), 3.921418495894848)
    (('considered', 'as'), 3.9207959737531954)
    (('costs', '('), 3.9192408425894776)
    (('justice', 'having'), 3.9192408425894776)
    (('is', 'defective'), 3.915329330819082)
    (('demurrer', 'declaration'), 3.914895372313513)
    (('on', 'point'), 3.9106569963687825)
    (('final', 'judgment'), 3.9101304241447643)
    (('on', 'grounds'), 3.907168010288901)
    (('either', 'or'), 3.9025234476445423)
    (('must', 'be'), 3.899823965774935)
    (('following', '“'), 3.8989299883532595)
    (('in', 'abatement'), 3.8988686870704132)
    (('view', 'case'), 3.8822907068815855)
    (('reverse', 'this'), 3.8798006977467914)
    (('for', 'payment'), 3.8746752901560324)
    (('et', '4'), 3.8734099531000137)
    (('co.', 'v.'), 3.869200160089882)
    (('v.', '8'), 3.869200160089882)
    (('equity', 'will'), 3.8641645701925835)
    (('said', 'did'), 3.855857059427345)
    (('for', 'amount'), 3.851088469838416)
    (('state', 'facts'), 3.8479784141394546)
    (('an', 'indictment'), 3.8477811764549745)
    (('an', 'original'), 3.8477811764549745)
    (('claim', 'been'), 3.841719423667774)
    (('costs', ')'), 3.840600393349913)
    (('justice', '*'), 3.8404264594126474)
    (('ought', 'reversed'), 3.839607245009276)
    (('for', 'years'), 3.8381494141309194)
    (('for', 'relief'), 3.8381494141309176)
    (('security', 'for'), 3.8381494141309176)
    (('upon', 'affidavit'), 3.834725023629529)
    (('can', 'not'), 3.826539497892668)
    (('it', 'conceded'), 3.8261314381979954)
    ((';', 'nor'), 3.824953247000714)
    (('question', 'whether'), 3.816740929160959)
    (('entitled', 'an'), 3.815359698762599)
    (('he', 'himself'), 3.810306471036313)
    (('on', 'oath'), 3.810306471036313)
    (('granting', 'which'), 3.8002528063723897)
    (('bryan', 'and'), 3.799342301425229)
    (('sufficient', 'cause'), 3.799251282527317)
    ((';', 'but'), 3.795806901341198)
    (('declares', 'that'), 3.7955849653295477)
    (('provides', 'that'), 3.7955849653295477)
    (('will', 'necessary'), 3.791595944241937)
    (('them', 'they'), 3.7877125097547086)
    (('not', 'necessarily'), 3.7858975133953194)
    (('on', 'merits'), 3.784311262503369)
    (('an', 'inquiry'), 3.783650839035259)
    (('point', 'we'), 3.7771946378407115)
    (('see', 'v.'), 3.773153472618846)
    (('his', 'costs'), 3.769664486538966)
    (('was', 'properly'), 3.7685439466450497)
    (('i', 'therefore'), 3.7646288966416694)
    (('be', 'inconsistent'), 3.76351225955737)
    (('is', 'perceived'), 3.763326237374031)
    (('reynolds', 'an'), 3.7628922788684633)
    (('it', 'appears'), 3.756969413690598)
    (('therefore', 'affirmed'), 3.7558586870139354)
    (('declared', 'on'), 3.7551649168438512)
    (('2', ')'), 3.7501733122219694)
    (('they', 'were'), 3.7501293269465528)
    (('i', 'opinion'), 3.744729339203964)
    (('be', 'presumed'), 3.7375170510244278)
    (('contract', 'made'), 3.731295433974122)
    (('may', 'make'), 3.7298634784809916)
    (('on', 'argument'), 3.7291697083109074)
    (('under', 'laws'), 3.7291003496396655)
    (('page', 'justice'), 3.710770797485397)
    ((')', '('), 3.710352685602775)
    (('?', 'i'), 3.7018931412937057)
    (('have', 'done'), 3.698581388477484)
    (('reversed', 'cause'), 3.6979679466901363)
    (('is', 'doubtful'), 3.6929369094826345)
    (('is', 'evident'), 3.6929369094826345)
    (('had', 'power'), 3.6916211298668564)
    (('must', 'therefore'), 3.6875526736230277)
    (('post', 'law'), 3.687420310208017)
    (('a', 'trespasser'), 3.6866263846403964)
    (('evidence', 'jury'), 3.6866263846403964)
    (('a', 'popular'), 3.6866263846403946)
    (('a', 'quo'), 3.6866263846403946)
    (('under', 'constitution'), 3.6832966600265404)
    (('it', 'necessary'), 3.6831734843559545)
    (('been', 'counsel'), 3.6815423987770224)
    (('was', 'tried'), 3.6810811053947106)
    (('in', 'favor'), 3.6764762657339674)
    (('in', 'pursuance'), 3.6764762657339674)
    (('in', 'relation'), 3.6764762657339674)
    (('in', 'respect'), 3.6764762657339674)
    (('in', 'saying'), 3.6764762657339674)
    (('whether', 'ought'), 3.6716242213011476)
    (('where', 'there'), 3.67047024127808)
    (('before', 'sheriff'), 3.669531083406394)
    (('cause', 'action'), 3.663736311870835)
    (('an', 'instrument'), 3.6633566053175475)
    (('and', 'carried'), 3.6618387776752943)
    (('and', 'carrying'), 3.6618387776752943)
    (('and', 'remained'), 3.6618387776752943)
    (('and', 'sisters'), 3.6618387776752943)
    (('there', 'demurrer'), 3.6572948525303453)
    (('scam.', 'v.'), 3.6530667292251167)
    (('note', 'set'), 3.6527651940498327)
    (('an', 'injunction'), 3.6518609664797186)
    (('declaration', 'states'), 3.6518609664797186)
    (('v.', '2'), 3.6434873804618206)
    (('can', 'now'), 3.643448726663518)
    (('established', 'by'), 3.641310576702111)
    (('verdict', 'against'), 3.6398584426681975)
    (('lockwood', 'this'), 3.6378279074408297)
    (('.', 'ev.'), 3.637469874174096)
    (('rev', '.'), 3.637469874174096)
    (('can', 'say'), 3.6367540740895112)
    (('court', 'chancery'), 3.6309958739584047)
    (('question', 'decided'), 3.6303278049300776)
    (('recover', '('), 3.6297342253944933)
    (('be', 'considered'), 3.6270930613307772)
    (('being', 'then'), 3.6237527222291064)
    (('counsel', 'for'), 3.618471377491508)
    (('erred', 'in'), 3.6150757210698234)
    (('these', 'are'), 3.6132542915711348)
    (('would', 'have'), 3.612367093424405)
    (('therefore', 'opinion'), 3.610428247492374)
    (('for', 'reason'), 3.609330723635038)
    (('r.', 'v.'), 3.6086726098666624)
    (('defendant', 's'), 3.6045758350192294)
    (('“', '”'), 3.603637019969364)
    (('been', 'commenced'), 3.6007113241639797)
    (('with', 'interest'), 3.5986189452027393)
    (('person', 'shall'), 3.597965698625634)
    (('well', 'as'), 3.5958338187762013)
    (('party', 'who'), 3.5951383764677782)
    (('is', 'apparent'), 3.5934012359317187)
    (('on', 'subject'), 3.5916661845609745)
    (('jury', 'should'), 3.588779061242249)
    (('appeal', 'had'), 3.586128001019542)
    (('any', 'matter'), 3.584746770621182)
    (('could', 'have'), 3.5845651281858295)
    (('did', 'not'), 3.583268346880301)
    (('indenture', 'was'), 3.5826774013337186)
    (('”', 'section'), 3.5826774013337186)
    (('judgment', 'below'), 3.5794321196296686)
    (('proceedings', 'before'), 3.5793332744348163)
    (('ought', 'been'), 3.579292340529074)
    (('circuit', 'affirmed'), 3.575286441372114)
    (('under', 'statute'), 3.574772203248372)
    (('was', 'rendered'), 3.5741659014781995)
    (('has', 'jurisdiction'), 3.571861644414499)
    (('this', 'view'), 3.570472639639064)
    (('defendant', 'counsel'), 3.568160599359281)
    (('for', 'new'), 3.5672955038378262)
    (('.', 'cas.'), 3.5670805462826998)
    (('ch', '.'), 3.5670805462826998)
    (('a', 'total'), 3.5610955025565367)
    (('who', 'are'), 3.560786871677003)
    (('this', 'appeal'), 3.5553657472488567)
    ((')', 'judgment'), 3.553343572045751)
    (('first', 'question'), 3.5523252929288027)
    (('note', 'out'), 3.551481858212652)
    (('general', 'we'), 3.551313230424382)
    (('*', 'justice'), 3.550919842217663)
    (('was', 'intention'), 3.546151525308602)
    (('should', 'have'), 3.5454516295781495)
    (('2', 'affirmed'), 3.5417338816610897)
    (('any', 'person'), 3.541119838220922)
    (('.', '38'), 3.537934200623182)
    (('brought', 'against'), 3.533549540521859)
    (('on', 'part'), 3.5327724955074036)
    (('has', 'some'), 3.531219659917154)
    (('before', 'justice'), 3.530970252873418)
    (('aside', 'judgment'), 3.5293086402038316)
    (('after', 'trial'), 3.528789778468017)
    (('do', 'act'), 3.528616866016092)
    (('can', 'be'), 3.527097271354055)
    (('below', 'ought'), 3.5258530815676146)
    (('reversed', 'with'), 3.5233308178985006)
    (('there', 'assigned'), 3.5197913287804123)
    (('application', 'for'), 3.516221319243556)
    (('have', 'effect'), 3.512714843166151)
    (('an', 'act'), 3.5030817739089564)
    (('manner', 'which'), 3.4962466194822905)
    (('an', 'officer'), 3.4895895375808426)
    (('did', 'upon'), 3.487705947779336)
    (('for', 'days'), 3.487652167046786)
    (('was', 'contended'), 3.4872578362550364)
    (('judgment', 'entered'), 3.4838656694426664)
    (('in', 'degree'), 3.4838311877915693)
    (('v.', 'cranch'), 3.48015786934398)
    (('was', 'erroneous'), 3.4757621974172057)
    (('has', 'decided'), 3.4723259708635847)
    (('brought', 'before'), 3.4705976348939362)
    (('a', 'compliance'), 3.464233963303947)
    (('without', 'any'), 3.4623831676160055)
    (('by', 'means'), 3.4619868772575497)
    (('disposition', 'of'), 3.4611154214528046)
    (('defendant', 'error'), 3.454908047053859)
    (('facts', 'case'), 3.454653170823109)
    (('in', 'parts'), 3.454083844397518)
    (('all', 'cases'), 3.4477363916516026)
    (('was', 'committed'), 3.446615851757688)
    (('was', 'overruled'), 3.446615851757688)
    (('on', 'trial'), 3.4408500060663663)
    (('due', 'one'), 3.4403568612860074)
    (('no', 'power'), 3.439771470715634)
    (('said', 'office'), 3.4395763932978696)
    (('and', 'cents'), 3.4394463563388467)
    (('it', 'duty'), 3.4262008313093606)
    (('shall', 'be'), 3.425784723242952)
    (('a', 'nonsuit'), 3.423591978806602)
    (('remanded', 'for'), 3.4231119148520754)
    (('was', 'argument'), 3.420620643224744)
    (('it', 'seem'), 3.4167405020602946)
    (('and', 'remanded'), 3.4167262798387625)
    (('be', 'amended'), 3.4155889561370643)
    (('be', 'awarded'), 3.4155889561370643)
    (('let', 'be'), 3.4155889561370643)
    (('presume', 'that'), 3.4070196774118955)
    (('1', 'johns'), 3.4036163144709874)
    (('parties', 'may'), 3.4033626538038035)
    (('”', 'see'), 3.4004740701129688)
    (('1', 'gilm.'), 3.3992378739604128)
    (('al.', '1'), 3.3992378739604128)
    (('reversed', '1'), 3.3974902116741497)
    (('1', 'judgment'), 3.397259546546003)
    (('a', 'juror'), 3.3971197674454103)
    (('debt', 'on'), 3.395268971757467)
    (('entry', 'or'), 3.3930511465076503)
    (('construed', 'to'), 3.3929648066984797)
    (('promised', 'to'), 3.3929648066984797)
    (('they', 'do'), 3.3902333818601686)
    (('reynolds', 'this'), 3.3895568542787746)
    (('it', 'appear'), 3.388726125890697)
    (('ill.', 'justice'), 3.385215307093546)
    (('before', 'first'), 3.3757998803496836)
    (('received', 'from'), 3.3757287853618223)
    (('.', 'crim'), 3.3744354683403035)
    (('bac', '.'), 3.3744354683403035)
    (('ibid', '.'), 3.3744354683403035)
    (('what', 'they'), 3.3738389436095026)
    (('constitution', 'state'), 3.3715403701964686)
    (('not', 'thereby'), 3.3708600141164755)
    (('surely', 'not'), 3.3708600141164755)
    (('can', 'received'), 3.3692737632245233)
    (('?', 'we'), 3.368283241358707)
    (('been', 'assigned'), 3.3677882353353628)
    (('right', 'office'), 3.3675902804721414)
    (('he', 'liable'), 3.366699819560699)
    (('it', 'urged'), 3.366699819560699)
    (('unless', 'he'), 3.366699819560699)
    (('gave', 'judgment'), 3.3658099079209514)
    (('1', 'note'), 3.3632585768548484)
    (('to', 'settle'), 3.36321746330443)
    (('facts', 'are'), 3.359153010507349)
    (('which', 'might'), 3.348689431068431)
    (('an', 'execution'), 3.347210863024193)
    (('may', 'be'), 3.345199628245666)
    (('court', 'st.'), 3.339947091955013)
    (('no', 'jurisdiction'), 3.3383944503405534)
    (('ought', 'be'), 3.337586444135791)
    (('by', 'fraud'), 3.33645599517369)
    (('by', 'granting'), 3.33645599517369)
    (('liable', 'for'), 3.3356490736017363)
    (('are', 'presented'), 3.3342783418683215)
    (('is', 'applicable'), 3.330366830097926)
    (('is', 'insufficient'), 3.330366830097926)
    (('had', 'legal'), 3.3297882477597565)
    (('instrument', 'which'), 3.3263216180399784)
    (('other', 'were'), 3.3263216180399784)
    (('court', 'grant'), 3.3261412924299822)
    (('erred', 'judgment'), 3.3251679234236065)
    (('v.', '5'), 3.3248796438660726)
    (('a', 'patent'), 3.324056305255686)
    (('suit', 'against'), 3.3203570851753756)
    (('we', 'opinion'), 3.318010034877485)
    (('be', 'reversed'), 3.3143056202998835)
    (('in', 'refusing'), 3.313906186349257)
    (('where', 'has'), 3.3114719598362683)
    (('violation', 'of'), 3.309112328007755)
    (('speaking', 'of'), 3.3091123280077532)
    (('a', 'corporation'), 3.3081147613866655)
    (('(', 'judgment'), 3.3080944100646654)
    (('say', 'that'), 3.3074840038609796)
    (('distinction', 'is'), 3.3038946187367344)
    (('court', 'refused'), 3.303421215929898)
    (('defendant', '’'), 3.3022522980713784)
    (('be', 'deemed'), 3.3001117387171277)
    (('affirmed', 'with'), 3.2911904200104924)
    (('correct', 'as'), 3.2909792372477806)
    (('in', 'way'), 3.289453142624719)
    (('could', 'not'), 3.2887589339015832)
    (('can', 'assigned'), 3.288353767840956)
    (('he', 'executed'), 3.284237659368726)
    (('?', 'if'), 3.28356056245328)
    (('an', 'affidavit'), 3.2816777029851334)
    (('his', 'office'), 3.2808786472788096)
    (('no', 'opinion'), 3.277852908405503)
    (('at', 'all'), 3.2772717623590157)
    (('275', '.'), 3.2748997947893876)
    (('has', 'legal'), 3.274879906657368)
    (('a', 'departure'), 3.2715888853615507)
    (('a', 'failure'), 3.2715888853615507)
    (('is', 'private'), 3.2714731410443587)
    (('plaintiff', 's'), 3.2709524064994895)
    (('in', 'manner'), 3.261438766455125)
    (('contained', 'in'), 3.2614387664551234)
    (('in', 'equal'), 3.2614387664551217)
    (('has', 'right'), 3.2594773221980073)
    (('has', 'power'), 3.2582011655107372)
    (('such', 'power'), 3.25753617585117)
    (('“', 'shall'), 3.2544341643224843)
    (('for', 'sum'), 3.2531869134097633)
    (('provided', 'for'), 3.2531869134097615)
    (('sec', '.'), 3.2451524513953363)
    (('vacancy', 'by'), 3.2433465907822097)
    (('himself', 'state'), 3.242257353251503)
    (('not', 'grant'), 3.24157699717151)
    (('nor', 'such'), 3.2377729210297392)
    (('apparent', 'that'), 3.2370946759695816)
    (('this', 'subject'), 3.233437652361493)
    (('wilson', 'this'), 3.233437652361493)
    (('a', 'continuance'), 3.227194766003098)
    (('new', 'had'), 3.2246715419455434)
    (('point', 'are'), 3.222464338538968)
    (('not', 'final'), 3.218856920671424)
    (('nor', 'any'), 3.2179644399495597)
    (('“', '*'), 3.2157356506280337)
    (('people', 'v.'), 3.2146967261156494)
    (('and', 'morrison'), 3.214379800704073)
    (('said', ':'), 3.2100123165591654)
    (('certificates', 'of'), 3.209576654456839)
    (('one-half', 'of'), 3.209576654456839)
    (('3', 's'), 3.206412154304779)
    (('court', 'called'), 3.2038855423789823)
    (('afford', 'a'), 3.2011995574701544)
    (('from', 'record'), 3.1935254541410742)
    (('from', 'what'), 3.1935254541410742)
    (('in', 'overruling'), 3.1910494385637254)
    (('been', 'made'), 3.189901948190366)
    (('we', 'have'), 3.1849576694765904)
    (('overruled', 'by'), 3.1844529017286405)
    (('madison', 'for'), 3.1827975855183652)
    (('trespass', 'for'), 3.1827975855183652)
    (('?', 'would'), 3.1785497588362137)
    (('be', 'sustained'), 3.1785497588362137)
    (('is', 'prosecuted'), 3.1783637366528765)
    (('debet', 'is'), 3.1783637366528747)
    (('nil', 'is'), 3.1783637366528747)
    (('remanded', 'court'), 3.1778903338460402)
    (('and', 'others'), 3.176411950505054)
    (('he', 'do'), 3.1728765504210212)
    (('assigned', 'for'), 3.172266918026194)
    (('a', 'perfect'), 3.1720532118106366)
    (('cause', 'circuit'), 3.1712200599142744)
    (('to', 'ascertain'), 3.1705723853620356)
    (('adhered', 'to'), 3.170572385362034)
    (('constrained', 'to'), 3.170572385362034)
    (('inclined', 'to'), 3.170572385362034)
    (('reference', 'to'), 3.170572385362034)
    (('refused', 'to'), 3.170572385362034)
    (('to', 'amend'), 3.170572385362034)
    (('to', 'reverse'), 3.170572385362034)
    (('according', 'to'), 3.170572385362032)
    (('bound', 'to'), 3.170572385362032)
    (('compelled', 'to'), 3.170572385362032)
    (('confined', 'to'), 3.170572385362032)
    (('lead', 'to'), 3.170572385362032)
    (('ought', 'to'), 3.170572385362032)
    (('purporting', 'to'), 3.170572385362032)
    (('relate', 'to'), 3.170572385362032)
    (('to', 'commit'), 3.170572385362032)
    (('to', 'enable'), 3.170572385362032)
    (('to', 'fill'), 3.170572385362032)
    (('to', 'wit'), 3.170572385362032)
    (('himself', 'from'), 3.169277907894397)
    (('must', 'with'), 3.1639907175660156)
    (('tried', 'by'), 3.1558837495318706)
    (('reynolds', 'was'), 3.153834102529844)
    (('was', 'held'), 3.153834102529842)
    (('ground', 'for'), 3.1536512398588457)
    (('be', 'allowed'), 3.1525545503032717)
    (('be', 'proved'), 3.1525545503032717)
    (('claimed', 'be'), 3.1525545503032717)
    (('we', 'will'), 3.150282589139419)
    (('motion', 'for'), 3.148850253595027)
    (('do', 'not'), 3.148467592780028)
    (('id', '.'), 3.145616777844424)
    (('contended', 'that'), 3.1416751108908993)
    (('be', 'paid'), 3.135481036944329)
    (('he', 'might'), 3.1322345659236763)
    (('received', 'any'), 3.1322345659236746)
    (('been', 'some'), 3.13122604086276)
    (('*', 'action'), 3.1303441593973904)
    (('assigned', 'but'), 3.1299244052364728)
    (('plaintiff', 'below'), 3.1278274644008945)
    (('3', '’'), 3.1264810386933757)
    (('it', 'seems'), 3.1256917200569028)
    (('was', 'granted'), 3.124687756870326)
    (('justice', 'counsel'), 3.124060634477976)
    (('rep.', 'justice'), 3.124060634477976)
    (('reversed', 'justice'), 3.124060634477976)
    (('is', 'bound'), 3.1194700475993073)
    (('chief', 'this'), 3.1179604349415584)
    (('from', 'decision'), 3.116166571434835)
    (('by', 'judge'), 3.1140635738372424)
    (('was', 'competent'), 3.1131921180324973)
    (('note', 'given'), 3.112196812687131)
    (('presented', 'case'), 3.1078504072967146)
    (('any', 'than'), 3.1047538295015684)
    (('a', 'private'), 3.1016638839192385)
    (('a', 'tract'), 3.1016638839192385)
    (('grant', 'a'), 3.1016638839192385)
    (('is', 'settled'), 3.1003612246516035)
    (('justice', 'decided'), 3.099813088231297)
    (('v.', 'smith'), 3.099813088231297)
    (('is', 'necessary'), 3.0942994718644012)
    (('in', 'regard'), 3.0915137650128095)
    (('named', 'in'), 3.0915137650128095)
    (('vested', 'in'), 3.0915137650128095)
    (('so', 'as'), 3.0905146548460856)
    (('sale', 'an'), 3.0899820788716035)
    (('referred', 'to'), 3.0881102251700607)
    (('costs', '1'), 3.087840446565222)
    (('ill.', 'v.'), 3.0864308763105477)
    (('suit', 'at'), 3.0853204545867072)
    (('johns', '.'), 3.0828810224964585)
    (('he', 'did'), 3.0816084928537055)
    (('and', 'away'), 3.07687627695414)
    (('and', 'consequently'), 3.0768762769541382)
    (('brothers', 'and'), 3.0768762769541382)
    (('below', 'must'), 3.07611796154068)
    (('this', 'case'), 3.076051362219541)
    (('1', 'rep.'), 3.0755621167867844)
    (('had', 'right'), 3.073377563796601)
    (('been', 'decided'), 3.072332351809191)
    (('plaintiff', 'costs'), 3.071243936034527)
    (('cause', 'opinion'), 3.0688583422845976)
    (('is', 'equally'), 3.06288651923294)
    (('judgment', 'must'), 3.0604646972291967)
    (('duty', 'court'), 3.055493702486313)
    (('suit', 'should'), 3.053940039946932)
    (('been', 'settled'), 3.053223528861487)
    (('.', 'dane'), 3.0525073734529418)
    (('ills', '.'), 3.0525073734529418)
    (('.', 'ch'), 3.05250737345294)
    (('.', 'rep.'), 3.05250737345294)
    (('not', 'lie'), 3.0489319192291138)
    (('said', 'being'), 3.0474872269931037)
    (('on', 'stated'), 3.0473456683371616)
    (('administration', 'of'), 3.0460779221739607)
    (('his', 'estate'), 3.0460779221739607)
    (('of', '1827'), 3.0460779221739607)
    (('of', 'limitations'), 3.0460779221739607)
    (('is', 'merely'), 3.040860212902942)
    (('is', 'objected'), 3.040860212902942)
    (('as', 'bound'), 3.039440470251817)
    (('should', 'be'), 3.039440470251815)
    (('relation', 'to'), 3.0393278520837814)
    (('by', 'counsel'), 3.0387754465330072)
    (('before', 'him'), 3.0387648930721127)
    (('“', 'act'), 3.0373134725091813)
    (('be', 'manifest'), 3.037077332883335)
    (('he', 'could'), 3.037077332883335)
    (('seems', 'be'), 3.037077332883335)
    (('a', 'copy'), 3.0345496880607)
    (('has', 'brought'), 3.033441729630372)
    (('leave', 'to'), 3.0330688616120973)
    (('of', 'illinois'), 3.03227212264893)
    (('shall', 'power'), 3.0315391493419046)
    (('gave', 'for'), 3.030794492073314)
    (('remanded', 'to'), 3.0285533804896048)
    (('commenced', 'by'), 3.0283336998113572)
    (('any', 'land'), 3.0253193620071634)
    (('in', 'hands'), 3.024399569154273)
    (('judgment', 'against'), 3.0234177104738738)
    (('for', 'proceedings'), 3.016147716108911)
    (('this', 'gave'), 3.0110452310250455)
    (('davidson', 'and'), 3.0097620810956)
    (('submission', 'and'), 3.0097620810956)
    (('could', 'made'), 3.0095965964612272)
    (('a', 'promissory'), 3.008554479527758)
    (('on', 'whom'), 3.0067036838398167)
    (('has', 'received'), 3.0007049432183734)
    (('regard', 'to'), 3.0006473839197216)
    (('be', 'admitted'), 3.000551456858222)
    (('in', 'consequence'), 2.998404360621329)
    (('in', 'kentucky'), 2.998404360621329)
    (('in', 'offices'), 2.998404360621329)
    (('made', 'parties'), 2.9973702941279683)
    (('there', 'being'), 2.9973702941279683)
    (('trial', 'cause'), 2.9892222261713854)
    (('certificate', 'of'), 2.9871842331203915)
    (('passage', 'of'), 2.9871842331203915)
    (('a', 'vide'), 2.9861866664993038)
    (('and', 'wife'), 2.983766872562658)
    (('ankeny', 'and'), 2.983766872562658)
    (('dollars', 'for'), 2.980168419003345)
    (('if', 'had'), 2.980168419003345)
    (('might', 'been'), 2.9792229474177105)
    (('to', 'impeach'), 2.9779273074196357)
    (('said', 'execution'), 2.9762285438232823)
    (('in', 'substance'), 2.9760365475928765)
    (('or', 'officer'), 2.975965818579949)
    (('no', 'other'), 2.975824370955845)
    (('he', 'had'), 2.974693288937196)
    (('whether', 'he'), 2.9746932889371944)
    (('first', 'may'), 2.973789061367082)
    (('such', 'would'), 2.9715873780693727)
    (('ill.', ';'), 2.970177807814702)
    (('3.', 'that'), 2.9696143651045954)
    (('satisfied', 'that'), 2.9696143651045954)
    (('plaintiff', '’'), 2.9686288695516385)
    (('presented', 'for'), 2.9677846945475146)
    (('for', 'error'), 2.967784694547513)
    (('a', 'mandamus'), 2.9667343038331317)
    (('one', 'who'), 2.9664256729535943)
    (('virtue', 'of'), 2.9651579267903934)
    (('amount', 'judgment'), 2.963711464349606)
    (('had', 'no'), 2.963333426772648)
    (('action', 'against'), 2.9624496931824496)
    (('smith', 'v.'), 2.9623095644813624)
    (('was', 'executed'), 2.9611890245874477)
    (('was', 'proved'), 2.961189024587446)
    (('whom', 'was'), 2.961189024587446)
    (('is', 'unnecessary'), 2.955971315316427)
    (('judgment', 'laws'), 2.9559341137578876)
    (('consent', 'not'), 2.9558225148376316)
    (('a', 'proceeding'), 2.949660790474189)
    (('able', 'to'), 2.9481799640255844)
    (('enable', 'to'), 2.9481799640255844)
    (('offer', 'to'), 2.9481799640255844)
    (('to', 'serve'), 2.9481799640255844)
    (('authorities', 'on'), 2.9478099947862475)
    (('which', 'presented'), 2.9478099947862475)
    (('that', 'came'), 2.9475880587745973)
    (('they', 'so'), 2.9474929537914782)
    (('was', 'taken'), 2.9423299973361328)
    (('”', 's'), 2.9343889652437323)
    (('laws', 'state'), 2.9319172326393534)
    (('contended', 'for'), 2.9312588185224)
    (('if', 'were'), 2.9312588185224)
    (('sufficient', 'for'), 2.9312588185224)
    (('be', 'payable'), 2.930162128966824)
    (('seems', 'to'), 2.9295642858582376)
    (('when', 'note'), 2.9289749886313583)
    ((':', 'where'), 2.9269864355379056)
    (('decision', 'made'), 2.926111610958147)
    (('they', 'have'), 2.925991884580558)
    (('would', 'him'), 2.9257836884562476)
    (('office', 'said'), 2.92500322046811)
    (('perfect', 'and'), 2.924873183509087)
    (('al', '.'), 2.9247518262545675)
    (('is', 'now'), 2.9220239833930908)
    (('appears', 'have'), 2.9216001154633986)
    (('authorized', 'by'), 2.921418495894848)
    (('executed', 'by'), 2.921418495894848)
    (('whose', 'by'), 2.921418495894848)
    (('by', 'section'), 2.921418495894846)
    (('obtained', 'by'), 2.921418495894846)
    (('provided', 'by'), 2.921418495894846)
    (('a', 'bar'), 2.921091638277419)
    (('a', 'chancery'), 2.921091638277419)
    (('there', 'other'), 2.9203292583641396)
    (('there', 'were'), 2.9203292583641396)
    (('it', 'otherwise'), 2.9192408425894776)
    (('have', 'assigned'), 2.917221674952824)
    (('will', 'be'), 2.916017946646555)
    (('facto', 'is'), 2.915329330819082)
    (('is', 'alleged'), 2.915329330819082)
    (('.', 'b'), 2.915003849703007)
    (('in', 'case'), 2.9137596771768983)
    (('“', 'all'), 2.9121860850555272)
    (('madison', 'court'), 2.9111037931511365)
    (('been', 'held'), 2.9088336195263107)
    (('injustice', 'to'), 2.9075379795282394)
    (('instructions', 'to'), 2.9075379795282394)
    (('to', 'affix'), 2.9075379795282394)
    (('to', 'authorize'), 2.9075379795282394)
    (('to', 'dismiss'), 2.9075379795282394)
    (('to', 'establish'), 2.9075379795282394)
    (('to', 'secure'), 2.9075379795282394)
    (('to', 'try'), 2.9075379795282394)
    (('judgment', '('), 2.8992888644973362)
    (('in', 'collateral'), 2.8988686870704132)
    (('irregularity', 'in'), 2.8988686870704132)
    (('is', 'well'), 2.8982558174601394)
    (('plaintiff', 'his'), 2.8972145362594794)
    (('on', 'execution'), 2.8946986583266856)
    (('is', 'good'), 2.892961517790628)
    (('there', 'are'), 2.8917601061673714)
    (('because', 'was'), 2.8907996966960496)
    (('injunction', 'was'), 2.8907996966960496)
    (('john', 'was'), 2.8907996966960496)
    (('was', 'trespass'), 2.8907996966960496)
    (('they', 'would'), 2.8901597787255273)
    (('an', 'post'), 2.885749026653995)
    (('act', '“'), 2.8853103790641317)
    (('demurred', 'and'), 2.884231199011742)
    (('for', 'sale'), 2.8839531037440427)
    (('(', '3'), 2.8827149665643628)
    (('to', 'decide'), 2.8810657681670477)
    (('to', 'permit'), 2.8810657681670477)
    (('debet', 'a'), 2.8792714625827927)
    (('nil', 'a'), 2.8792714625827927)
    (('all', 'may'), 2.877420666894851)
    (('opinion', '('), 2.877314978821774)
    (('chief', '.'), 2.8756296113688613)
    (('been', 'entered'), 2.8748862876029744)
    (('award', 'or'), 2.867589657535154)
    (('he', 'has'), 2.8632014194684388)
    (('thing', 'that'), 2.8626991611880825)
    (('contended', 'by'), 2.862524806841277)
    (('intended', 'by'), 2.862524806841277)
    (('required', 'by'), 2.862524806841277)
    (('shown', 'by'), 2.862524806841277)
    (('repeal', 'of'), 2.8616533510365336)
    (('would', 'been'), 2.8608282466154797)
    (('.', 'scates'), 2.8598622955105437)
    (('is', 'follows'), 2.856435641765515)
    (('justice', 'this'), 2.8549260291077623)
    (('proceed', 'to'), 2.848644290474672)
    (('to', 'countersign'), 2.848644290474672)
    (('they', 'their'), 2.8470910568336407)
    (('was', 'found'), 2.845711807167511)
    (('on', 'motion'), 2.843473334971513)
    (('person', 'act'), 2.8402766254496274)
    (('nor', 'he'), 2.839452816695829)
    (('a', 'criminal'), 2.838629478085446)
    (('a', 'judicial'), 2.838629478085446)
    (('of', 'peace'), 2.8366245565450114)
    (('has', 'do'), 2.8348960502482914)
    (('a', 'new'), 2.8347250236295274)
    (('in', 'rendering'), 2.828479359179017)
    (('to', 'pay'), 2.8226490819417265)
    (('2', 's'), 2.822362347509621)
    (('that', 'indorsement'), 2.8220571766907376)
    (('wit', 'that'), 2.8220571766907376)
    (('having', 'this'), 2.8184001530826492)
    (('duties', 'of'), 2.8172592316780793)
    (('james', 'and'), 2.8138418711203457)
    (('not', 'necessary'), 2.8128645609955907)
    (('.', 'vide'), 2.8114992739491456)
    (('this', 'action'), 2.808416064510027)
    (('applicable', 'to'), 2.8080023059773236)
    (('to', 'inquire'), 2.8080023059773236)
    (('to', 'obtain'), 2.8080023059773236)
    (('execution', 'must'), 2.804366338605778)
    (('in', 'ordinary'), 2.8020071478178252)
    (('in', 'territory'), 2.8020071478178252)
    ((')', 'note'), 2.8017036128620525)
    (('is', 'valid'), 2.7998521133991474)
    (('an', 'award'), 2.796250875814893)
    (('want', 'of'), 2.794539155177997)
    (('which', 'could'), 2.7941005793907934)
    (('a', 'man'), 2.7935415885569057)
    (('not', 'now'), 2.7925921659693262)
    (('permitted', 'to'), 2.792060762108303)
    (('opinion', ')'), 2.7917881439969694)
    (('this', 'state'), 2.790919416660543)
    (('.', 'c'), 2.7894729676191474)
    (('tiel', '.'), 2.7894729676191474)
    (('viz', '.'), 2.7894729676191474)
    (('and', 'detainer'), 2.787369659759152)
    (('plaintiff', 'declaration'), 2.786790546565827)
    (('on', 'second'), 2.784311262503369)
    (('seem', 'to'), 2.7835492622527855)
    (('note', 'was'), 2.781279934572513)
    (('mandamus', 'be'), 2.7781590355217745)
    (('law', '?'), 2.776687648305103)
    (('have', 'received'), 2.7757492489999436)
    (('be', 'shown'), 2.771732766362341)
    (('said', '“'), 2.7713235492156763)
    (('circuit', 'must'), 2.770014833814999)
    (('in', 'force'), 2.7695856701254495)
    (('”', '’'), 2.769568952045816)
    ((')', '1'), 2.7690182625456785)
    (('for', 'consideration'), 2.767760086239523)
    (('is', 'too'), 2.7633262373740326)
    (('is', 'directly'), 2.763326237374031)
    (('from', 'state'), 2.763285548218562)
    (('court', 'jurisdiction'), 2.7591006997060887)
    (('in', 'chancery'), 2.7589384259259404)
    (('governor', 'said'), 2.7573166912217975)
    (('authorizes', 'to'), 2.755534886083188)
    (('refuse', 'to'), 2.755534886083188)
    (('to', 'sell'), 2.755534886083188)
    (('at', 'when'), 2.7538476719469926)
    (('rendered', 'for'), 2.7506865728805785)
    (('would', 'be'), 2.749706460032339)
    (('should', 'been'), 2.7489253279959147)
    (('john', 'this'), 2.748010825191251)
    (('?', 'there'), 2.7465621906274347)
    (('in', 'discharge'), 2.743590461592504)
    (('offered', 'in'), 2.743590461592504)
    (('1.', 'was'), 2.7387966032509983)
    (('“', 'any'), 2.738156685280476)
    (('who', 'has'), 2.735360376697379)
    (('(', 'note'), 2.7352273542418057)
    (('3', ')'), 2.7336851894334)
    (('on', 'filed'), 2.7336851894334)
    (('reverse', 'that'), 2.731454627909832)
    (('with', 'opinion'), 2.730722481210085)
    (('.', 'let'), 2.73057927856558)
    (('under', 'act'), 2.7271625453981727)
    (('commencement', 'of'), 2.724149827286599)
    (('expire', 'of'), 2.724149827286599)
    (('failure', 'of'), 2.724149827286599)
    (('foundation', 'of'), 2.724149827286599)
    (('interposition', 'of'), 2.724149827286599)
    (('quantity', 'of'), 2.724149827286599)
    (('receipt', 'of'), 2.724149827286599)
    (('rendition', 'of'), 2.724149827286599)
    (('of', 'kin'), 2.724149827286597)
    (('performance', 'of'), 2.724149827286597)
    (('to', 'support'), 2.7231134083908124)
    (('shows', 'that'), 2.7225215031398218)
    (('an', 'interest'), 2.7222502943711167)
    (('matter', 'which'), 2.7222502943711167)
    (('lockwood', '.'), 2.7199320343660673)
    (('is', 'sufficient'), 2.7189321180155783)
    (('is', 'true'), 2.7189321180155783)
    (('overruled', 'court'), 2.718458715208742)
    (('for', 'three'), 2.7171340131695523)
    (('not', 'liable'), 2.716356580142243)
    (('be', 'received'), 2.7151492379959734)
    (('him', 'from'), 2.7135984241182065)
    (('to', 'execute'), 2.7111407667247356)
    (('property', 'which'), 2.7096502575914823)
    (('case', 'decided'), 2.708754451886893)
    (('says', 'that'), 2.706579959270803)
    ((')', 'decree'), 2.703937846039347)
    (('i', 'have'), 2.7035994632441103)
    (('answer', 'this'), 2.7029229356627127)
    (('suit', 'law'), 2.701919879903132)
    (('because', 'not'), 2.7010086158088065)
    (('we', 'can'), 2.700479671699899)
    (('appears', 'that'), 2.7004277322892083)
    (('by', 'our'), 2.6990260745583985)
    (('issued', 'by'), 2.6990260745583985)
    (('intended', 'to'), 2.6966411970296207)
    (('whether', 'has'), 2.6911605727868437)
    (('.', '4.'), 2.6899372940682316)
    (('this', 'clearly'), 2.6891171361376838)
    (('a', 'conviction'), 2.6866263846403964)
    (('a', 'vested'), 2.6866263846403964)
    (('granting', 'a'), 2.6866263846403964)
    (('a', 'fee'), 2.6866263846403946)
    (('applied', 'to'), 2.685145558191792)
    (('known', 'to'), 2.685145558191792)
    (('to', 'crimes'), 2.685145558191792)
    (('to', 'prevent'), 2.685145558191792)
    (('such', 'rule'), 2.6825577637026363)
    (('court', 'county'), 2.6809840097900803)
    (('.', '*'), 2.6794639628387245)
    (('reynolds', '.'), 2.6781118586714427)
    (('his', 'declaration'), 2.677856175012664)
    (('to', 'show'), 2.6775323740819132)
    (('again', 'in'), 2.6764762657339674)
    (('put', 'in'), 2.6764762657339674)
    (('forth', 'in'), 2.6764762657339656)
    (('rep.', '('), 2.6699233155101894)
    (('defendant', 'below'), 2.6694439351964636)
    (('have', 'made'), 2.6658063523858644)
    (('case', 'however'), 2.6653321715957645)
    (('is', 'authorized'), 2.6637905638231167)
    (('is', 'void'), 2.6637905638231167)
    (('and', 'future'), 2.6618387776752943)
    (('countersign', 'and'), 2.6618387776752943)
    (('act', 'act'), 2.661123783835345)
    (('proved', 'by'), 2.6583840900610536)
    (('which', 'he'), 2.6583033775912632)
    (('been', 'given'), 2.657294852530347)
    (('consider', 'it'), 2.6562064367556832)
    (('it', 'became'), 2.6562064367556832)
    (('look', 'to'), 2.655999212532274)
    (('entered', 'on'), 2.650828257029115)
    (('be', 'liable'), 2.6500542097740887)
    (('be', 'tried'), 2.6500542097740887)
    (('governor', 'had'), 2.647528545683686)
    (('no', 'right'), 2.647498504870331)
    (('been', 'cause'), 2.6462396640213512)
    (('failed', 'to'), 2.6400576686632533)
    (('v.', '1'), 2.639636083351178)
    (('ground', 'was'), 2.639260929700084)
    (('at', 'trial'), 2.637861477615486)
    (('.', 'ibid.'), 2.637469874174096)
    (('will', 'not'), 2.63389441995027)
    (('been', 'brought'), 2.633448110575978)
    (('has', 'no'), 2.632790643056831)
    ((')', 'now'), 2.6293485296186656)
    (('if', 'so'), 2.6286960485019684)
    (('manner', 'as'), 2.6244029709729713)
    (('by', 'constitution'), 2.6218582140359388)
    (('s', 'plea'), 2.6214496535836247)
    (('for', 'further'), 2.62091869791025)
    (('suit', 'would'), 2.6183551632640025)
    (('have', 'power'), 2.6182079720134652)
    (('judgment', 'circuit'), 2.6162167054269965)
    (('could', 'be'), 2.615887606622895)
    ((',', '88.'), 2.6137258375929484)
    ((',', '27.'), 2.6137258375929466)
    ((',', '390.'), 2.6137258375929466)
    ((',', '419.'), 2.6137258375929466)
    ((',', '480.'), 2.6137258375929466)
    ((',', '51'), 2.6137258375929466)
    ((',', '59.'), 2.6137258375929466)
    ((',', '77.'), 2.6137258375929466)
    (('boswell', ','), 2.6137258375929466)
    (('eames', ','), 2.6137258375929466)
    (('harwood', ','), 2.6137258375929466)
    (('september', ','), 2.6137258375929466)
    (('this', 'decision'), 2.6132857231780378)
    (('are', 'therefore'), 2.6132542915711348)
    (('part', 'of'), 2.6116750980281864)
    (('this', 'objection'), 2.6089467874537)
    (('of', 'october'), 2.6086726098666606)
    (('of', 'st.'), 2.6086726098666606)
    (('a', 'penalty'), 2.6086238726391233)
    (('2.', 'that'), 2.59966475535429)
    (('a', 'subsequent'), 2.5991635433900573)
    (('an', 'brought'), 2.5988678788658337)
    (('of', 'gatewood'), 2.5986189452027393)
    (('of', 'importance'), 2.5986189452027393)
    (('could', 'been'), 2.594559097182385)
    (('not', 'within'), 2.5943660557636328)
    (('power', 'at'), 2.5936616737049505)
    (('cases', 'may'), 2.592359954731057)
    (('further', 'that'), 2.5903268907749535)
    (('not', 'decide'), 2.5895003005918173)
    (('knowledge', 'of'), 2.5866463035366642)
    (('demurred', 'to'), 2.5856098846408777)
    (('entitled', 'to'), 2.5856098846408777)
    (('to', 'receive'), 2.5856098846408777)
    (('feel', 'to'), 2.585609884640876)
    (('refusal', 'to'), 2.585609884640876)
    (('their', 'they'), 2.584056650999848)
    (('and', 'seal'), 2.583836265674023)
    (('was', 'commenced'), 2.5826774013337186)
    (('of', '1819'), 2.5821308224141717)
    (('be', 'granted'), 2.5790876884199427)
    (('post', '.'), 2.5785761851205287)
    (('for', 'done'), 2.575115008297125)
    (('avail', 'of'), 2.5721467338415493)
    (('correctness', 'of'), 2.5721467338415493)
    (('end', 'of'), 2.5721467338415493)
    (('necessary', 'to'), 2.5719349477438023)
    (('a', 'variance'), 2.571149167220458)
    (('after', 'they'), 2.569981465788125)
    (('.', 'allegation'), 2.5670805462826998)
    (('.', 'hence'), 2.5670805462826998)
    (('.', 'sims'), 2.5670805462826998)
    (('“', 'we'), 2.565320088418261)
    (('a', 'peace'), 2.5646358602617862)
    ((')', 'rule'), 2.563760187991088)
    (('to', 'prove'), 2.5575955084712803)
    (('any', 'or'), 2.556549878790527)
    (('(', '('), 2.555868987544873)
    (('31st', 'of'), 2.554224825844287)
    (('tract', 'of'), 2.554224825844287)
    (('such', 'as'), 2.5540136430815714)
    (('only', 'one'), 2.55283159054442)
    ((';', 'if'), 2.552747195268669)
    (('for', 'costs'), 2.552747195268669)
    (('if', 'he'), 2.552747195268669)
    (('give', 'an'), 2.5523252929288045)
    (('a', 'detainer'), 2.54912286089046)
    (('called', 'to'), 2.549084008615763)
    (('cent', 'and'), 2.5463615602553578)
    (('was', 'given'), 2.5461515253086056)
    (('defendant', 'plea'), 2.543913053112604)
    (('opinion', 'below'), 2.5433140516338337)
    (('have', 'their'), 2.5407292193357467)
    (('this', 'cause'), 2.5398646537086655)
    (('opinion', 'that'), 2.539123213419238)
    (('acted', 'in'), 2.5389727419840327)
    (('settled', 'that'), 2.5376037872220376)
    (('to', 'give'), 2.536700284159931)
    (('was', 'entered'), 2.53492426988535)
    (('unnecessary', 'to'), 2.533142464746742)
    (('where', 'made'), 2.5329667175281454)
    (('affirmed', '.'), 2.5316752101515014)
    (('18th', 'of'), 2.531504749344201)
    (('of', 'july'), 2.531504749344201)
    (('reversal', 'of'), 2.531504749344201)
    (('judgment', '1'), 2.527320087110379)
    (('has', 'made'), 2.525765229825044)
    (('and', 'imprisonment'), 2.5243352539253596)
    (('be', 'found'), 2.5225041600535754)
    (('this', 'point'), 2.520719604441963)
    (('2', '’'), 2.52003881056177)
    (('but', 'do'), 2.518966695982373)
    (('to', 'render'), 2.5184956887823375)
    (('not', 'received'), 2.5184172025303333)
    (('court', 'reverse'), 2.5168248540390916)
    (('a', 'witness'), 2.5167013831980842)
    (('make', 'a'), 2.5167013831980825)
    (('said', 'state'), 2.516432316690496)
    (('that', 'should'), 2.5119062495797273)
    (('see', 'case'), 2.5106095767971226)
    ((',', 'sec'), 2.5068106336764373)
    (('in', 'criminal'), 2.506551264291655)
    (('from', 'declaration'), 2.506312895171968)
    (('it', 'admitted'), 2.504203343310632)
    (('might', 'be'), 2.503051797387405)
    (('article', 'of'), 2.5017574059501495)
    (('also', 'that'), 2.500129081803376)
    (('doubt', 'that'), 2.500129081803376)
    (('that', 'erred'), 2.500129081803376)
    (('held', 'that'), 2.500129081803374)
    ((')', ')'), 2.4971925710521)
    (('authorities', 'this'), 2.4964720581952875)
    (('smith', 'this'), 2.4964720581952875)
    (('court', 'gave'), 2.4960662938722944)
    (('court', 'held'), 2.4960662938722944)
    (('gave', '.'), 2.4930799648389232)
    (('to', 'whom'), 2.4925004802493973)
    (('essential', 'to'), 2.4925004802493955)
    (('to', 'presumed'), 2.4925004802493955)
    (('if', 'should'), 2.4923745772891888)
    (('we', 'must'), 2.4882438653996193)
    (('january', ','), 2.4881949555090888)
    (('of', 'exceptions'), 2.4871106299857466)
    (('be', 'affirmed'), 2.4866720541985465)
    (('directly', 'in'), 2.4838311877915693)
    (('by', 'secretary'), 2.4808459045088647)
    (('whether', 'such'), 2.479928597187616)
    (('or', 'shall'), 2.478466159109132)
    (('cause', 'below'), 2.4773231876399535)
    (('al.', ','), 2.4762223138430137)
    (('reversed', 'plaintiff'), 2.4740031055349334)
    (('in', 'cases'), 2.473384400356455)
    (('it', 'is'), 2.472610218615319)
    (('to', 'pass'), 2.470132667220941)
    (('door', '.'), 2.467544872731784)
    (('before', 'circuit'), 2.4658752246515334)
    (('sustained', 'by'), 2.461986877257548)
    ((',', '39.'), 2.461722744147899)
    (('advantage', 'of'), 2.4611154214528046)
    (('descendants', 'of'), 2.4611154214528046)
    (('of', 'quo'), 2.4611154214528046)
    (('competent', 'to'), 2.460079002557018)
    (('to', 'philips'), 2.460079002557018)
    (('to', 'say'), 2.4600790025570163)
    (('it', 'true'), 2.459809223952181)
    (('ought', 'not'), 2.4563562343980827)
    (('there', 'was'), 2.455660991360695)
    (('where', 'is'), 2.4545735312344057)
    (('in', 'equity'), 2.4540838443975197)
    (('(', 'decree'), 2.4524296935865824)
    (('a', 'part'), 2.4521611310033737)
    (('all', 'shall'), 2.452114832162179)
    (('shall', 'all'), 2.452114832162179)
    (('or', 'any'), 2.449634674874016)
    (('provisions', 'of'), 2.448515384673172)
    (('that', 'ever'), 2.4476616619092404)
    (('to', 'take'), 2.446206827975459)
    (('trial', 'at'), 2.445216399673088)
    (('of', 'register'), 2.444041908093862)
    (('as', ':'), 2.442505327864584)
    (('said', 'then'), 2.4423750266505415)
    (('is', 'called'), 2.441398142486669)
    (('whose', 'is'), 2.441398142486669)
    (('any', 'consideration'), 2.4403568612860074)
    (('between', 'and'), 2.4394463563388484)
    (('a', 'certain'), 2.438698871196811)
    (('october', ','), 2.4368480755088697)
    (('.', '('), 2.4363502511281894)
    (('copy', 'of'), 2.4346432100916147)
    (('favor', 'of'), 2.4346432100916147)
    (('had', 'him'), 2.434124907574491)
    (('it', 'held'), 2.4338140154192356)
    (('authorized', 'to'), 2.4336067911958263)
    (('to', 'replevy'), 2.4336067911958263)
    (('to', 'make'), 2.4336067911958246)
    (('v.', 'bank'), 2.431794847782582)
    (('decision', 'court'), 2.4308821251121575)
    (('was', 'filed'), 2.4251361243472385)
    (('a', 'collateral'), 2.423591978806602)
    (('a', 'discretion'), 2.423591978806602)
    (('p.', '('), 2.4232833479270663)
    (('it', 'sufficient'), 2.4232833479270646)
    (('show', 'that'), 2.422126569802101)
    (('that', 'purpose'), 2.422126569802101)
    ((',', '157.'), 2.4210807596505504)
    ((',', '201.'), 2.4210807596505504)
    ((',', '209.'), 2.4210807596505504)
    ((',', '25.'), 2.4210807596505504)
    ((',', '259.'), 2.4210807596505504)
    ((',', '263.'), 2.4210807596505504)
    (('there', 'is'), 2.4200696330461575)
    (('judgment', 'therefore'), 2.4182773278150886)
    (('was', 'clearly'), 2.4168685083636365)
    (('be', 'done'), 2.415588956137066)
    (('be', 'sold'), 2.4155889561370643)
    (('requires', 'be'), 2.4155889561370643)
    (('in', 'claimed'), 2.413441859900173)
    (('in', 'facias'), 2.413441859900173)
    (('in', 'scire'), 2.413441859900173)
    (('in', 'setting'), 2.413441859900173)
    (('is', 'bar'), 2.412828990289899)
    (('lockwood', 'is'), 2.412828990289899)
    (('before', 'court'), 2.4126502856846592)
    (('mandamus', 'not'), 2.4115019986138204)
    (('argument', 'that'), 2.4070196774118955)
    (('defendant', 'his'), 2.4064095293626675)
    (('bar', 'to'), 2.4050376389990564)
    (('subsequent', 'to'), 2.4050376389990564)
    (('plea', 'was'), 2.4047956760630633)
    (('recovery', 'of'), 2.402221732399237)
    (('validity', 'of'), 2.4022217323992354)
    (('a', 'good'), 2.4012241657781477)
    ((':', 'any'), 2.3992378739604128)
    (('and', 'took'), 2.3988043718415017)
    (('goods', 'and'), 2.3988043718415017)
    (('immaterial', 'and'), 2.3988043718415017)
    (('they', 'had'), 2.3985009978437706)
    (('court', 'decided'), 2.3965306203213785)
    (('in', 'year'), 2.39636834654123)
    (('agreed', 'to'), 2.3929648066984797)
    (('claimed', 'to'), 2.3929648066984797)
    (('it', 'answer'), 2.3887261258906953)
    (('prove', 'it'), 2.3887261258906953)
    (('think', 'not'), 2.3859669065066846)
    (('by', 'three'), 2.385365595654637)
    (('filed', 'by'), 2.385365595654637)
    (('a', 'defense'), 2.3840636146199667)
    (('sheriff', 'or'), 2.383713281928312)
    ((':', 'there'), 2.3822878050304777)
    (('duty', 'to'), 2.382076490555745)
    (('but', 'does'), 2.381463172232438)
    (('a', 'deed'), 2.379965046406344)
    (('shall', '“'), 2.379965046406342)
    (('have', 'given'), 2.3766532935901203)
    (('of', 'article'), 2.3762265238662916)
    (('of', 'march'), 2.3762265238662916)
    (('.', 'kent'), 2.3744354683403035)
    (('reversal', '.'), 2.3744354683403035)
    (('brought', 'court'), 2.3726838783670114)
    (('he', 'should'), 2.371422229803102)
    (('decided', 'this'), 2.370941176111428)
    (('not', 'proved'), 2.3708600141164755)
    (('exercise', 'of'), 2.370512872671899)
    (('the', 'relator'), 2.3695911836187626)
    (('a', 'sentence'), 2.364698289753033)
    (('payable', 'to'), 2.36321746330443)
    ((';', 'therefore'), 2.3628474940650896)
    (('kind', 'of'), 2.3615797479018887)
    (('of', 'frauds'), 2.3615797479018887)
    (('of', 'instruments'), 2.3615797479018887)
    (('following', 'v.'), 2.359645062026747)
    (('oyer', 'be'), 2.356695267083495)
    (('have', 'no'), 2.356304648198318)
    (('court', 'say'), 2.3558886358240336)
    (('in', 'name'), 2.3545481708466056)
    (('in', 'order'), 2.354548170846604)
    (('in', 'support'), 2.354548170846604)
    (('on', 'verdict'), 2.352200249865584)
    (('missouri', '.'), 2.352067655311849)
    (('he', 'would'), 2.3508748523990146)
    ((',', '122.'), 2.350691431759154)
    ((',', '386'), 2.350691431759154)
    ((',', '82'), 2.350691431759154)
    ((',', 'mcdonald'), 2.350691431759154)
    (('ashby', ','), 2.350691431759154)
    (('beloved', ','), 2.350691431759154)
    (('burrow', ','), 2.350691431759154)
    (('digest', ','), 2.350691431759154)
    (('duryee', ','), 2.350691431759154)
    (('hall', ','), 2.350691431759154)
    (('payne', ','), 2.350691431759154)
    (('sherwood', ','), 2.350691431759154)
    (('short', ','), 2.350691431759154)
    (('stevenson', ','), 2.350691431759154)
    (('november', ','), 2.3506914317591523)
    (('was', 'made'), 2.3487457874441837)
    (('am', 'that'), 2.3481259883583245)
    (('a', 'legal'), 2.3478244711886376)
    (('of', '1825'), 2.34563820403287)
    (('mode', 'of'), 2.345638204032868)
    (('of', 'missouri'), 2.345638204032868)
    (('matter', 'on'), 2.3437386711173875)
    (('on', 'first'), 2.3437386711173875)
    (('power', 'to'), 2.3423387336285124)
    (('instrument', 'is'), 2.341862468935755)
    (('there', 'can'), 2.341793026802417)
    (('a', 'person'), 2.341490898591708)
    (('presented', 'this'), 2.3403528562780043)
    (('damages', 'and'), 2.3399106827879326)
    (('.', 'see'), 2.3397893255334132)
    (('an', 'opinion'), 2.339780657548703)
    (('.', ')'), 2.3395767111286503)
    ((':', 'will'), 2.3376187556967487)
    (('of', 'chancery'), 2.3371267041773507)
    (('court', 'affirmed'), 2.3370296085727205)
    (('objection', 'was'), 2.3366981596796528)
    (('shall', 'have'), 2.3366376147422425)
    (('court', 'united'), 2.3337948649734184)
    (('verdict', 'was'), 2.333157801974405)
    (('scam.', ';'), 2.331138634337753)
    (('before', 'can'), 2.330945644565425)
    (('counsel', 'have'), 2.330511511945401)
    (('is', 'competent'), 2.330366830097926)
    (('for', 'money'), 2.32718749485354)
    (('proper', 'for'), 2.32718749485354)
    (('this', 'question'), 2.326547056752972)
    (('it', 'does'), 2.3248796438660726)
    (('before', 'trial'), 2.324519375405991)
    (('in', 'writing'), 2.3228393111192673)
    (('et', '1'), 2.3226165923575)
    (('apply', 'to'), 2.3225754788070834)
    (('refusing', 'to'), 2.3225754788070834)
    (('to', 'determine'), 2.3225754788070834)
    (('to', 'george'), 2.3225754788070834)
    (('that', 'administrator'), 2.3195568361615546)
    (('that', 'hobson'), 2.3195568361615546)
    (('court', 'ought'), 2.318528108320109)
    (('are', 'parties'), 2.315899812553466)
    (('being', 'no'), 2.315899812553466)
    (('wilson', '.'), 2.315541779286736)
    (('.', '†'), 2.3155417792867343)
    (('bad', '.'), 2.3155417792867343)
    (('july', ','), 2.3141655557340393)
    (('brought', 'by'), 2.312609253219323)
    (('not', 'give'), 2.311966325062908)
    (('not', 'himself'), 2.311966325062908)
    (('not', 'entitled'), 2.3119663250629063)
    (('be', 'entered'), 2.31125229632233)
    (('total', 'of'), 2.309112328007755)
    (('assignment', 'of'), 2.3091123280077532)
    (('custody', 'of'), 2.3091123280077532)
    (('of', '1823'), 2.3091123280077532)
    (('of', 'ejectment'), 2.3091123280077532)
    (('of', 'improvement'), 2.3091123280077532)
    (('a', 'special'), 2.3081147613866655)
    (('a', 'valid'), 2.3081147613866655)
    (('provided', 'that'), 2.307484003860978)
    (('appear', 'on'), 2.306263965698724)
    (('be', 'costs'), 2.3001117387171277)
    (('appointment', 'was'), 2.298224011865017)
    (('in', 'missouri'), 2.2979646424802382)
    (('questions', 'in'), 2.2979646424802382)
    (('recovered', 'in'), 2.2979646424802382)
    (('can', 'no'), 2.297752465843205)
    (('with', 'statute'), 2.297449410482173)
    (('objected', 'to'), 2.29610326744589)
    (('permit', 'to'), 2.29610326744589)
    (('.', '1'), 2.2935154729567344)
    (('court', 'motion'), 2.292193960506646)
    ((',', '24.'), 2.2917977427055867)
    (('kennedy', ','), 2.2917977427055867)
    (('r', ','), 2.2917977427055867)
    (('plaintiff', 'has'), 2.2917537252217635)
    (('any', 'act'), 2.2915776687152434)
    (('but', 'they'), 2.2906977083092563)
    (('this', 'writ'), 2.2878854363838705)
    (('but', 'if'), 2.2874026287476745)
    (('upon', 'which'), 2.2867932538533395)
    (('decision', 'was'), 2.286589311381615)
    (('practice', 'in'), 2.2841588429552075)
    (('manifest', 'and'), 2.283327154421565)
    (('(', 'a'), 2.2792832771515528)
    (('clearly', 'that'), 2.2777366604669282)
    (('use', 'of'), 2.2766908503153775)
    (('the', 'civilians'), 2.2764817792272805)
    (('it', 'will'), 2.275934355637517)
    (('can', 'only'), 2.275297615015509)
    (('.', '8'), 2.2748997947893876)
    (('of', 'land'), 2.2743469098470754)
    (('case', 'whether'), 2.272237225197598)
    (('if', 'parties'), 2.271334260120021)
    (('statute', 'has'), 2.2706921096939343)
    (('his', 'against'), 2.267889665281242)
    (('march', ','), 2.2658025341726393)
    (('a', 'party'), 2.265783263680344)
    (('decided', 'that'), 2.2656638281663533)
    (('disposed', 'of'), 2.2647182086493025)
    (('hands', 'of'), 2.2647182086493025)
    (('of', 'assumpsit'), 2.2647182086493025)
    (('terms', 'of'), 2.2647182086493025)
    (('territory', 'of'), 2.2647182086493025)
    (('cause', 'no'), 2.2643294643057565)
    (('by', 'jury'), 2.264306209417855)
    (('jurisdiction', 'case'), 2.263969609213996)
    (('be', 'given'), 2.2635858626920147)
    (('used', 'in'), 2.2614387664551234)
    (('court', 'sustained'), 2.259027096571444)
    (('it', 'considered'), 2.2562758298670484)
    (('it', 'settled'), 2.2562758298670484)
    (('sustained', '.'), 2.2560407675380745)
    (('therefore', '('), 2.253358346484754)
    (('below', 'plaintiff'), 2.2533583464847524)
    (('no', 'suit'), 2.2533583464847524)
    (('as', 'general'), 2.2528441083610105)
    (('is', 'considered'), 2.252364318096653)
    (('is', 'mere'), 2.252364318096653)
    (('february', ','), 2.251155758208238)
    (('v.', 'al'), 2.249591516561834)
    (('erroneous', '.'), 2.245152451395338)
    (('merits', '.'), 2.245152451395338)
    (('no', 'execution'), 2.244641129383645)
    (('a', 'verdict'), 2.243019733164779)
    (('people', '('), 2.2427111022852433)
    (('it', 'provided'), 2.2411689374768393)
    (('whom', 'it'), 2.2411689374768393)
    (('inquiry', 'is'), 2.2397642813170187)
    (('exceptions', 'was'), 2.238723000116357)
    (('report', 'of'), 2.238723000116357)
    (('to', 'grant'), 2.2376865812205704)
    (('proof', 'that'), 2.2370946759695816)
    (('that', 'objections'), 2.2370946759695816)
    (('that', 'president'), 2.2370946759695816)
    (('reversed', '.'), 2.2359535145488714)
    (('to', 'exercise'), 2.231972930026176)
    (('.', 'r.'), 2.2293851355370204)
    (('court', 'opinion'), 2.2290738744698153)
    (('to', 'aside'), 2.2271559137283994)
    (('a', ')'), 2.2262377543322565)
    (('*', 'an'), 2.2258244682516164)
    (('made', 'trial'), 2.2243354452542405)
    (('decided', 'case'), 2.223327624716653)
    (('paper', 'be'), 2.222943878194668)
    (('register', 'of'), 2.221649486757414)
    (('with', '('), 2.221649486757414)
    (('action', 'before'), 2.2214258824418867)
    (('such', 'made'), 2.21767171497323)
    (('if', 'they'), 2.2175630036790412)
    (('of', 'jurors'), 2.217189838566714)
    (('in', 'indictment'), 2.217044647096671)
    (('states', 'that'), 2.216336115802786)
    (('of', 'united'), 2.2139550949674174)
    (('nor', 'it'), 2.211421594082788)
    (('all', 'which'), 2.21084440062004)
    (('that', 'during'), 2.21062246460839)
    (('judgment', 'writ'), 2.2096907060036717)
    (('bills', 'of'), 2.209576654456839)
    (('of', 'valuation'), 2.209576654456839)
    (('rules', 'of'), 2.209576654456839)
    (('not', 'jurisdiction'), 2.2050511211463952)
    (('.', 'comp.'), 2.2045104668979913)
    (('a', 'agreement'), 2.2011995574701544)
    (('of', 'senate'), 2.2005878712295868)
    (('had', 'with'), 2.2000695687124647)
    (('by', 'supreme'), 2.1989524714237554)
    ((',', '185.'), 2.1986883383141027)
    ((',', '20.'), 2.1986883383141027)
    ((',', '255.'), 2.1986883383141027)
    ((',', '47.'), 2.1986883383141027)
    ((',', '48'), 2.1986883383141027)
    ((',', '63.'), 2.1986883383141027)
    ((',', 'speaking'), 2.1986883383141027)
    ((',', 'testified'), 2.1986883383141027)
    (('1827', ','), 2.1986883383141027)
    (('bibb', ','), 2.1986883383141027)
    (('ev.', ','), 2.1986883383141027)
    (('merritt', ','), 2.1986883383141027)
    (('stephenson', ','), 2.1986883383141027)
    (('wells', ','), 2.1986883383141027)
    (('such', 'shall'), 2.198358239916395)
    (('laws', 'of'), 2.1962182716018184)
    (('lockwood', 'was'), 2.1956542782244703)
    (('was', 'liable'), 2.1956542782244703)
    (('section', 'of'), 2.1936351105878202)
    (('value', 'of'), 2.1936351105878185)
    (('that', 'appellant'), 2.192006786441043)
    (('judgment', 'ought'), 2.188271722368766)
    (('into', 'court'), 2.1879439985099634)
    (('record', 'can'), 2.1870704320037735)
    (('fact', 'that'), 2.1869711965437464)
    (('below', '('), 2.186244150626214)
    (('(', '4'), 2.184496488339949)
    (('a', 'false'), 2.1841260441112116)
    (('intention', 'of'), 2.1835814459238954)
    (('it', 'clearly'), 2.18227524842327)
    (('it', 'shown'), 2.18227524842327)
    (('1', '’'), 2.181622592885839)
    (('was', 'assigned'), 2.1798293110627895)
    (('then', 'said'), 2.179340620816749)
    (('is', 'very'), 2.1783637366528765)
    (('is', 'provided'), 2.1783637366528747)
    (('that', 'coles'), 2.178200986916014)
    (('.', 'wheat.'), 2.1780382555367996)
    (('inconsistent', '.'), 2.1780382555367996)
    (('defendant', 'had'), 2.177785154314707)
    (('action', 'on'), 2.177291840909014)
    (('plaintiff', 'action'), 2.1767370648818414)
    (('seal', 'of'), 2.1766620319841046)
    (('existence', 'and'), 2.176411950505054)
    (('a', 'contract'), 2.175664465363017)
    (('not', 'until'), 2.1744628013129716)
    (('which', 'were'), 2.174318524594929)
    (('paid', 'in'), 2.1739759252047826)
    (('if', 'any'), 2.172266918026194)
    (('a', 'plea'), 2.1720532118106384)
    (('a', 'want'), 2.1720532118106366)
    (('to', 'do'), 2.1705723853620356)
    (('opportunity', 'to'), 2.170572385362034)
    (('supposed', 'to'), 2.170572385362034)
    (('to', 'moore'), 2.170572385362034)
    (('importance', 'to'), 2.170572385362032)
    (('mandamus', 'to'), 2.170572385362032)
    (('to', 'rankin'), 2.170572385362032)
    (('money', 'be'), 2.167661442693481)
    (('during', 'the'), 2.1675474076741175)
    (('verdict', 'for'), 2.165724072159424)
    (('a', 'trial'), 2.1643996392672022)
    (('united', 'v.'), 2.162608214967193)
    (('interest', 'in'), 2.1619030929042076)
    (('shall', 'made'), 2.1615996899062786)
    (('court', 'however'), 2.1604632620878554)
    (('shall', 'at'), 2.160088222988678)
    (('require', 'that'), 2.1590921639683085)
    (('page', 'v.'), 2.1582297744566183)
    (('point', 'court'), 2.157743760734263)
    (('suit', 'been'), 2.157221249395702)
    (('party', 'or'), 2.1555152389106613)
    (('cranch', ','), 2.154294218955652)
    (('demurrer', 'was'), 2.153834102529844)
    (('facts', 'in'), 2.152914309676955)
    (('be', 'up'), 2.1525545503032717)
    (('we', 'no'), 2.1509110775139337)
    (('cause', 'with'), 2.1501584443864434)
    (('against', 'at'), 2.1498924558827888)
    (('be', 'taken'), 2.148802415442166)
    (('2', ';'), 2.148722688712244)
    (('found', 'in'), 2.1459615490351887)
    (('the', 'senate'), 2.14523724594903)
    (('but', 'when'), 2.143099793984206)
    (('set', 'in'), 2.1430440656528944)
    (('from', 'circuit'), 2.140708755697627)
    (('on', 'demurrer'), 2.1404550727286438)
    (('this', 'done'), 2.140328247970011)
    (('account', 'of'), 2.139187326565443)
    (('of', 'november'), 2.139187326565443)
    (('of', 'lord'), 2.139187326565441)
    (('possession', 'of'), 2.139187326565441)
    (('is', 'principle'), 2.1388353724662394)
    (('by', 'sheriff'), 2.137147186950285)
    (('a', 'jury'), 2.1364293020799145)
    (('in', 'argument'), 2.1359078843712638)
    (('it', 'further'), 2.134969533644913)
    (('court', 'equity'), 2.133496214487584)
    (('p.', 'v.'), 2.1322345659236746)
    (('v.', 'p.'), 2.1322345659236746)
    (('there', 'must'), 2.1295572205021394)
    (('issue', 'on'), 2.1267801358315666)
    (('cause', 'for'), 2.1266545074808327)
    (('dollars', 'and'), 2.125785877435085)
    (('there', 'been'), 2.1257716107706504)
    (('et', '.'), 2.1252637158882557)
    (('the', 'appellee'), 2.124478685782231)
    (('where', 'an'), 2.1239909710365037)
    (('appear', 'that'), 2.121617458549647)
    (('can', 'then'), 2.120178548348605)
    (('john', '.'), 2.1196215693114784)
    (('is', 'contended'), 2.1194700475993073)
    (('is', 'required'), 2.1194700475993073)
    (('oyer', 'is'), 2.1194700475993073)
    (('sentence', 'is'), 2.1194700475993073)
    (('state', 'when'), 2.1179292182493015)
    (('when', 'are'), 2.114530641787436)
    (('(', 'laws'), 2.112943227314915)
    (('legislature', 'which'), 2.112196812687131)
    (('which', 'had'), 2.112196812687131)
    (('because', 'it'), 2.111885920531874)
    (('required', 'to'), 2.111678696308463)
    (('if', 'there'), 2.1102289595677206)
    (('in', 'declaration'), 2.1066206574030204)
    (('counsel', 'this'), 2.1056821051631207)
    (('which', 'do'), 2.103929196703529)
    (('an', 'office'), 2.1033404617266207)
    (('at', 'any'), 2.10324236258397)
    (('was', 'action'), 2.103208029459875)
    (('name', 'of'), 2.102661450540328)
    (('it', 'principle'), 2.1021048997392864)
    (('a', 'obtained'), 2.1016638839192403)
    (('a', 'commission'), 2.1016638839192385)
    (('a', 'correct'), 2.1016638839192385)
    (('a', 'personal'), 2.1016638839192385)
    (('offered', 'to'), 2.1001830574706357)
    (('campbell', ','), 2.0991526647631886)
    (('clark', ','), 2.0991526647631886)
    (('executors', ','), 2.0991526647631886)
    (('sprinkle', ','), 2.0991526647631886)
    (('opinion', 'judgment'), 2.098817488518476)
    ((')', '2'), 2.098096615642275)
    (('persons', 'not'), 2.0978415197100606)
    (('rule', 'is'), 2.0959015764609035)
    (('error', 'has'), 2.093814347609854)
    (('be', 'contended'), 2.0936608612497025)
    (('be', 'reason'), 2.0936608612497025)
    (('intended', 'be'), 2.0936608612497025)
    (('which', 'shall'), 2.0936608612497007)
    (('go', 'to'), 2.0925698733607607)
    (('to', 'require'), 2.0925698733607607)
    (('it', 'would'), 2.0920774394516926)
    (('construction', 'of'), 2.091881611787084)
    (('averment', 'in'), 2.0915137650128113)
    (('judge', 'in'), 2.0915137650128095)
    (('pleadings', 'in'), 2.0915137650128095)
    (('justice', '.'), 2.089033249478053)
    (('are', 'so'), 2.0882991002142557)
    (('guilty', 'of'), 2.0867199066713074)
    (('was', 'debt'), 2.0867199066713056)
    (('court', 'further'), 2.0861904997092307)
    (('declared', 'that'), 2.085091582524532)
    (('follows', 'that'), 2.085091582524532)
    (('requires', 'that'), 2.085091582524532)
    (('1.', 'that'), 2.08509158252453)
    (('stated', 'that'), 2.08509158252453)
    (('may', 'have'), 2.0838715443622746)
    (('i', 'not'), 2.08369733738979)
    (('for', 'trial'), 2.083261911967451)
    (('nature', 'of'), 2.0802936375118755)
    (('now', 'by'), 2.0801162419139043)
    (('right', 'to'), 2.078313877510041)
    (('it', 'now'), 2.0779385886085358)
    (('and', 'again'), 2.0768762769541382)
    (('and', 'discharge'), 2.0768762769541382)
    (('and', 'duties'), 2.0768762769541382)
    (('and', 'important'), 2.0768762769541382)
    (('and', 'shows'), 2.0768762769541382)
    (('commission', 'and'), 2.0768762769541382)
    (('a', 'right'), 2.075900787854156)
    (('case', 'v.'), 2.074683543361516)
    (('9', ','), 2.073157456230245)
    (('been', 'time'), 2.072332351809191)
    (('1', 'statute'), 2.0720731307069045)
    (('came', 'of'), 2.0720731307069045)
    (('made', 'by'), 2.0679671592477895)
    (('there', 'has'), 2.0663336111877477)
    (('could', 'no'), 2.0650917090529326)
    (('judgment', 'with'), 2.0646403732003904)
    (('think', 'it'), 2.0636307519246504)
    (('is', 'answer'), 2.06288651923294)
    (('is', 'presented'), 2.06288651923294)
    (('he', 'may'), 2.0618452380322783)
    ((',', '24'), 2.0611848145641716)
    (('notice', 'of'), 2.061184814564168)
    (('in', 'evidence'), 2.0608169677898935)
    (('ill.', 'if'), 2.0605418354673652)
    (('new', 'on'), 2.060284724044662)
    (('as', 'governor'), 2.0601990304186124)
    (('by', 'law'), 2.0601247667264424)
    (('below', 'opinion'), 2.0578872244635935)
    (('it', 'also'), 2.0567443663394123)
    (('was', 'also'), 2.05429842897893)
    (('when', 'is'), 2.054035601650675)
    (('be', 'executed'), 2.053018876752356)
    (('.', 'et'), 2.0525073734529435)
    (('.', 'errors'), 2.0525073734529418)
    (('properly', '.'), 2.0525073734529418)
    (('prosecuted', '.'), 2.0525073734529418)
    (('thereon', '.'), 2.0525073734529418)
    (('waggoner', '.'), 2.0525073734529418)
    (('8', '.'), 2.05250737345294)
    (('a', 'remedy'), 2.049196464025105)
    (('is', 'clearly'), 2.049080719707911)
    (('yet', 'not'), 2.048931919229112)
    (('land', 'on'), 2.0473456683371616)
    (('consent', 'of'), 2.0460779221739607)
    (('consequence', 'of'), 2.0460779221739607)
    (('examination', 'of'), 2.0460779221739607)
    (('of', 'deceased'), 2.0460779221739607)
    (('of', 'paymaster-general'), 2.0460779221739607)
    (('session', 'of'), 2.0460779221739607)
    (('land', 'or'), 2.045127843087343)
    (('nature', 'a'), 2.042770194865671)
    (('case', 'people'), 2.0427553790748316)
    (('action', 'may'), 2.041807484795733)
    (('if', 'has'), 2.040737295320893)
    (('which', 'brought'), 2.039440470251817)
    (('that', 'ought'), 2.038797930250599)
    (('against', 'defendant'), 2.038022123382648)
    (('appears', 'on'), 2.037077332883335)
    (('be', 'commenced'), 2.037077332883335)
    (('been', 'taken'), 2.0343645016101704)
    (('came', 'to'), 2.0330688616120973)
    (('credit', 'to'), 2.0330688616120973)
    (('giving', 'to'), 2.0330688616120973)
    (('in', 'nature'), 2.032620075959244)
    (('discharge', 'of'), 2.03227212264893)
    (('of', 'madison'), 2.03227212264893)
    (('2d', ','), 2.0287633368717923)
    (('april', ','), 2.0287633368717923)
    (('that', 'intended'), 2.026197893470963)
    (('the', 'institution'), 2.024943012231315)
    (('the', 'overseers'), 2.024943012231315)
    (('1', 's'), 2.0245145111963936)
    (('in', 'form'), 2.024399569154273)
    (('of', 'replevin'), 2.0237101091455063)
    (('of', 'york'), 2.0237101091455063)
    (('provision', 'of'), 2.0237101091455063)
    (('a', 'mere'), 2.0236613719179655)
    (('held', 'be'), 2.0232715333583062)
    (('without', 'or'), 2.0227600300588904)
    (('would', 'not'), 2.0224597078679203)
    ((',', 'page'), 2.022190682948306)
    (('this', 'opinion'), 2.0208930169813932)
    (('is', 'whether'), 2.0208224596663946)
    (('decision', 'case'), 2.018213194812038)
    (('presented', 'court'), 2.0180189970676494)
    (('nature', 'and'), 2.017982587900571)
    (('if', 'right'), 2.0179704517157298)
    (('decided', 'in'), 2.017513183569033)
    (('in', 'error'), 2.0155649117795136)
    (('defendants', 'not'), 2.014984587305774)
    (('due', 'by'), 2.0145279002863283)
    (('in', 'penalty'), 2.0135112530115364)
    (('therefore', 'be'), 2.0111987010577295)
    (('this', 'decree'), 2.0110452310250437)
    (('entry', 'and'), 2.0097620810956)
    (('be', 'necessary'), 2.009596596461229)
    (('there', 'be'), 2.009596596461229)
    (('although', 'not'), 2.009403555042475)
    (('a', 'final'), 2.008554479527758)
    (('another', 'is'), 2.0084387352105626)
    (('of', 'character'), 2.0079427932871887)
    (('trial', 'upon'), 2.0058369400222844)
    (('other', 'which'), 2.0043935231526167)
    (('term', 'court'), 2.002251681209332)
    (('unless', 'it'), 2.001703002781449)
    (('be', 'very'), 2.0005514568582203)
    (('and', 'sworn'), 1.9988737649528652)
    (('rule', 'as'), 1.9987984857544685)
    (('a', 'warrant'), 1.9985703909551358)
    (('filed', 'a'), 1.9985703909551358)
    (('in', 'absence'), 1.998404360621329)
    (('day', 'of'), 1.9983247907255937)
    (('*', 'been'), 1.9973702941279683)
    (('justice', 'had'), 1.9967195952671943)
    (('court', 'decide'), 1.9959926907376513)
    (('appointment', 'by'), 1.9954190773386244)
    (('settled', 'by'), 1.9954190773386244)
    (('a', 'bond'), 1.9947486800027274)
    (('there', 'an'), 1.9943298398079161)
    (('plea', 'is'), 1.9939391655154495)
    (('which', 'they'), 1.9911374264503472)
    (('opinion', '1'), 1.9907730286050551)
    (('am', 'of'), 1.9871842331203915)
    (('was', 'sufficient'), 1.9871842331203915)
    (('a', 'writ'), 1.9861866664993038)
    (('him', 'his'), 1.9846773775098168)
    (('plaintiff', 'had'), 1.9840919869394433)
    (('directly', 'and'), 1.9837668725626578)
    (('by', 'statute'), 1.98281904055899)
    (('brought', 'judgment'), 1.9793930865818776)
    (('left', 'to'), 1.9779273074196357)
    (('this', 'now'), 1.9770978991017056)
    (('are', 'other'), 1.975824370955845)
    (('who', 'no'), 1.975824370955845)
    (('but', 'he'), 1.9738052033191913)
    (('to', 'recover'), 1.972633007750126)
    (('on', 'record'), 1.9720575410329246)
    (('a', 'character'), 1.9704193506409844)
    (('this', 'rule'), 1.9704032465276988)
    (('people', ','), 1.9692064921913364)
    (('making', 'it'), 1.9681504430704226)
    (('be', 'decided'), 1.9681299791658429)
    (('be', 'due'), 1.9681299791658429)
    (('.', '2'), 1.967115882222295)
    ((')', 'where'), 1.9655008646564749)
    (('.', 'certainly'), 1.9650445322026027)
    (('.', 'lockwood'), 1.965044532202601)
    (('affirmed', 'a'), 1.9641603601693056)
    (('counsel', 'in'), 1.963758217814437)
    (('cases', 'v.'), 1.9623095644813624)
    (('suit', ';'), 1.9623095644813624)
    ((',', 'provides'), 1.961649141013254)
    (('a', 'motion'), 1.9608013480793893)
    (('decree', 'for'), 1.9604051641819158)
    (('presented', 'by'), 1.957944371919961)
    (('but', 'we'), 1.9570878083742578)
    (('not', 'be'), 1.9570498182216376)
    ((';', 'there'), 1.9568551343892562)
    (('having', 'an'), 1.9567155480081393)
    (('be', 'set'), 1.9561573374997678)
    (('any', 'shall'), 1.956157337499766)
    (('be', 'doubt'), 1.956157337499766)
    (('not', 'done'), 1.9558225148376316)
    (('court', 'having'), 1.9529239688457665)
    (('any', 'record'), 1.9500312347029265)
    (('“', 'an'), 1.949660790474189)
    (('?', 'it'), 1.949614491632996)
    (('law', 'he'), 1.9490127418679037)
    (('when', 'he'), 1.9490127418679037)
    (('said', 'note'), 1.9489664638188255)
    (('the', 'legislature'), 1.948321730628404)
    (('county', 'which'), 1.9478099947862475)
    (('age', 'of'), 1.9465422486230448)
    (('of', 'clair'), 1.9465422486230448)
    (('of', 'description'), 1.9465422486230448)
    (('court', 'will'), 1.9458692113118126)
    (('.', '5'), 1.9455921695364289)
    (('courts', 'this'), 1.9439310351665071)
    (('“', 'there'), 1.9435990376869885)
    (('exceptions', 'is'), 1.941324539352026)
    (('they', 'no'), 1.940200461225123)
    (('p.', '.'), 1.9400326441945275)
    (('.', '3.'), 1.9370301560330034)
    (('.', 'indenture'), 1.9370301560330034)
    (('v.', 'page'), 1.9358373531201707)
    ((',', '103.'), 1.9356539324803101)
    ((',', '17.'), 1.9356539324803101)
    ((',', '1825.'), 1.9356539324803101)
    (('1818', ','), 1.9356539324803101)
    (('comm.', ','), 1.9356539324803101)
    (('holmes', ','), 1.9356539324803101)
    (('hugsby', ','), 1.9356539324803101)
    (('june', ','), 1.9356539324803101)
    (('lessee', ','), 1.9356539324803101)
    (('miller', ','), 1.9356539324803101)
    (('offering', ','), 1.9356539324803101)
    (('pull.', ','), 1.9356539324803101)
    (('ryan', ','), 1.9356539324803101)
    (('warren', ','), 1.9356539324803101)
    (('weeks', ','), 1.9356539324803101)
    (('they', 'can'), 1.9345538980839798)
    (('be', 'with'), 1.9315553642666892)
    (('payment', 'of'), 1.9306007047540277)
    (('&', 'v.'), 1.9306007047540241)
    (('v.', '&'), 1.9306007047540241)
    (('been', 'by'), 1.9304636354978548)
    (('appointment', 'is'), 1.9304362232092913)
    (('plaintiff', 'defendant'), 1.930381400194678)
    (('be', 'pleaded'), 1.930162128966824)
    (('had', 'made'), 1.9285968743477397)
    (('that', 'did'), 1.9275503055380518)
    (('court', 'therefore'), 1.9270453370201608)
    (('who', 'his'), 1.9257836884562494)
    (('fact', 'was'), 1.9255651148567257)
    (('this', 'section'), 1.9253153569991603)
    (('that', 'he'), 1.925220245746143)
    (('and', 'am'), 1.9248731835090869)
    (('and', 'since'), 1.9248731835090869)
    (('and', 'whose'), 1.9248731835090869)
    (('both', 'and'), 1.9248731835090869)
    (('sold', 'and'), 1.9248731835090869)
    (('sum', 'and'), 1.9248731835090869)
    (('brought', 'on'), 1.9239632528318786)
    (('to', 'go'), 1.9226448719184486)
    (('a', 'promise'), 1.921091638277419)
    (('replication', 'a'), 1.921091638277419)
    (('why', 'a'), 1.921091638277419)
    (('circuit', 'on'), 1.9192408425894776)
    (('it', 'therefore'), 1.9192408425894776)
    (('yet', 'it'), 1.9192408425894758)
    (('case', 'before'), 1.9175191951925648)
    (('existence', 'of'), 1.9167949052289952)
    (('merits', 'of'), 1.9167949052289952)
    (('of', 'nil'), 1.9167949052289952)
    (('whom', 'is'), 1.9153293308190822)
    (('that', 'executed'), 1.9151665810822198)
    (('.', 'distinction'), 1.915003849703007)
    (('building', '.'), 1.915003849703007)
    (('page', '.'), 1.9150038497030053)
    (('for', 'defendant'), 1.9143849999580027)
    (('this', 'was'), 1.9137480296701277)
    (('cent', ','), 1.9132861194518576)
    (('not', 'doubt'), 1.911428395479179)
    (('parties', 'have'), 1.9093738131301397)
    (('question', 'is'), 1.9089030616596467)
    (('allowed', 'to'), 1.9075379795282394)
    (('appoint', 'to'), 1.9075379795282394)
    (('to', 'enter'), 1.9075379795282394)
    (('great', 'and'), 1.9069512755118279)
    (('which', 'has'), 1.9052853781396912)
    ((';', 'then'), 1.9051656573777542)
    ((',', 'n.e.2d'), 1.9032324547879327)
    (('comp.', ','), 1.9032324547879327)
    (('the', 'police'), 1.9020862644457832)
    (('after', 'judgment'), 1.901701802074184)
    (('it', 'was'), 1.9012069197592147)
    (('or', 'suit'), 1.9011980501779764)
    (('of', 'county'), 1.8981792270616467)
    (('whether', 'had'), 1.8980720073342834)
    (('he', 'right'), 1.8970181042296428)
    (('and', 'false'), 1.8963040313123187)
    (('and', 'nothing'), 1.8963040313123187)
    (('hundred', 'and'), 1.8963040313123187)
    (('absence', 'of'), 1.894074828728911)
    (('johns.', ','), 1.893833756785682)
    (('very', 'that'), 1.8924465045821357)
    (('bond', 'was'), 1.8907996966960496)
    (('promise', 'to'), 1.8904644661692966)
    (('to', 'paid'), 1.8904644661692966)
    (('matter', 'in'), 1.8879803709276786)
    (('costs', 'judgment'), 1.8877626111163082)
    (('?', 'not'), 1.886660490330236)
    (('law', 'can'), 1.886491657309847)
    (('what', 'was'), 1.885900897283209)
    (('office', 'of'), 1.882847573305657)
    (('.', '7'), 1.8825823720106296)
    (('from', 'any'), 1.8797712906994128)
    (('a', 'demurrer'), 1.8792714625827927)
    (('or', 'his'), 1.8784779736778923)
    ((',', '1822'), 1.876760243426741)
    ((',', 'p.'), 1.876760243426741)
    (('corporations', ','), 1.876760243426741)
    (('daughter', ','), 1.876760243426741)
    (('acts', 'of'), 1.8761529207316485)
    (('admit', 'of'), 1.8761529207316485)
    (('meaning', 'of'), 1.8761529207316485)
    (('of', 'forcible'), 1.8761529207316485)
    (('of', 'incorporation'), 1.8761529207316485)
    (('of', 'tract'), 1.8761529207316485)
    (('presence', 'of'), 1.8761529207316485)
    (('service', 'of'), 1.8761529207316485)
    (('statement', 'of'), 1.8761529207316485)
    (('affidavit', 'and'), 1.8733428828690073)
    (('and', 'award'), 1.8733428828690073)
    (('which', 'was'), 1.87192168649036)
    (('that', 'had'), 1.8709667771716845)
    (('appeal', 'court'), 1.8704618086537934)
    (('yet', 'court'), 1.8704618086537934)
    (('in', 'gave'), 1.8691213436763636)
    (('1825', ','), 1.8674824298387307)
    ((':', 'that'), 1.8666680633910282)
    ((';', 'v.'), 1.8647542550586884)
    (('by', 'other'), 1.8625248068412787)
    (('scates', ','), 1.8616533510365336)
    (('assess', 'the'), 1.8614442799484365)
    (('dismissing', 'the'), 1.8614442799484365)
    (('the', '17th'), 1.8614442799484365)
    (('the', 'client'), 1.8614442799484365)
    (('the', 'close'), 1.8614442799484365)
    (('the', 'event'), 1.8614442799484365)
    (('the', 'idea'), 1.8614442799484365)
    (('the', 'maker'), 1.8614442799484365)
    (('the', 'payee'), 1.8614442799484365)
    (('the', 'sufficiency'), 1.8614442799484365)
    (('the', 'transfer'), 1.8614442799484365)
    (('on', 'its'), 1.8603471535359084)
    (('all', 'are'), 1.8603471535359066)
    (('.', '1819'), 1.8598622955105455)
    (('1819', '.'), 1.8598622955105455)
    (('which', 'decision'), 1.8581727823015708)
    (('action', 'at'), 1.8566960795387448)
    (('is', 'intended'), 1.8564356417655148)
    (('is', 'shown'), 1.8564356417655148)
    (('is', 'therefore'), 1.8564356417655148)
    (('every', 'is'), 1.856435641765513)
    (('22d', ','), 1.8547339370967428)
    (('statutes', ','), 1.8547339370967428)
    (('and', 'return'), 1.8544838556176924)
    (('not', 'sufficient'), 1.8525347064256117)
    (('that', 'shall'), 1.8524308257342597)
    (('by', 'because'), 1.8510291680034499)
    (('held', 'by'), 1.8510291680034499)
    (('opinion', 'court'), 1.8505622512160862)
    (('assignee', 'of'), 1.8496807093704568)
    (('delivered', 'of'), 1.8496807093704568)
    (('entry', 'of'), 1.8496807093704568)
    (('of', '22d'), 1.8496807093704568)
    (('of', 'entry'), 1.8496807093704568)
    (('of', 'establishing'), 1.8496807093704568)
    (('of', 'king'), 1.8496807093704568)
    (('?', 'this'), 1.8487738021261677)
    (('order', 'to'), 1.8486442904746703)
    (('sufficient', 'to'), 1.8486442904746703)
    (('to', 'heirs'), 1.8486442904746703)
    (('.', 'however'), 1.8481488749467552)
    (('true', 'that'), 1.8480523852236814)
    (('cases', 'have'), 1.8468323470614258)
    (('was', 'received'), 1.8457118071675112)
    (('award', 'it'), 1.8412383305882027)
    (('of', 'years'), 1.8396270447065355)
    (('case', 'could'), 1.8386637744813257)
    (('a', 'competent'), 1.838629478085446)
    (('a', 'constable'), 1.838629478085446)
    (('was', 'brought'), 1.8378066090821648)
    (('amount', 'of'), 1.8366245565450114)
    ((',', '23'), 1.8361182589293943)
    ((',', 'whatever'), 1.8361182589293943)
    (('ill.app.2d', ','), 1.8361182589293943)
    (('martin', ','), 1.8361182589293943)
    (('thomas', ','), 1.8361182589293943)
    (('court', 'reversed'), 1.8358156658469014)
    (('.', 'v.'), 1.8351505420137926)
    (('but', 'question'), 1.834468521710301)
    (('by', 'their'), 1.8339556546445053)
    (('of', 'purchase'), 1.8310650312031083)
    (('court', 'new'), 1.8309334444671546)
    (('subject', 'be'), 1.83062645541591)
    (('16', '.'), 1.8301149521164923)
    (('lands', 'in'), 1.828479359179017)
    (('referred', 'in'), 1.828479359179017)
    (('not', 'nor'), 1.8265394978926643)
    (('paper', 'it'), 1.8261314381979954)
    (('made', 'before'), 1.8254176113386738)
    (('here', 'is'), 1.8247267820381765)
    (('that', 'used'), 1.8220571766907376)
    (('judgment', 'court'), 1.821910063821683)
    (('where', 'judgment'), 1.8200180044804757)
    (('.', 'johns'), 1.819846616662666)
    (('persons', 'in'), 1.8184952706063964)
    (('this', 'constitution'), 1.8184001530826492)
    (('issue', 'was'), 1.8182310707454032)
    (('would', 'with'), 1.8172592316780793)
    (('opinion', 'justice'), 1.816879124959799)
    (('and', 'became'), 1.8138418711203457)
    (('court', 'give'), 1.8115681196002225)
    (('see', 'to'), 1.8114912920572053)
    (('note', 'or'), 1.8100771988799487)
    (('the', 'circuit'), 1.8096053484288177)
    (('.', 'smith'), 1.8085817905668513)
    (('president', 'to'), 1.8080023059773236)
    (('be', 'brought'), 1.806779713461541)
    (('person', 'as'), 1.806779713461541)
    (('agent', 'of'), 1.806611987478572)
    ((',', 'perhaps'), 1.8063709155353447)
    (('armstrong', ','), 1.8063709155353447)
    (('cas.', ','), 1.8063709155353447)
    (('scott', ','), 1.8063709155353447)
    (('be', 'made'), 1.8031457189938003)
    (('affirm', 'the'), 1.8025505908948674)
    (('let', 'the'), 1.8025505908948674)
    (('raise', 'the'), 1.8025505908948674)
    (('stating', 'the'), 1.8025505908948674)
    (('the', 'applicant'), 1.8025505908948674)
    (('the', 'cashier'), 1.8025505908948674)
    (('the', 'executive'), 1.8025505908948674)
    (('the', 'old'), 1.8025505908948674)
    (('the', 'principal'), 1.8025505908948674)
    (('the', 'rendition'), 1.8025505908948674)
    (('decree', 'be'), 1.8008791120218568)
    (('as', 'point'), 1.8006536106646998)
    (('without', 'an'), 1.7994181548935764)
    (('a', 'principle'), 1.799101113898809)
    (('duty', 'of'), 1.7981504087303755)
    (('purpose', 'of'), 1.7981504087303755)
    (('view', 'of'), 1.7981504087303755)
    (('judgment', 'execution'), 1.7981253986316315)
    ((';', 'so'), 1.7972503182108674)
    (('where', 'been'), 1.796001123361938)
    (('court', 'think'), 1.7924592966525204)
    (('the', 'united'), 1.792282255441041)
    (('answer', 'to'), 1.792060762108303)
    (('deemed', 'to'), 1.792060762108303)
    (('as', 'evidence'), 1.7915129568082317)
    (('object', 'of'), 1.7912640231451356)
    (('smith', 'was'), 1.7912640231451356)
    (('of', 'state'), 1.7906452390735055)
    (('but', 'whether'), 1.7902687177997674)
    (('years', '.'), 1.7894729676191474)
    (('courts', 'of'), 1.7882801647063147)
    (('he', 'his'), 1.7882801647063147)
    (('judgment', 'justice'), 1.788226937565394)
    (('but', 'as'), 1.7879017032558515)
    (('not', 'say'), 1.7858975133953194)
    (('of', 'governor'), 1.785550371950741)
    (('of', 'inquiry'), 1.785550371950741)
    (('of', 'property'), 1.785550371950741)
    (('(', 'where'), 1.784382040095105)
    (('debt', 'by'), 1.7839149721449115)
    (('ibid.', ','), 1.7836508390352606)
    (('ills.', ','), 1.7836508390352606)
    (('liable', 'to'), 1.7835492622527838)
    (('commenced', 'in'), 1.7833914696504785)
    (('prisoner', 'in'), 1.7833914696504785)
    (('there', 'any'), 1.782825734614205)
    (('true', 'it'), 1.781737318839543)
    (('before', 'they'), 1.7807485620049448)
    (('justices', 'of'), 1.7807333556529663)
    (('oyer', 'of'), 1.7807333556529663)
    (('the', 'assignor'), 1.7805242845648692)
    (('the', 'supreme'), 1.7805242845648692)
    (('writ', 'of'), 1.7785976113089745)
    (('must', 'have'), 1.7785263336705182)
    (('in', 'following'), 1.77835587975318)
    (('statute', 'state'), 1.7775890862480583)
    (('an', 'taken'), 1.7773918485635782)
    (('him', 'as'), 1.7764060644180226)
    (('rep.', ','), 1.7754769075895602)
    (('bill', 'was'), 1.773562021411676)
    (('case', 'i'), 1.772666215707087)
    (('a', 'decree'), 1.772356258666278)
    (('at', 'one'), 1.7698186288587774)
    (('one', 'at'), 1.7698186288587774)
    (('or', 'should'), 1.7697423341370122)
    (('it', 'decided'), 1.767237749144428)
    ((',', '17'), 1.765728931037998)
    (('6', ','), 1.765728931037998)
    (('per', ','), 1.765728931037998)
    (('ground', 'that'), 1.7631634876371685)
    (('within', 'of'), 1.762623975101235)
    (('affix', 'the'), 1.7619086063975207)
    (('the', 'father'), 1.7619086063975207)
    (('a', 'particular'), 1.760626966084173)
    (('penalty', 'a'), 1.760626966084173)
    (('defendant', 'has'), 1.7603067338276084)
    (('having', 'in'), 1.7589384259259404)
    (('on', 'contract'), 1.7587761703962315)
    (('such', 'will'), 1.7579651663606572)
    (('a', 'suit'), 1.7577094827018769)
    (('given', 'to'), 1.7555348860831899)
    (('this', 'court'), 1.752547605788024)
    (('not', 'what'), 1.7512513705884292)
    ((',', '19'), 1.751229361342883)
    (('in', 'particular'), 1.7504768471777439)
    (('in', 'view'), 1.7504768471777439)
    (('below', 'be'), 1.749706460032339)
    (('his', 'right'), 1.7494609158157868)
    (('exercise', 'a'), 1.7480269293045367)
    (('but', 'it'), 1.747872424277496)
    (('objection', 'not'), 1.7463691492086824)
    (('governor', 'on'), 1.7461761336165953)
    (('is', 'words'), 1.74540432937677)
    (('ill.', '('), 1.745211442814428)
    (('and', 'rendered'), 1.7443009378672691)
    ((':', 'was'), 1.7427655054539457)
    (('wheat.', ','), 1.7392567196768063)
    (('case', 'consideration'), 1.737900797546411)
    (('sworn', 'and'), 1.7358393591190726)
    (('scam.', ','), 1.7340200713106597)
    (('by', 'bill'), 1.7337914927190763)
    ((';', 'et'), 1.7336851894334)
    (('did', 'on'), 1.7336851894334)
    (('the', 'obligation'), 1.7321612630034693)
    (('smith', '.'), 1.7305792785655782)
    (('support', '.'), 1.7305792785655782)
    (('to', 'seal'), 1.7299997939760505)
    (('co.', ','), 1.7292030550128814)
    (('wash', ','), 1.7292030550128814)
    (('sheriff', 'as'), 1.7291003496396655)
    (('deed', 'was'), 1.7285282677971736)
    (('not', 'make'), 1.7270038243417503)
    (('children', 'of'), 1.724149827286599)
    (('during', 'of'), 1.724149827286599)
    (('of', 'debt'), 1.724149827286599)
    (('of', 'lands'), 1.724149827286599)
    (('recess', 'of'), 1.724149827286599)
    (('secretary', 'of'), 1.724149827286599)
    (('services', 'of'), 1.724149827286599)
    (('sum', 'of'), 1.724149827286599)
    (('description', 'of'), 1.7241498272865972)
    (('of', 'debet'), 1.7241498272865972)
    (('of', 'intestate'), 1.7241498272865972)
    (('and', 'senate'), 1.7232393223394364)
    (('most', 'and'), 1.7232393223394364)
    (('one', 'or'), 1.7231997481999812)
    (('or', 'consideration'), 1.7231997481999795)
    (('such', 'may'), 1.7231997481999795)
    (('to', 'commission'), 1.7231134083908124)
    (('the', 'duties'), 1.7200884307028943)
    (('on', 'bill'), 1.7195410071131292)
    (('is', 'affirmed'), 1.71893211801558)
    (('will', 'from'), 1.7186164988848311)
    (('under', 'which'), 1.7175123753644552)
    (('which', 'may'), 1.7153639087858785)
    (('20', ','), 1.7132615111438625)
    (('was', 'evidence'), 1.7132615111438625)
    (('the', '18th'), 1.709441186503387)
    (('the', 'assignment'), 1.709441186503387)
    (('the', 'highest'), 1.709441186503387)
    (('same', 'as'), 1.7092918685594842)
    ((',', '21'), 1.7068352419844288)
    (('answer', 'that'), 1.7065799592708029)
    (('pleas', 'that'), 1.7065799592708029)
    (('that', 'prisoner'), 1.7065799592708029)
    ((';', 'and'), 1.7055601551046102)
    (('.', '2.'), 1.7045840700326345)
    (('it', 'be'), 1.704501993796466)
    (('as', '“'), 1.7042562786621858)
    (('said', 'defendant'), 1.7026107991316621)
    (('decision', 'on'), 1.7020535803842876)
    (('bank', 'is'), 1.7003164398482298)
    (('the', 'register'), 1.699980857254321)
    (('.', 'reverse'), 1.6988704188382417)
    (('costs', 'and'), 1.6983646537004091)
    (('the', 'appellant'), 1.6970574620475567)
    (('for', 'plaintiff'), 1.696238788858203)
    (('against', 'law'), 1.696102553307819)
    (('brought', 'this'), 1.6950177375773663)
    (('and', 'therefore'), 1.6935476374026344)
    (('.', 'al'), 1.6934262801481132)
    (('be', 'true'), 1.6931229316659735)
    (('proceedings', 'be'), 1.6931229316659735)
    (('13', ','), 1.6917283495942215)
    (('7', ','), 1.6917283495942215)
    (('trial', 'was'), 1.691728349594218)
    (('obtain', 'the'), 1.6915192785061244)
    (('the', '31st'), 1.6915192785061244)
    (('the', 'commencement'), 1.6915192785061244)
    (('.', 'therefore'), 1.6899372940682333)
    (('whether', 'was'), 1.6881705301810292)
    (('purple', ','), 1.687726419036725)
    (('of', 'bank'), 1.6866751218679354)
    (('application', 'a'), 1.6866263846403964)
    (('a', 'debt'), 1.6866263846403946)
    (('a', 'land'), 1.6866263846403946)
    (('a', 'years'), 1.6866263846403946)
    (('debt', 'a'), 1.6866263846403946)
    (('up', 'a'), 1.6866263846403946)
    (('no', 'any'), 1.6863177537608607)
    (('said', 'had'), 1.685932057985033)
    (('obligation', 'to'), 1.6851455581917918)
    (('remedy', 'to'), 1.6851455581917918)
    (('to', 'merits'), 1.6851455581917918)
    (('cause', 'court'), 1.6850357136712937)
    (('*', 'this'), 1.6845444063478592)
    (('principle', 'that'), 1.6829931389531847)
    (('this', 'is'), 1.6819379105333745)
    (('supreme', 'of'), 1.6797557079281447)
    (('than', 'that'), 1.679099222848695)
    (('court', 'are'), 1.6783015887367583)
    ((';', 'such'), 1.6781118586714445)
    (('regulating', 'the'), 1.6770197088110095)
    (('in', 'present'), 1.6764762657339674)
    (('in', 'thereof'), 1.6764762657339674)
    (('offense', 'in'), 1.6764762657339674)
    (('is', 'liable'), 1.6758633961236917)
    (('justice', 'was'), 1.675786805725199)
    (('recovered', '.'), 1.6739957501992109)
    (('cases', 'been'), 1.671794422225462)
    (('not', 'good'), 1.6704202959753847)
    (('that', 'plaintiffs'), 1.670054083245688)
    (('to', 'her'), 1.668072044832849)
    (('it', 'not'), 1.6661708229740277)
    (('new', 'be'), 1.6655672091454115)
    (('gilm.', ','), 1.6652561382330298)
    (('office', 'by'), 1.6650787426350622)
    (('delivered', 'the'), 1.6650470671449327)
    (('the', 'submission'), 1.6650470671449327)
    (('plea', 'an'), 1.6633566053175475)
    (('may', 'his'), 1.6627492826224533)
    (('and', 'ever'), 1.6618387776752943)
    (('and', 'sold'), 1.6618387776752943)
    (('overruled', 'and'), 1.6618387776752943)
    (('v.', ','), 1.6608688209920892)
    (('relation', 'of'), 1.6600194898668832)
    (('not', 'assigned'), 1.6598896284832136)
    (('he', 'only'), 1.659746794460931)
    (('he', 'so'), 1.659746794460931)
    (('?', 'have'), 1.658565709629606)
    (('1819', ','), 1.6581199569514027)
    (('who', 'been'), 1.657294852530347)
    (('a', 'between'), 1.6568790412463414)
    ((':', 'if'), 1.6562517710225304)
    (('application', 'to'), 1.655999212532274)
    (('requires', 'to'), 1.655999212532274)
    (('id.', ','), 1.6555460132875766)
    (('the', 'fourth'), 1.6549934024810078)
    (('by', 'one'), 1.654631955199946)
    (('county', 'v.'), 1.6541872691190296)
    (('of', 'government'), 1.6537604993952009)
    (('any', 'action'), 1.6527651940498327)
    (('trial', 'on'), 1.6523541112600775)
    (('ante', ','), 1.6521999854075844)
    (('the', 'peace'), 1.6519909143194873)
    (('for', 'its'), 1.6511508993296626)
    (('be', 'unless'), 1.6500542097740887)
    (('liable', 'be'), 1.6500542097740887)
    (('real', 'be'), 1.6500542097740887)
    (('out', 'of'), 1.6488616999823584)
    (('the', 'lieutenant'), 1.646431388977586)
    (('the', 'prisoner'), 1.646431388977586)
    (('of', 'witnesses'), 1.6461473152853259)
    (('in', 'this'), 1.6457140119632783)
    (('but', 'one'), 1.6444975780662308)
    (('brought', 'for'), 1.6443776707342384)
    (('the', 'words'), 1.6426096780251775)
    (('a', 'doubt'), 1.642232265281942)
    (('to', 'effect'), 1.6400576686632533)
    (('was', 'therefore'), 1.639260929700086)
    (('the', 'complainant'), 1.6390518586119889)
    (('the', 'intestate'), 1.6390518586119889)
    (('refused', '.'), 1.637469874174096)
    (('principle', 'in'), 1.6369479015473303)
    (('causes', 'of'), 1.6366869860362598)
    (('of', 'her'), 1.6366869860362598)
    (('ill.', ','), 1.636093650621401)
    (('a', 'note'), 1.6360003115704274)
    (('this', 'plea'), 1.6339755819452222)
    (('that', 'has'), 1.6316338649318674)
    (('note', 'case'), 1.6283810354228745)
    (('process', 'is'), 1.6281666540923947)
    (('a', 'nature'), 1.6277326955868254)
    (('aside', 'a'), 1.6277326955868254)
    (('the', 'plaintiffs'), 1.6269790263114103)
    (('in', 'making'), 1.6258501926639983)
    (('that', 'exceptions'), 1.6256599638872338)
    ((')', 'statute'), 1.6251607326552318)
    (('case', 'where'), 1.6250683013820364)
    (('judgment', 'costs'), 1.6247282052825156)
    (('to', 'appointment'), 1.6230845900595376)
    (('assigned', 'this'), 1.6220029402791454)
    (('which', 'is'), 1.6219703881284921)
    (('where', 'a'), 1.6188817780045603)
    (('what', 'is'), 1.6176487821783976)
    (('in', 'aside'), 1.6175825766803982)
    (('the', 'president'), 1.6175186970623479)
    (('of', 'states'), 1.6172346233700878)
    (('appears', 'to'), 1.6159835336843962)
    (('then', 'been'), 1.6146505151218538)
    ((',', '22'), 1.6137258375929484)
    ((',', 'c.'), 1.6137258375929484)
    ((',', 'directions'), 1.6137258375929484)
    ((',', 'ills'), 1.6137258375929484)
    ((',', 'ills.'), 1.6137258375929484)
    ((',', 'inasmuch'), 1.6137258375929484)
    ((',', 'inclined'), 1.6137258375929484)
    ((',', 'leaving'), 1.6137258375929484)
    ((',', 'scales'), 1.6137258375929484)
    ((',', 'uses'), 1.6137258375929484)
    (('11', ','), 1.6137258375929484)
    (('12', ','), 1.6137258375929484)
    (('1821', ','), 1.6137258375929484)
    (('1822', ','), 1.6137258375929484)
    (('18th', ','), 1.6137258375929484)
    (('24', ','), 1.6137258375929484)
    (('ankeny', ','), 1.6137258375929484)
    (('beaird', ','), 1.6137258375929484)
    (('chenoweth', ','), 1.6137258375929484)
    (('company', ','), 1.6137258375929484)
    (('constrained', ','), 1.6137258375929484)
    (('crouch', ','), 1.6137258375929484)
    (('dane', ','), 1.6137258375929484)
    (('instruments', ','), 1.6137258375929484)
    (('judges', ','), 1.6137258375929484)
    (('mason', ','), 1.6137258375929484)
    (('mayo', ','), 1.6137258375929484)
    (('mccann', ','), 1.6137258375929484)
    (('moore', ','), 1.6137258375929484)
    (('overseers', ','), 1.6137258375929484)
    (('t.', ','), 1.6137258375929484)
    (('thompson', ','), 1.6137258375929484)
    (('widow', ','), 1.6137258375929484)
    (('york', ','), 1.6137258375929484)
    ((',', '18'), 1.6137258375929466)
    ((',', '275'), 1.6137258375929466)
    ((',', 'capital'), 1.6137258375929466)
    ((',', 'ill.app.2d'), 1.6137258375929466)
    ((',', 'j.'), 1.6137258375929466)
    ((',', 'viz'), 1.6137258375929466)
    (('18', ','), 1.6137258375929466)
    (('baker', ','), 1.6137258375929466)
    (('dig.', ','), 1.6137258375929466)
    (('duncan', ','), 1.6137258375929466)
    (('jones', ','), 1.6137258375929466)
    (('russell', ','), 1.6137258375929466)
    (('sawyer', ','), 1.6137258375929466)
    (('sims', ','), 1.6137258375929466)
    (('vanlandingham', ','), 1.6137258375929466)
    (('whiteside', ','), 1.6137258375929466)
    (('williams', ','), 1.6137258375929466)
    ((',', 'though'), 1.6137258375929449)
    ((',', 'wit'), 1.6137258375929449)
    (('14', ','), 1.6137258375929449)
    (('16', ','), 1.6137258375929449)
    (('.', 'purple'), 1.6119347820669585)
    (('that', 'were'), 1.6111603941921189)
    (('for', 'time'), 1.6093307236350363)
    (('of', 'circumstances'), 1.6086726098666606)
    (('purchase', 'of'), 1.6086726098666606)
    (('settled', 'a'), 1.6086238726391233)
    (('show', 'a'), 1.6086238726391233)
    (('bond', 'be'), 1.6082340340794605)
    (('the', 'following'), 1.6082289351450854)
    (('that', 'recover'), 1.607044285719887)
    (('in', 'injunction'), 1.6060869378425693)
    (('a', 'sufficient'), 1.6057063892568273)
    (('the', 'defendants'), 1.605104526688649)
    (('the', 'plaintiff'), 1.6040564372557853)
    (('which', 'would'), 1.6038555935688876)
    (('he', 'not'), 1.6033061001168498)
    (('said', 'said'), 1.6030751255807463)
    (('out', 'in'), 1.6011881384297268)
    (('they', 'not'), 1.5982705102195496)
    (('has', 'any'), 1.5978568529474444)
    (('said', 'upon'), 1.5974940933174473)
    (('was', 'after'), 1.5966165922915927)
    (('which', 'party'), 1.5951383764677782)
    (('is', 'entitled'), 1.5934012359317204)
    (('is', 'say'), 1.5934012359317187)
    (('and', 'offered'), 1.591449449783898)
    (('shall', 'by'), 1.5892220655536562)
    (('demurrer', 'court'), 1.5891756982637766)
    (('in', 'circuit'), 1.5890134244836283)
    (('the', 'same'), 1.5878792066224747)
    (('22d', 'of'), 1.5866463035366642)
    (('charge', 'of'), 1.5866463035366642)
    (('happen', 'of'), 1.5866463035366642)
    (('ordinary', 'of'), 1.5866463035366642)
    (('reversed', 'and'), 1.5865506503710556)
    (('exception', 'to'), 1.5856098846408777)
    (('title', 'to'), 1.5856098846408777)
    (('to', 'amendment'), 1.5856098846408777)
    (('to', 'morrison'), 1.5856098846408777)
    (('too', 'to'), 1.5856098846408777)
    (('to', 'correct'), 1.585609884640876)
    (('that', '“'), 1.585520573034021)
    (('.', 'appears'), 1.5853813630256433)
    (('suit', 'have'), 1.5837979412276333)
    (('by', 'defendant'), 1.5826165824430873)
    (('the', 'clerk'), 1.5801581695584197)
    (('has', 'from'), 1.5783166665517996)
    (('considered', 'it'), 1.5782039247544102)
    (('notice', 'it'), 1.5782039247544102)
    (('a', 'general'), 1.5781019278622281)
    (('in', 'every'), 1.5769405921830515)
    (('note', 'had'), 1.5761439124469199)
    (('who', 'not'), 1.5750007308967007)
    (('”', 'this'), 1.5749461162183707)
    (('verdict', 'and'), 1.574375936424957)
    (('now', 'be'), 1.5742867021561224)
    ((',', 'ante'), 1.5741974734063113)
    (('was', 'before'), 1.57291843487139)
    (('a', 'taking'), 1.571149167220458)
    (('commenced', 'a'), 1.571149167220458)
    (('is', 'brought'), 1.5695544939773534)
    (('which', 'have'), 1.5692983715325184)
    (('consideration', 'was'), 1.568871601808688)
    (('the', 'court'), 1.5676397859754623)
    (('.', 'doctrine'), 1.5670805462826998)
    (('.', 'johns.'), 1.5670805462826998)
    (('awarded', '.'), 1.5670805462826998)
    (('cited', '.'), 1.5670805462826998)
    (('johns.', '.'), 1.5670805462826998)
    (('decree', 'is'), 1.563653892537669)
    (('is', 'between'), 1.563653892537669)
    (('presented', 'in'), 1.5609990483140308)
    (('which', 'its'), 1.560786871677001)
    (('a', 'decision'), 1.5595144667370562)
    (('to', 'set'), 1.559137673279686)
    (('no', '('), 1.5582129280131731)
    (('law', 'but'), 1.5581372932630515)
    (('.', 'laws'), 1.5577426817033633)
    ((',', 'chief'), 1.557142309226581)
    (('record', 'on'), 1.5570200417540825)
    ((',', 'ill.'), 1.5562403429321883)
    (('state', '”'), 1.5551966649116107)
    (('than', 'was'), 1.5551966649116107)
    (('was', 'necessary'), 1.5551966649116107)
    (('said', 'he'), 1.553966717776527)
    (('principle', 'is'), 1.5538728717450816)
    (('one', 'which'), 1.5513881736747521)
    (('johns', ','), 1.5509900822449865)
    (('.', 'replication'), 1.550007032923757)
    (('paid', '.'), 1.550007032923757)
    (('that', 'persons'), 1.5490386822843227)
    (('that', 'was'), 1.5485905707600232)
    (('states', 'v.'), 1.5472720652025185)
    (('taking', 'and'), 1.5463615602553578)
    (('be', 'evidence'), 1.5461730659472082)
    (('is', 'further'), 1.5460955211533634)
    (('this', 'whether'), 1.5453816586762308)
    (('note', 'to'), 1.5452521470247333)
    (('do', 'it'), 1.5448453278079803)
    (('had', 'before'), 1.5433539773292502)
    (('illinois', ','), 1.5433365097015503)
    (('question', 'are'), 1.5428649636797385)
    (('is', 'no'), 1.5415623044120998)
    (('debt', 'be'), 1.5411198382209221)
    (('present', 'is'), 1.540933816037585)
    (('more', 'that'), 1.5407710663007208)
    (('a', 'sale'), 1.5397849963111234)
    (('affect', 'the'), 1.5395161850610748)
    (('directs', 'the'), 1.5395161850610748)
    (('expire', 'the'), 1.5395161850610748)
    (('exposition', 'the'), 1.5395161850610748)
    (('one-half', 'the'), 1.5395161850610748)
    (('prosecuting', 'the'), 1.5395161850610748)
    (('the', 'agency'), 1.5395161850610748)
    (('the', 'american'), 1.5395161850610748)
    (('the', 'appraisement'), 1.5395161850610748)
    (('the', 'arbitration'), 1.5395161850610748)
    (('the', 'arbitrators'), 1.5395161850610748)
    (('the', 'assembly'), 1.5395161850610748)
    (('the', 'assignee'), 1.5395161850610748)
    (('the', 'computation'), 1.5395161850610748)
    (('the', 'incorporation'), 1.5395161850610748)
    (('the', 'opposite'), 1.5395161850610748)
    (('the', 'premises'), 1.5395161850610748)
    (('the', 'presence'), 1.5395161850610748)
    (('the', 'purchaser'), 1.5395161850610748)
    (('the', 'taker'), 1.5395161850610748)
    (('the', 'truth'), 1.5395161850610748)
    (('conceive', 'the'), 1.539516185061073)
    (('overruling', 'the'), 1.539516185061073)
    (('the', 'condition'), 1.539516185061073)
    (('the', 'devisee'), 1.539516185061073)
    (('the', 'died'), 1.539516185061073)
    (('the', 'former'), 1.539516185061073)
    (('the', 'mentioned'), 1.539516185061073)
    (('the', 'mother'), 1.539516185061073)
    (('the', 'precise'), 1.539516185061073)
    (('the', 'usual'), 1.539516185061073)
    (('try', 'the'), 1.539516185061073)
    (('the', 'next'), 1.5395161850610712)
    (('the', 'performance'), 1.5395161850610712)
    (('in', 'courts'), 1.5389727419840327)
    (('4', ','), 1.538437710288708)
    (('judgment', '.'), 1.538428358888293)
    (('to', 'them'), 1.5383041698625206)
    (('.', 'also'), 1.5379342006231838)
    (('recess', '.'), 1.537934200623182)
    (('note', 'v.'), 1.5372879766298944)
    (('its', 'as'), 1.5369401297226322)
    (('15', ','), 1.5357233255916753)
    (('22', ','), 1.5357233255916753)
    (('“', 'that'), 1.5348944999640501)
    (('a', 'manner'), 1.5346232911953468)
    ((',', 'scam.'), 1.5331073773846633)
    ((',', 'gilm.'), 1.532805842209381)
    (('and', 'clerk'), 1.5325557607303288)
    (('17', ','), 1.5312636774009754)
    (('it', 'must'), 1.5290067343112064)
    (('principle', 'be'), 1.5280636853954768)
    (('is', 'debt'), 1.526287040073182)
    ((',', 'id.'), 1.5262629963426093)
    (('manner', 'in'), 1.5244731722889178)
    (('form', 'and'), 1.5243352539253614)
    (('writ', 'be'), 1.5225041600535754)
    (('was', 'an'), 1.520616433201468)
    (('but', 'may'), 1.5189666959823729)
    (('that', 'party'), 1.5174070732352085)
    (('governor', 'this'), 1.517230618362083)
    (('so', 'it'), 1.5171423990181303)
    (('the', 'constitution'), 1.516796108560989)
    (('a', 'executed'), 1.5167013831980842)
    (('a', 'great'), 1.5167013831980825)
    (('a', 'rule'), 1.5167013831980825)
    (('a', 'title'), 1.5167013831980825)
    (('below', 'no'), 1.5163927523185468)
    (('contract', 'is'), 1.5153987239304474)
    (('is', 'proper'), 1.5153987239304456)
    (('think', 'is'), 1.5153987239304456)
    (('when', 'was'), 1.5149327946978843)
    (('21', ','), 1.5141901640420308)
    (('the', 'defendant'), 1.5141308666411604)
    (('must', 'as'), 1.5117028382236093)
    (('parties', 'to'), 1.510647826959655)
    (('if', 'it'), 1.509678473376784)
    (('sworn', 'to'), 1.5076073726396029)
    (('”', 'if'), 1.5072325360163)
    ((',', '16'), 1.5068106336764338)
    ((',', '20'), 1.5068106336764338)
    (('some', 'in'), 1.5065512642916552)
    (('in', 'another'), 1.5065512642916534)
    (('in', 'words'), 1.5065512642916534)
    (('him', 'by'), 1.506380996616004)
    (('an', 'from'), 1.5063128951719662)
    (('manner', 'it'), 1.5042033433106337)
    (('act', 'an'), 1.5030817739089564)
    (('there', 'a'), 1.503026446301007)
    (('case', 'is'), 1.5019028814462239)
    (('but', 'are'), 1.5018931826234319)
    (('of', 'personal'), 1.5017574059501513)
    (('obligation', 'of'), 1.5017574059501495)
    (('next', 'of'), 1.5017574059501477)
    (('in', 'cause'), 1.50103425932409)
    (('plaintiff', 'judgment'), 1.5007394880070617)
    (('however', ','), 1.5005152271449553)
    (('that', 'consideration'), 1.500129081803376)
    (('that', 'ground'), 1.500129081803376)
    (('not', 'being'), 1.5000386732154798)
    ((',', '11'), 1.4982486201730119)
    (('to', 'have'), 1.4976134036430473)
    (('on', 'i'), 1.4971485857766815)
    (('this', 'an'), 1.4964720581952875)
    (('court', 'states'), 1.4960662938722944)
    (('“', 'such'), 1.4949307605268647)
    (('these', 'is'), 1.4938655623808046)
    (('intention', 'to'), 1.4925004802493955)
    (('.', '4'), 1.4917924189784628)
    (('if', 'such'), 1.4916356809652829)
    (('a', 'power'), 1.4916104022352528)
    (('award', 'be'), 1.4895895375808426)
    (('be', 'proper'), 1.4895895375808426)
    (('be', 'settled'), 1.4895895375808426)
    (('be', 'show'), 1.4895895375808426)
    (('.', 'people'), 1.488606488259613)
    ((',', '14'), 1.488194955509087)
    (('8', ','), 1.488194955509087)
    (('we', 'not'), 1.4870530316209987)
    (('the', 'second'), 1.4870487651669357)
    (('a', 'inquiry'), 1.4849925234707442)
    (('a', 'term'), 1.4849925234707442)
    (('authority', 'to'), 1.4825163916767732)
    (('r.', ','), 1.482481304314696)
    (('that', 'they'), 1.4824270800699146)
    (('proceedings', 'court'), 1.4814195179078915)
    (('for', 'jury'), 1.4805974095128338)
    (('whether', 'or'), 1.479928597187616)
    (('execution', 'on'), 1.47966115904784)
    (('constitution', 'of'), 1.4790373294500654)
    (('held', 'to'), 1.478694680724363)
    (('section', 'is'), 1.4779240185117857)
    (('presented', 'is'), 1.4779240185117821)
    (('has', 'not'), 1.475892585883777)
    (('in', 'premises'), 1.474842404564317)
    (('property', 'in'), 1.474842404564317)
    (('under', 'this'), 1.4726253162409186)
    (('himself', 'of'), 1.4726110602906353)
    (('the', 'original'), 1.4724019892025382)
    (('is', 'only'), 1.4714107115531085)
    (('first', 'court'), 1.4705312017651586)
    (('effect', 'to'), 1.4701326672209412)
    (('and', 'those'), 1.469193699732898)
    (('the', 'below'), 1.4678147197068867)
    (('of', 'office'), 1.467810074026815)
    (('.', 'id'), 1.4675448727317857)
    (('.', 'statutes'), 1.4675448727317857)
    (('.', 'wilson'), 1.4675448727317857)
    (('account', '.'), 1.4675448727317857)
    (('appeared', '.'), 1.4675448727317857)
    (('proved', '.'), 1.4675448727317857)
    (('.', 'consider'), 1.467544872731784)
    (('.', 'object'), 1.467544872731784)
    (('2', 'judgment'), 1.4671869282960355)
    (('defense', 'in'), 1.4670229001050181)
    (('the', 'recess'), 1.4655156036172983)
    (('law', 'which'), 1.4650278888715675)
    (('defendant', 'have'), 1.464961711717784)
    (('to', 'be'), 1.4639804402393466)
    (('for', 'decision'), 1.4631099827839922)
    (('is', 'not'), 1.4626667592403209)
    (('set', 'by'), 1.4619868772575497)
    (('taken', 'by'), 1.461986877257548)
    (('19', ','), 1.4617227441478988)
    (('ills', ','), 1.4617227441478988)
    (('complaint', 'of'), 1.4611154214528046)
    (('of', 'complaint'), 1.4611154214528046)
    (('of', 'wash'), 1.4611154214528046)
    (('and', 'character'), 1.4602049165056439)
    (('and', 'most'), 1.4602049165056439)
    (('character', 'and'), 1.4602049165056439)
    (('executed', 'to'), 1.4600790025570163)
    (('an', 'one'), 1.4554637536762165)
    (('subject', 'court'), 1.4554243093749495)
    (('as', 'question'), 1.4544779695306609)
    (('as', 'would'), 1.4544779695306609)
    (('to', 'most'), 1.4543653513626236)
    (('writing', 'to'), 1.4543653513626236)
    (('pleaded', 'in'), 1.454083844397518)
    ((',', '15'), 1.4532611653997023)
    (('construction', 'that'), 1.4528233670250188)
    (('we', 'that'), 1.4528233670250188)
    (('the', 'administrator'), 1.4520533438107357)
    (('the', 'language'), 1.4520533438107357)
    (('the', 'people'), 1.4520533438107357)
    (('of', 'legislature'), 1.4511313328801805)
    (('is', 'however'), 1.4504432820896778)
    (('a', 'exceptions'), 1.449587187339544)
    (('been', 'upon'), 1.4478414869013978)
    ((',', '&'), 1.4477158861540182)
    (('the', 'remanded'), 1.4464067806695944)
    (('deed', 'be'), 1.4459626051805827)
    (('appearance', 'of'), 1.4440419080938618)
    ((',', '13'), 1.4438008361506363)
    ((',', '1821'), 1.4438008361506363)
    ((',', 'brothers'), 1.4438008361506363)
    (('31st', ','), 1.4438008361506363)
    (('aforesaid', ','), 1.4438008361506363)
    (('assembly', ','), 1.4438008361506363)
    (('six', ','), 1.4438008361506363)
    (('the', 'sheriff'), 1.4433008698017709)
    (('the', 'whole'), 1.4433008698017709)
    (('is', 'part'), 1.441398142486669)
    (('judgment', 'be'), 1.4396058372555984)
    (('bond', 'and'), 1.4394463563388484)
    (('and', 'gave'), 1.4394463563388467)
    (('in', 'below'), 1.4394370684331186)
    (('in', 'suit'), 1.4394370684331186)
    (('in', 'exceptions'), 1.4394370684331168)
    (('a', 'proper'), 1.4386988711968112)
    (('that', 'could'), 1.437393326455414)
    ((',', '1825'), 1.4368480755088697)
    (('where', 'it'), 1.4364587366747976)
    (('land', 'by'), 1.435991668724606)
    (('.', '&'), 1.4358360130044474)
    (('received', 'in'), 1.435468166230173)
    (('court', 'can'), 1.4346657492081505)
    (('pay', 'it'), 1.4338140154192356)
    (('sold', 'to'), 1.4336067911958263)
    (('the', 'doctrine'), 1.4326009811445601)
    (('the', 'merits'), 1.4326009811445601)
    (('here', 'not'), 1.4322605587806194)
    (('all', 'he'), 1.4317948477825837)
    (('he', 'have'), 1.4317948477825837)
    ((',', 'post'), 1.4293012664555214)
    (('they', 'been'), 1.4290258648572287)
    (('party', 'not'), 1.4287799900456584)
    (('the', 'jurors'), 1.4284848726723283)
    (('court', 'decree'), 1.4256769659808963)
    (('and', 'until'), 1.4247995803744438)
    (('sustained', 'and'), 1.4247995803744438)
    (('court', 'no'), 1.424544996490976)
    (('the', 'october'), 1.4240389676411382)
    (('obtained', 'a'), 1.423591978806602)
    (('which', 'been'), 1.4228295988933262)
    (('but', 'is'), 1.4220328176197405)
    ((',', '9'), 1.4210807596505504)
    ((',', 'collins'), 1.4210807596505504)
    ((',', 'deceased'), 1.4210807596505504)
    ((',', 'still'), 1.4210807596505504)
    ((',', 'two-thirds'), 1.4210807596505504)
    (('1823', ','), 1.4210807596505504)
    (('kent', ','), 1.4210807596505504)
    (('the', 'secretary'), 1.4208716885624568)
    (('them', '.'), 1.4202391579534286)
    (('such', 'an'), 1.4201306805639362)
    ((';', 'must'), 1.4200723627580398)
    (('plaintiff', 'in'), 1.419088423041316)
    (('have', 'so'), 1.4187386949571366)
    (('therefore', 'judgment'), 1.4182773278150886)
    (('as', 'does'), 1.4165101193316403)
    (('before', 'an'), 1.4164418648470303)
    (('5', '.'), 1.4150774528376502)
    (('been', 'against'), 1.4144383286341817)
    (('demurred', 'the'), 1.4139853029772134)
    (('obtained', 'in'), 1.413441859900173)
    (('that', 'plaintiff'), 1.4126662405530368)
    (('the', 'declaration'), 1.412604072578855)
    (('costs', '.'), 1.4109613443654183)
    (('the', 'governor'), 1.4082716517828189)
    (('that', 'paper'), 1.4070196774118955)
    (('given', 'by'), 1.4068453230650881)
    (('a', 'public'), 1.4065184654476592)
    (('.', 'if'), 1.4059058069393977)
    (('legal', 'and'), 1.4054990244155086)
    (('administrator', 'to'), 1.4050376389990564)
    (('nothing', 'to'), 1.4050376389990564)
    (('to', 'own'), 1.4050376389990564)
    (('to', 'replication'), 1.4050376389990564)
    (('it', 'from'), 1.4037431615314198)
    (('in', 'action'), 1.4034577713275525)
    (('this', 'can'), 1.403362653803807)
    (('by', 'act'), 1.4026896935737767)
    (('it', 'then'), 1.4026653168485694)
    (('each', 'of'), 1.4022217323992354)
    (('support', 'of'), 1.4022217323992354)
    (('with', 'any'), 1.4022217323992354)
    (('ground', 'of'), 1.4022217323992336)
    (('the', 'building'), 1.40201266131114)
    (('deed', 'not'), 1.4012336631599958)
    (('costs', 'a'), 1.4012241657781495)
    (('state', 'said'), 1.4009550992705595)
    (('was', 'counsel'), 1.4004740701129688)
    (('original', '.'), 1.4004306768732473)
    (('declaration', 'which'), 1.400322199483755)
    (('legal', 'it'), 1.3998666834958975)
    (('if', 'been'), 1.3997355767627049)
    (('too', 'and'), 1.3988043718415017)
    (('up', 'and'), 1.3988043718415017)
    (('years', 'and'), 1.3988043718415017)
    (('had', 'they'), 1.3985009978437706)
    (('was', 'by'), 1.3978565398378322)
    (('court', 'also'), 1.3965306203213785)
    (('court', 'demurrer'), 1.3965306203213785)
    (('well', 'in'), 1.3963683465412302)
    (('action', 'for'), 1.3952059182821905)
    (('the', 'government'), 1.3951262757259002)
    (('their', '('), 1.3947141957302946)
    (('have', 'right'), 1.3929755988920558)
    (('right', 'have'), 1.3929755988920558)
    (('admitted', 'to'), 1.3929648066984797)
    (('which', 'defendant'), 1.3919099596675544)
    (('madison', ','), 1.3913334162565008)
    ((',', 'scott'), 1.391333416256499)
    (('dissolved', ','), 1.391333416256499)
    (('former', ','), 1.391333416256499)
    (('negotiable', ','), 1.391333416256499)
    ((',', 'except'), 1.3913334162564972)
    ((',', 'johns.'), 1.3913334162564972)
    (('5', ','), 1.3913334162564972)
    (('before', 'said'), 1.390107717033569)
    (('his', 'action'), 1.3897307882160383)
    (('witnesses', '.'), 1.3895423607305109)
    ((';', 'it'), 1.3887261258906989)
    (('act', 'of'), 1.3876243561862758)
    (('such', 'been'), 1.387596716415544)
    (('the', 'certificate'), 1.3875130916160252)
    (('the', 'passage'), 1.3875130916160252)
    (('the', 'valuation'), 1.3875130916160252)
    (('are', 'by'), 1.3864966776732857)
    (('to', 'whole'), 1.386301076417471)
    (('.', 'this'), 1.3861565458911294)
    (('by', 'legislature'), 1.385365595654637)
    (('the', 'bank'), 1.3833969831437933)
    (('proper', 'to'), 1.3820764905557432)
    (('law', 'state'), 1.380963624083094)
    (('an', 'against'), 1.3798580969240373)
    ((',', '3'), 1.3792605839559258)
    (('3', ','), 1.3792605839559258)
    (('the', 'first'), 1.3790515128678287)
    (('the', 'seal'), 1.3790515128678287)
    ((',', 'however'), 1.3776584793594218)
    (('demurrer', 'by'), 1.377097979671035)
    (('was', 'decided'), 1.3762265238662916)
    (('of', 'next'), 1.3762265238662899)
    (('not', 'any'), 1.3760966624826239)
    (('within', 'the'), 1.3744569387905798)
    (('directly', '.'), 1.3744354683403035)
    (('final', '.'), 1.3744354683403035)
    (('johnson', '.'), 1.3744354683403035)
    (('is', 'taken'), 1.371008814595271)
    (('.', 'judgment'), 1.3709716130367298)
    (('of', 'premises'), 1.370512872671899)
    (('of', 'several'), 1.370512872671899)
    (('regard', 'the'), 1.3695911836187626)
    (('the', 'door'), 1.3695911836187626)
    (('the', 'mind'), 1.3695911836187626)
    (('of', 'constitution'), 1.3680060170613224)
    (('be', 'without'), 1.3672259345756643)
    (('by', 'person'), 1.3668296442172103)
    (('bill', 'of'), 1.366597822668517)
    (('a', 'granted'), 1.3646982897530329)
    (('a', 'ground'), 1.364698289753031)
    (('one', 'of'), 1.3642538822002166)
    (('return', 'to'), 1.36321746330443)
    (('to', '1.'), 1.36321746330443)
    (('where', 'was'), 1.3629297012528347)
    (('1', '.'), 1.3624628266742285)
    (('the', 'statute'), 1.3617290658457897)
    (('it', 'however'), 1.3612453894685892)
    (('legislature', 'not'), 1.3608759255438532)
    (('had', 'not'), 1.3608759255438514)
    (('governor', 'and'), 1.360669242954728)
    ((',', 'although'), 1.3584687823508723)
    (('after', 'an'), 1.3576778620752616)
    (('the', 'counsel'), 1.3573128538403232)
    (('that', 'there'), 1.3571711279613332)
    (('before', 'any'), 1.3569408530983704)
    (('judgment', '?'), 1.3555415724671267)
    (('also', 'in'), 1.354548170846602)
    (('.', 'chief'), 1.3520676553118491)
    (('.', 'questions'), 1.3520676553118491)
    (('between', 'the'), 1.3510710956479635)
    ((',', 'conclusive'), 1.350691431759154)
    ((',', 'died'), 1.350691431759154)
    ((',', 'furnish'), 1.350691431759154)
    ((',', 'penalties'), 1.350691431759154)
    (('23', ','), 1.350691431759154)
    (('capital', ','), 1.350691431759154)
    (('davis', ','), 1.350691431759154)
    (('descendants', ','), 1.350691431759154)
    (('guard', ','), 1.350691431759154)
    (('rebecca', ','), 1.350691431759154)
    (('cornelius', ','), 1.3506914317591523)
    (('statute', 'which'), 1.3470801782067738)
    (('authorizes', 'the'), 1.3468711071186767)
    (('entering', 'the'), 1.3468711071186767)
    (('hearing', 'the'), 1.3468711071186767)
    (('perceived', 'the'), 1.3468711071186767)
    (('the', 'erred'), 1.3468711071186767)
    (('the', 'july'), 1.3468711071186767)
    (('time', 'for'), 1.3462963178012437)
    (('of', 'payment'), 1.3456382040328698)
    (('taking', 'of'), 1.3456382040328698)
    (('circumstances', 'of'), 1.345638204032868)
    (('filing', 'of'), 1.345638204032868)
    (('of', 'even'), 1.345638204032868)
    (('of', 'substance'), 1.345638204032868)
    (('of', 'variance'), 1.345638204032868)
    (('“', 'has'), 1.3435926567413823)
    (('at', 'with'), 1.3431182120543532)
    (('or', 'will'), 1.3429276670818133)
    (('error', 'have'), 1.3427898417238389)
    (('is', 'plea'), 1.341862468935755)
    (('declaration', 'judgment'), 1.3402748158138156)
    (('declaration', 'and'), 1.3399106827879343)
    (('and', 'damages'), 1.3399106827879326)
    (('aside', 'and'), 1.3399106827879326)
    (('are', 'that'), 1.3386656591092603)
    (('that', 'appears'), 1.337857652904498)
    (('united', 'that'), 1.337857652904498)
    (('that', 'person'), 1.3378576529044963)
    (('that', 'may'), 1.3366303495204974)
    (('.', 'several'), 1.3363003394535315)
    (('but', 'there'), 1.3353667576429853)
    (('which', 'made'), 1.3353667576429853)
    (('give', 'it'), 1.3342783418683215)
    (('authorities', 'to'), 1.3340711176449105)
    (('at', 'but'), 1.3338552907253831)
    ((',', 'unless'), 1.3336179184002113)
    (('.', 'question'), 1.3332437810201938)
    (('not', 'with'), 1.3327248852297053)
    (('state', 'if'), 1.3326213809041665)
    (('judgment', 'defendant'), 1.331862575997615)
    (('practice', 'of'), 1.3318324045078391)
    (('is', 'appeal'), 1.3303668300979261)
    (('is', 'some'), 1.3303668300979261)
    (('third', 'is'), 1.3303668300979261)
    (('for', 'without'), 1.3303547739322212)
    (('the', 'amount'), 1.3300628194321256)
    (('an', 'may'), 1.3299328715923568)
    ((',', '5'), 1.329932871592355)
    (('a', 'bill'), 1.3290743800223126)
    (('have', 'against'), 1.326441847636353)
    (('on', 'which'), 1.3263216180399766)
    (('bill', 'as'), 1.3257446554084567)
    (('is', 'made'), 1.3249124000058181)
    (('he', 'can'), 1.3248796438660726)
    (('taken', 'on'), 1.3248796438660708)
    ((',', 'cranch'), 1.3242192203979641)
    ((',', 'wheat.'), 1.3242192203979641)
    (('a', 'third'), 1.3240563052556862)
    (('constable', 'to'), 1.3225754788070834)
    (('to', 'constable'), 1.3225754788070834)
    (('to', 'some'), 1.3225754788070816)
    (('this', 'being'), 1.321974326963149)
    (('when', 'it'), 1.320981519254861)
    (('court', 'whether'), 1.3199093387184675)
    (('in', 'bill'), 1.3189242611158853)
    (('plaintiff', 'for'), 1.3177271656044738)
    (('who', 'was'), 1.3173328348127242)
    (('the', 'practice'), 1.3171237637246271)
    (('affecting', 'the'), 1.3171237637246254)
    (('compliance', 'the'), 1.3171237637246254)
    (('condition', 'the'), 1.3171237637246254)
    (('prevent', 'the'), 1.3171237637246254)
    (('regards', 'the'), 1.3171237637246254)
    (('review', 'the'), 1.3171237637246254)
    (('sense', 'the'), 1.3171237637246254)
    (('the', 'car'), 1.3171237637246254)
    (('the', 'commissioners'), 1.3171237637246254)
    (('the', 'above'), 1.3171237637246236)
    (('question', 'case'), 1.3164370291081315)
    (('officer', 'to'), 1.316423251825487)
    (('their', 'this'), 1.315899812553468)
    (('of', 'opinion'), 1.3156849819312413)
    (('.', 'am'), 1.3155417792867343)
    (('void', '.'), 1.3155417792867343)
    (('from', 'such'), 1.3146172710584363)
    (('then', 'be'), 1.3140509296750018)
    (('jury', 'that'), 1.3125020786276025)
    (('not', 'given'), 1.3119663250629081)
    (('it', 'might'), 1.3115582653682356)
    (('that', 'all'), 1.3106512829396628)
    (('by', 'any'), 1.309983783812502)
    (('of', 'diligence'), 1.3091123280077532)
    (('is', 'an'), 1.3089478464630204)
    (('injunction', ','), 1.3088712560645277)
    (('state', 'at'), 1.3086847149904788)
    (('a', 'answer'), 1.3081147613866655)
    (('a', 'payment'), 1.3081147613866655)
    (('.', 'we'), 1.307764428695016)
    (('ill.', 'an'), 1.307212795092271)
    (('since', 'the'), 1.3050509314240522)
    (('the', 'name'), 1.3050509314240522)
    (('the', 'office'), 1.303075989238991)
    (('error', 'judgment'), 1.302800110395152)
    (('case', 'there'), 1.302762092211056)
    (('ante', '.'), 1.3024856264612872)
    (('the', 'parties'), 1.301824627341741)
    ((',', 'yet'), 1.301781831278209)
    (('be', 'into'), 1.3001117387171277)
    (('this', 'act'), 1.299671350761578)
    (('the', 'pass'), 1.2985080855572804)
    (('the', 'section'), 1.2985080855572804)
    (('the', 'st.'), 1.2985080855572804)
    (('the', 'value'), 1.2985080855572804)
    (('in', 'answer'), 1.2979646424802382)
    (('in', 'county'), 1.2979646424802382)
    (('person', 'or'), 1.2978939134673109)
    (('no', 'can'), 1.297752465843205)
    (('the', 'states'), 1.2969458817869963)
    (('proceedings', 'to'), 1.2961032674458934)
    (('to', 'debt'), 1.2961032674458899)
    (('record', 'court'), 1.2952472844841978)
    (('law', 'if'), 1.2923575106904384)
    (('be', 'as'), 1.2922065406317813)
    ((',', '12'), 1.2917977427055867)
    ((',', 'find'), 1.2917977427055867)
    ((',', 'whose'), 1.2917977427055867)
    ((',', '†'), 1.2917977427055867)
    (('vacant', ','), 1.2917977427055867)
    ((',', 'id'), 1.2917977427055831)
    (('the', 'penalty'), 1.2915886716174896)
    (('is', 'within'), 1.290838465911289)
    (('the', 'bill'), 1.2904886372211593)
    (('being', 'a'), 1.2897362320718138)
    (('for', 'county'), 1.2897127894348763)
    (('them', 'by'), 1.2891502803953347)
    (('contract', 'and'), 1.2883803821478494)
    (('bar', '.'), 1.2869726270899644)
    (('replication', '.'), 1.2869726270899644)
    (('.', 'having'), 1.2869726270899626)
    (('to', 'him'), 1.2860496027819686)
    (('been', 'with'), 1.2857359899183862)
    (('declaration', 'on'), 1.2848449820638166)
    (('held', 'in'), 1.2841588429552075)
    (('of', 'contract'), 1.2835772359006157)
    (('of', 'money'), 1.2835772359006157)
    (('are', 'not'), 1.2833971728661346)
    (('and', 'into'), 1.283327154421567)
    (('1825', 'and'), 1.2833271544215652)
    (('constitution', 'was'), 1.2831171194748094)
    (('made', 'on'), 1.2787832292766161)
    (('of', 'parties'), 1.2783500742370677)
    (('equity', 'that'), 1.2777366604669265)
    (('by', 'who'), 1.2775623061201244)
    (('the', 'shows'), 1.2764817792272822)
    (('the', 'third'), 1.2764817792272822)
    (('assume', 'the'), 1.2764817792272805)
    (('men', 'the'), 1.2764817792272805)
    (('the', 'back'), 1.2764817792272805)
    (('the', 'country'), 1.2764817792272805)
    (('the', 'fine'), 1.2764817792272805)
    (('the', 'gallatin'), 1.2764817792272805)
    (('the', 'instructions'), 1.2764817792272805)
    (('the', 'omission'), 1.2764817792272805)
    (('the', 'prayer'), 1.2764817792272805)
    (('violation', 'the'), 1.2764817792272805)
    (('the', 'november'), 1.2764817792272787)
    (('an', 'law'), 1.2759940644815515)
    (('note', 'which'), 1.2756955449700076)
    (('court', 'power'), 1.2755152193600168)
    (('time', 'it'), 1.275384652814754)
    (('at', 'such'), 1.275053937109199)
    (('after', 'was'), 1.274688497404231)
    (('was', 'then'), 1.274688497404231)
    (('jurisdiction', 'this'), 1.274079636858838)
    (('c.', ','), 1.2726889197578828)
    (('however', 'be'), 1.2726310022950216)
    (('officer', 'it'), 1.2715425865203578)
    (('is', 'also'), 1.271473141044357)
    (('is', 'these'), 1.271473141044357)
    (('was', 'against'), 1.2708735237200628)
    (('this', 'error'), 1.269963528386608)
    (('.', 'further'), 1.268236064508379)
    (('this', 'i'), 1.2682030705221692)
    (('means', 'of'), 1.2647182086493025)
    (('not', 'them'), 1.2646606102845492)
    (('money', '.'), 1.2640114786466512)
    (('a', 'record'), 1.2634149539158521)
    (('reversed', 'a'), 1.2634149539158521)
    (('done', 'in'), 1.2614387664551217)
    (('in', 'paper'), 1.2614387664551217)
    (('.', 'these'), 1.261093995264357)
    (('senate', ','), 1.2600888829782484)
    (('peace', 'and'), 1.2597403341039488)
    (('the', 'real'), 1.2594082658683377)
    (('court', ':'), 1.2590270965714438)
    (('said', 'justice'), 1.2585108342503553)
    (('argument', ','), 1.2575820273676719)
    (('issue', 'and'), 1.2558464179994573)
    (('that', 'might'), 1.255016583966846)
    (('from', 'his'), 1.2541668054809083)
    (('is', 'matter'), 1.252364318096653)
    (('it', 'that'), 1.2513853335199734)
    ((',', '6'), 1.2511557582082382)
    ((',', 'consequently'), 1.2511557582082382)
    ((',', 'necessarily'), 1.2511557582082382)
    (('george', ','), 1.2511557582082382)
    (('purposes', ','), 1.2511557582082382)
    (('retrospective', ','), 1.2511557582082382)
    (('served', ','), 1.2511557582082382)
    (('decision', 'this'), 1.2507156437933276)
    (('the', "''"), 1.2500095678660905)
    (('the', 'acted'), 1.2500095678660905)
    (('the', 'assumed'), 1.2500095678660905)
    (('the', 'hands'), 1.2500095678660905)
    (('the', 'king'), 1.2500095678660905)
    (('the', 'terms'), 1.2500095678660905)
    (('a', 'judgment'), 1.2495093518906337)
    (('suit', 'is'), 1.2494468347143588)
    (('by', 'plaintiff'), 1.248993153923351)
    (('maker', 'and'), 1.2468012783964504)
    (('.', 'pleaded'), 1.245152451395338)
    (('correct', '.'), 1.2451524513953363)
    (('grant', '.'), 1.2451524513953363)
    (('judge', '.'), 1.2451524513953363)
    (('to', 'award'), 1.2445729668058085)
    (('to', 'money'), 1.2445729668058085)
    (('legal', 'that'), 1.2437893285435901)
    (('in', 'possession'), 1.2435168584578609)
    (('*', 'was'), 1.2423707771314998)
    (('that', 'would'), 1.24233132433573)
    (('consideration', 'not'), 1.24157699717151)
    (('left', 'the'), 1.2399559032021656)
    (('if', 'not'), 1.2388316204326912)
    (('of', 'equity'), 1.2387230001163587)
    (('hand', 'of'), 1.238723000116357)
    (('his', '”'), 1.238723000116357)
    (('of', 'existence'), 1.238723000116357)
    (('of', 'others'), 1.238723000116357)
    (('jurisdiction', 'of'), 1.2387230001163552)
    (('was', 'upon'), 1.2371624861287387)
    (('office', 'and'), 1.2355740229731964)
    ((',', 'even'), 1.2352142143392193)
    ((',', 'seems'), 1.2352142143392193)
    (('the', 'bond'), 1.234661603532654)
    (('the', 'object'), 1.234661603532654)
    (('was', 'not'), 1.233714879398537)
    (('demurrer', 'and'), 1.2329954788714197)
    (('to', 'premises'), 1.231972930026176)
    (('it', 'did'), 1.2311848489042152)
    (('sale', 'of'), 1.2293851355370187)
    (('whole', 'of'), 1.2293851355370187)
    (('and', 'great'), 1.2288793703991896)
    (('jurors', 'and'), 1.2288793703991896)
    (('or', 'or'), 1.2284350564504045)
    (('not', 'however'), 1.2279020602744328)
    (('a', 'true'), 1.2271947660030964)
    (('if', 'is'), 1.2269075886109135)
    (('own', ','), 1.2267027144837002)
    (('the', 'present'), 1.2240143593331432)
    (('right', 'an'), 1.2239994268432923)
    ((',', '4'), 1.2229358845607798)
    (('law', 'or'), 1.2218019641179545)
    (('language', 'of'), 1.2216494867574141)
    (('of', 'hundred'), 1.2216494867574141)
    (('subsequent', 'of'), 1.2216494867574141)
    ((',', 'because'), 1.2214084148141886)
    (('government', ','), 1.2214084148141886)
    (('and', 'secretary'), 1.2212661862893128)
    (('secretary', 'and'), 1.2212661862893128)
    (('that', 'no'), 1.2200211626106423)
    (('be', 'assigned'), 1.2191917433335604)
    (('be', 'sufficient'), 1.2191917433335604)
    (('look', 'the'), 1.217588090173713)
    (('passage', 'the'), 1.217588090173713)
    (('raised', 'the'), 1.217588090173713)
    (('the', 'best'), 1.217588090173713)
    (('the', 'correctness'), 1.217588090173713)
    (('the', 'dismissed'), 1.217588090173713)
    (('the', 'end'), 1.217588090173713)
    (('the', 'sum'), 1.217588090173713)
    (('ascertain', 'the'), 1.2175880901737095)
    (('the', 'averment'), 1.2175880901737095)
    (('the', 'validity'), 1.2175880901737095)
    (('in', 'debt'), 1.217044647096671)
    (('taken', 'in'), 1.2170446470966674)
    (('action', 'of'), 1.2166660792431578)
    (('such', 'case'), 1.2166638944356407)
    (('to', 'construction'), 1.2163760749751589)
    (('aside', '.'), 1.2160061057358185)
    (('damages', '.'), 1.2160061057358185)
    (('error', 'is'), 1.2148896126779931)
    (('as', 'we'), 1.2145271768099057)
    (('estate', 'be'), 1.2139550949674138)
    (('but', 'must'), 1.2136214852906146)
    (('state', 'he'), 1.2116690334180795)
    (('to', 'more'), 1.2112143698593805)
    (('such', 'a'), 1.2104773710581647)
    (('this', 'brought'), 1.209590910407126)
    (('rights', 'of'), 1.2095766544568392)
    (('his', 'if'), 1.208792794051309)
    (('?', 'is'), 1.208737385696395)
    (('appear', 'to'), 1.2070982613871486)
    (('point', 'it'), 1.2065227946699473)
    (('only', 'be'), 1.206135590508115)
    (('.', 'again'), 1.2045104668979913)
    (('.', 'shows'), 1.2045104668979913)
    (('17', '.'), 1.2045104668979913)
    (('7', '.'), 1.2045104668979913)
    (('became', '.'), 1.2045104668979913)
    (('per', '.'), 1.2045104668979913)
    (('the', 'jury'), 1.2043319934714418)
    (('now', 'to'), 1.2037392492972323)
    (('have', 'had'), 1.2031704727346195)
    (('general', 'it'), 1.2030338085900674)
    (('being', 'by'), 1.2026002484388982)
    (('person', 'to'), 1.2009460344055505)
    (('character', 'of'), 1.2005878712295868)
    (('such', 'was'), 1.199637792142969)
    (('case', 'should'), 1.199480882762284)
    (('consideration', 'on'), 1.199348761782213)
    (('it', 'may'), 1.1993487617822112)
    ((',', 'gave'), 1.1986883383141027)
    (('death', ','), 1.1986883383141027)
    (('deceased', ','), 1.1986883383141027)
    (('degree', ','), 1.1986883383141027)
    (('hearing', ','), 1.1986883383141027)
    (('johnson', ','), 1.1986883383141027)
    (('taylor', ','), 1.1986883383141027)
    (('twenty', ','), 1.1986883383141027)
    (('the', 'affidavit'), 1.1984792672260092)
    (('the', 'appointment'), 1.1984792672260092)
    (('the', 'duty'), 1.1984792672260092)
    (('that', 'although'), 1.1975663117829445)
    (('error', '('), 1.196774818118385)
    (('’', 'law'), 1.194797981633572)
    (('.', 'filed'), 1.1945263783253708)
    (('2', '.'), 1.1945263783253672)
    (('the', 'deed'), 1.1943806990123882)
    (('a', 'sheriff'), 1.191861692890818)
    (('therefore', 'as'), 1.1914435636968665)
    (('remedy', 'in'), 1.1910494385637254)
    (('stated', 'in'), 1.1910494385637254)
    (('of', 'page'), 1.190717627205526)
    (('or', 'from'), 1.1890863889745766)
    (('any', 'which'), 1.188818094290042)
    (('of', 'dollars'), 1.188096927046388)
    (('of', 'persons'), 1.188096927046388)
    (('the', 'facts'), 1.1858792304463748)
    (('a', 'rendered'), 1.1841260441112116)
    (('a', 'well'), 1.1841260441112116)
    (('unless', 'a'), 1.1841260441112116)
    (('and', 'recover'), 1.1837914808706493)
    (('good', 'and'), 1.1837914808706493)
    (('the', 'argument'), 1.1833723748357983)
    (('note', 'as'), 1.181459475124246)
    (('a', 'against'), 1.1807354549104367)
    (('the', 'record'), 1.1804350917562445)
    (('had', 'at'), 1.1786241744261066)
    (('.', 'cranch'), 1.1780382555367996)
    (('admit', 'the'), 1.1769461056763646)
    (('issuing', 'the'), 1.1769461056763646)
    (('statement', 'the'), 1.1769461056763646)
    (('the', 'civil'), 1.1769461056763646)
    (('the', 'complainants'), 1.1769461056763646)
    (('the', 'frauds'), 1.1769461056763646)
    (('be', 'its'), 1.1761230214416756)
    (('not', 'taken'), 1.1744628013129716)
    (('proceedings', 'not'), 1.1744628013129716)
    (('court', 'do'), 1.1741381989849309)
    (('jurisdiction', 'court'), 1.1741381989849309)
    (('rendered', 'in'), 1.1739759252047826)
    (('.', 'then'), 1.1733617683273252)
    (('but', 'before'), 1.1725162819609416)
    (('which', 'before'), 1.1725162819609416)
    (('for', 'any'), 1.172266918026196)
    (('part', 'the'), 1.1717844005605862)
    (('never', 'to'), 1.1705723853620338)
    (('to', 'authorized'), 1.1705723853620338)
    (('to', 'want'), 1.1705723853620338)
    (('to', 'mandamus'), 1.170572385362032)
    (('to', 'our'), 1.170572385362032)
    (('to', 'paper'), 1.170572385362032)
    (('was', 'for'), 1.1701975487387308)
    (('it', 'only'), 1.1692190955978248)
    (('the', 'note'), 1.1669620171037423)
    (('was', 'only'), 1.1667731582373442)
    (('or', 'after'), 1.16598223796173)
    (('judgment', 'was'), 1.1641760467513045)
    (('property', 'a'), 1.1630644285833824)
    (('should', 'not'), 1.1625887010246814)
    (('common', 'in'), 1.1619030929042076)
    (('not', 'so'), 1.161406648487528)
    (('the', 'indenture'), 1.1610045618073457)
    (('the', 'mode'), 1.1610045618073457)
    (('received', '.'), 1.159422577369451)
    (('and', 'according'), 1.1593384371461113)
    (('public', 'and'), 1.1593384371461113)
    (('considered', 'that'), 1.1590921639683085)
    (('secretary', 'that'), 1.1590921639683085)
    (('that', 'secretary'), 1.1590921639683085)
    (('defendant', 'in'), 1.1571021066403908)
    (('decision', 'of'), 1.1564653179972773)
    (('by', 'verdict'), 1.1558837495318706)
    ((',', 'approved'), 1.154294218955652)
    (('approved', ','), 1.154294218955652)
    (('juror', ','), 1.154294218955652)
    (('was', 'demurrer'), 1.153834102529844)
    (('be', 'by'), 1.1534260061080168)
    (('estate', 'in'), 1.1529143096769552)
    (('the', 'promise'), 1.1524930619518265)
    (('and', 'cause'), 1.151977732194858)
    (('taken', 'to'), 1.1517133581107188)
    (('but', 'such'), 1.149732886316654)
    (('been', 'this'), 1.1493733875730179)
    (('nor', 'is'), 1.1486163932588234)
    (('common', 'and'), 1.1472656048455345)
    (('the', 'injunction'), 1.147198762282315)
    (('it', 'i'), 1.1466513386925499)
    (('united', ','), 1.1465998271656517)
    (('that', 'said'), 1.1442536549926743)
    ((',', '7'), 1.144240554291727)
    (('philips', ','), 1.144240554291727)
    (('is', '“'), 1.1427398269221563)
    (('action', 'be'), 1.1425704617306494)
    (('reynolds', 'a'), 1.1423058684165834)
    (('he', 'he'), 1.1422882305875994)
    (('act', 'as'), 1.1422000446770184)
    (('the', 'making'), 1.1409668085707985)
    (('authority', 'in'), 1.1404233654937581)
    (('amendment', 'of'), 1.1391873265654429)
    (('averment', 'of'), 1.1391873265654429)
    (('of', 'trespass'), 1.139187326565441)
    ((',', 'remanded'), 1.1372877936499606)
    (('taken', 'this'), 1.1365761131089034)
    (('this', 'does'), 1.1365761131089034)
    (('i', 'to'), 1.1349484756313117)
    (('judgment', 'for'), 1.1342458406862548)
    (('issue', '.'), 1.131941840947345)
    (('if', 'be'), 1.1315574690082286)
    (('by', 'execution'), 1.1313415652690768)
    (('objection', 'to'), 1.131044021175395)
    (('only', 'to'), 1.131044021175395)
    (('to', 'defense'), 1.131044021175395)
    (('that', 'sheriff'), 1.130895272137657)
    (('not', 'made'), 1.128366386723517)
    ((',', 'affecting'), 1.1282990104227064)
    ((',', 'among'), 1.1282990104227064)
    ((',', 'appearing'), 1.1282990104227064)
    ((',', 'deliver'), 1.1282990104227064)
    ((',', 'generally'), 1.1282990104227064)
    ((',', 'noticed'), 1.1282990104227064)
    ((',', 'sisters'), 1.1282990104227064)
    ((',', 'thought'), 1.1282990104227064)
    ((',', 'together'), 1.1282990104227064)
    (('die', ','), 1.1282990104227064)
    (('noticed', ','), 1.1282990104227064)
    (('perhaps', ','), 1.1282990104227064)
    (('produced', ','), 1.1282990104227064)
    (('remark', ','), 1.1282990104227064)
    (('specified', ','), 1.1282990104227064)
    (('surplusage', ','), 1.1282990104227064)
    (('error', 'in'), 1.128039641037926)
    (('but', 'not'), 1.127541753925481)
    (('so', 'by'), 1.1270026295447408)
    (('been', 'on'), 1.1267801358315666)
    (('.', 'view'), 1.1265079548967183)
    (('it', 'received'), 1.1256917200569028)
    (('by', 'his'), 1.1249518899799789)
    (('with', 'this'), 1.124913195583325)
    (('with', 'which'), 1.1246877568703262)
    (('overruled', 'the'), 1.1244786857822326)
    (('recess', 'the'), 1.1244786857822326)
    (('averred', 'the'), 1.124478685782229)
    (('come', 'the'), 1.124478685782229)
    (('departure', 'the'), 1.124478685782229)
    (('examined', 'the'), 1.124478685782229)
    (('fully', 'the'), 1.124478685782229)
    (('intention', 'the'), 1.124478685782229)
    (('substantially', 'the'), 1.124478685782229)
    (('the', 'administration'), 1.124478685782229)
    (('the', 'body'), 1.124478685782229)
    (('the', 'contained'), 1.124478685782229)
    (('the', 'custody'), 1.124478685782229)
    (('the', 'improvement'), 1.124478685782229)
    (('the', 'levy'), 1.124478685782229)
    (('the', 'limitations'), 1.124478685782229)
    (('the', 'prayed'), 1.124478685782229)
    (('the', 'session'), 1.124478685782229)
    (('that', 'where'), 1.124262179805207)
    (('”', 'he'), 1.1232457826964204)
    (('deed', 'in'), 1.1218874140563315)
    (('that', 'appear'), 1.1216174585496468)
    (('note', 'been'), 1.1212419522901378)
    (('such', 'are'), 1.1211637341198823)
    (('such', 'no'), 1.1211637341198823)
    (('it', 'could'), 1.1195394930753082)
    (('party', 'a'), 1.118941875351073)
    (('“', 'have'), 1.1186369625229524)
    (('must', '('), 1.1179381372353063)
    (('be', 'out'), 1.1179084074963779)
    (('be', 'what'), 1.1179084074963779)
    (('and', 'demurrer'), 1.1175182614514831)
    (('jurisdiction', 'and'), 1.1175182614514831)
    (('party', 'to'), 1.11746104890247)
    (('to', 'execution'), 1.11746104890247)
    (('a', 'declaration'), 1.1167707763094477)
    (('in', 'what'), 1.1157613112594866)
    (('is', 'error'), 1.1153539391270755)
    (('are', 'law'), 1.1145306417874359)
    (('(', 'we'), 1.112943227314915)
    (('sentence', 'to'), 1.1116786963084628)
    (('shown', 'to'), 1.1116786963084628)
    (('agent', ','), 1.1112254970637636)
    (('chancery', ','), 1.1112254970637636)
    ((';', 'he'), 1.109866752895222)
    (('below', 'have'), 1.1098667528952202)
    (('decision', 'in'), 1.1087917564446457)
    (('before', 'the'), 1.1081768734531323)
    (('consideration', 'is'), 1.1079744087614785)
    (('of', 'estate'), 1.1074784668381064)
    (('united', 'and'), 1.1072499259976567)
    (('estate', 'to'), 1.1064420479423163)
    (('any', 'state'), 1.1047538295015684)
    (('it', 'ought'), 1.104272736421997)
    (('states', ','), 1.1040514641760275)
    (('but', 'can'), 1.103929196703529)
    ((',', 'johns'), 1.1035311052737669)
    (('the', 'provisions'), 1.1034170702544017)
    (('the', 'general'), 1.1034170702543982)
    (('action', 'was'), 1.103208029459875)
    (('a', 'equity'), 1.1016638839192385)
    (('held', 'a'), 1.1016638839192385)
    ((')', 'we'), 1.1014169739338868)
    (('declaration', 'that'), 1.100198474914741)
    (('evidence', 'that'), 1.100198474914741)
    (('which', 'error'), 1.099813088231297)
    (('by', 'suit'), 1.0994167978728413)
    ((',', 'express'), 1.0991526647631886)
    (('thereof', ','), 1.0991526647631886)
    (('void', ','), 1.0991526647631886)
    (('the', 'decisions'), 1.0989435936750915)
    (('it', 'been'), 1.0982109836347966)
    (('so', 'that'), 1.0980306382320286)
    (('whether', 'not'), 1.0978415197100588)
    ((')', 'v.'), 1.0973758087120231)
    (('.', 'there'), 1.0971764227866672)
    (('of', 'real'), 1.0961186046735563)
    (('justice', 'has'), 1.0938143476098556)
    (('.', 'reynolds'), 1.0931493579502884)
    (('court', 'must'), 1.0911854096296203)
    (('such', 'or'), 1.0909315327004698)
    (('this', 'there'), 1.0904796985194487)
    (('on', 'note'), 1.0898289996586747)
    (('”', 'was'), 1.0897037651101265)
    (('not', 'do'), 1.0895739037264587)
    (('constitution', 'it'), 1.089165844031788)
    (('good', '.'), 1.0890332494780566)
    (('from', 'the'), 1.0873444393718437)
    (('law', ';'), 1.0865162656178384)
    (('common', 'that'), 1.085091582524532)
    (('only', 'is'), 1.0843875884438603)
    (('jury', 'not'), 1.0836973373897898)
    (('corporation', ','), 1.0832111208941697)
    (('says', ','), 1.0832111208941697)
    (('valid', ','), 1.0832111208941697)
    (('but', 'was'), 1.0828675811757016)
    ((';', 'are'), 1.082739574872356)
    (('is', 'one'), 1.0815021974002867)
    (('be', 'so'), 1.0806047084242572)
    (('made', 'case'), 1.0803696708746084)
    (('time', 'of'), 1.080293637511872)
    (('happen', 'the'), 1.0800845664237784)
    (('objected', 'the'), 1.0800845664237784)
    (('the', 'correctly'), 1.0800845664237784)
    (('the', 'entry'), 1.0800845664237784)
    (('the', 'territory'), 1.0800845664237784)
    (('brought', 'a'), 1.0778171419648714)
    ((',', '2'), 1.077672937352741)
    (('and', 'commission'), 1.0768762769541382)
    (('and', 'third'), 1.0768762769541382)
    (('.', 'point'), 1.0767549196996207)
    (('not', 'been'), 1.0758989668293815)
    (('is', 'legal'), 1.0740270768381404)
    (('note', 'for'), 1.0732778233948288)
    (('any', 'such'), 1.0711230516202885)
    (('the', 'referred'), 1.0700309017598535)
    (('is', 'statute'), 1.0698392798747065)
    (('not', 'against'), 1.0691098011667428)
    (('declaration', 'is'), 1.067939746959226)
    (('legal', 'to'), 1.0662357255472976)
    (('the', 'justices'), 1.0655849967286635)
    (('is', 'into'), 1.06288651923294)
    (('writ', 'is'), 1.06288651923294)
    (('principle', 'a'), 1.0621355197326015)
    ((',', 'and'), 1.0616095203008271)
    (('evidence', 'of'), 1.0611848145641716)
    (('mere', 'of'), 1.061184814564168)
    (('the', 'term'), 1.060348348362517)
    (('of', 'act'), 1.0595701585020727)
    (('can', 'said'), 1.058486225942362)
    (('i', 'be'), 1.0580369515189822)
    (('defendants', 'in'), 1.0575664330894732)
    (('motion', 'in'), 1.0575664330894732)
    ((',', 'damages'), 1.057332489068564)
    (('judgment', 'upon'), 1.056820868741088)
    (('court', 'first'), 1.055493702486313)
    (('presented', 'to'), 1.055095167942099)
    (('to', 'answer'), 1.0550951679420955)
    (('to', 'prisoner'), 1.0550951679420955)
    (('before', 'or'), 1.05482123929119)
    (('clearly', 'of'), 1.0542984289789281)
    (('the', 'offered'), 1.0540893578908346)
    (('dissolved', 'the'), 1.0540893578908328)
    (('incumbent', 'the'), 1.0540893578908328)
    (('pleading', 'the'), 1.0540893578908328)
    (('settle', 'the'), 1.0540893578908328)
    (('sisters', 'the'), 1.0540893578908328)
    (('the', 'conduct'), 1.0540893578908328)
    (('the', 'controversy'), 1.0540893578908328)
    (('the', 'die'), 1.0540893578908328)
    (('the', 'evident'), 1.0540893578908328)
    (('the', 'mortgage'), 1.0540893578908328)
    (('the', 'offer'), 1.0540893578908328)
    (('the', 'report'), 1.0540893578908328)
    (('warranted', 'the'), 1.0540893578908328)
    (('2.', 'the'), 1.054089357890831)
    (('forth', 'the'), 1.054089357890831)
    (('merits', 'the'), 1.054089357890831)
    (('the', 'article'), 1.054089357890831)
    (('the', 'cited'), 1.054089357890831)
    (('2', ','), 1.0534253911060603)
    (('but', 'had'), 1.05330312363356)
    (('al', ','), 1.0530108831184677)
    (('committed', '.'), 1.0525073734529418)
    (('overruled', '.'), 1.0525073734529418)
    (('.', 'plea'), 1.05250737345294)
    (('.', 'section'), 1.05250737345294)
    (('.', 'states'), 1.05250737345294)
    (('.', 'third'), 1.05250737345294)
    (('remedy', '.'), 1.05250737345294)
    (('would', 'to'), 1.0521776845598048)
    (('that', 'defendant'), 1.0511442506011939)
    (('assigned', 'and'), 1.05040406559295)
    (('following', 'that'), 1.0494676727938117)
    (('that', 'will'), 1.04946767279381)
    (('as', 'there'), 1.048485609854822)
    ((';', 'can'), 1.0473456683371616)
    (('and', 'decree'), 1.0471289335600886)
    (('the', 'award'), 1.046476173780956)
    (('of', 'those'), 1.0460779221739607)
    (('of', 'wife'), 1.0460779221739607)
    (('day', ','), 1.045442078018425)
    (('subject', 'to'), 1.0450415032781741)
    (('the', 'sale'), 1.0447514933114945)
    (('been', 'plaintiff'), 1.0437631996124193)
    (('only', 'for'), 1.0437335477808105)
    (('is', 'form'), 1.0408602129029418)
    (('that', 'below'), 1.0406974631660795)
    (('for', 'a'), 1.0400248181268523)
    (('the', 'fact'), 1.039945175570562)
    (('as', 'other'), 1.039440470251817)
    ((';', 'as'), 1.0394404702518152)
    (('question', 'be'), 1.039213077227327)
    (('in', 'second'), 1.0390463451186776)
    (('upon', 'his'), 1.0382584177146619)
    (('and', 'so'), 1.0373479127675012)
    (('subsequent', 'the'), 1.03701584453189)
    (('the', 'causes'), 1.03701584453189)
    (('the', 'year'), 1.03701584453189)
    (('had', 'an'), 1.0341943006858578)
    (('much', 'to'), 1.0330688616120973)
    (('under', 'it'), 1.0328976243850434)
    (('on', 'his'), 1.0326988923727605)
    (('the', 'question'), 1.03255619634119)
    (('not', 'its'), 1.031858405870171)
    (('statute', 'this'), 1.0318037911918445)
    (('his', 'from'), 1.0317743841444607)
    (('sheriff', 'is'), 1.0315223483236053)
    (('the', 'objection'), 1.0305025375732164)
    (('.', 'justice'), 1.0301395604244838)
    (('and', 'whole'), 1.029570562175783)
    ((',', '31st'), 1.0287633368717906)
    ((',', '8'), 1.0287633368717906)
    ((',', 'apply'), 1.0287633368717906)
    ((',', 'believed'), 1.0287633368717906)
    ((',', 'john'), 1.0287633368717906)
    ((',', 'less'), 1.0287633368717906)
    ((',', 'purposes'), 1.0287633368717906)
    ((',', 'six'), 1.0287633368717906)
    ((',', 'surely'), 1.0287633368717906)
    ((',', 'thereby'), 1.0287633368717906)
    (('bonds', ','), 1.0287633368717906)
    (('brothers', ','), 1.0287633368717906)
    (('clair', ','), 1.0287633368717906)
    (('complaint', ','), 1.0287633368717906)
    (('demurred', ','), 1.0287633368717906)
    (('forcible', ','), 1.0287633368717906)
    (('frauds', ','), 1.0287633368717906)
    (('gatewood', ','), 1.0287633368717906)
    (('little', ','), 1.0287633368717906)
    (('positive', ','), 1.0287633368717906)
    (('private', ','), 1.0287633368717906)
    (('written', ','), 1.0287633368717906)
    (('the', 'power'), 1.0273100267752326)
    (('for', 'cause'), 1.027118833929915)
    (('case', 'would'), 1.0269304119131473)
    (('be', 'any'), 1.026546665391166)
    (('that', 'time'), 1.0261978934709646)
    (('can', 'any'), 1.0253193620071634)
    (('contrary', 'the'), 1.024943012231315)
    (('correctness', 'the'), 1.024943012231315)
    (('parol', 'the'), 1.024943012231315)
    (('place', 'the'), 1.024943012231315)
    (('repeal', 'the'), 1.024943012231315)
    (('the', 'place'), 1.024943012231315)
    (('the', 'reasons'), 1.024943012231315)
    (('the', 'rules'), 1.024943012231315)
    (('the', 'services'), 1.024943012231315)
    (('the', 'showing'), 1.024943012231315)
    (('in', 'same'), 1.0243995691542729)
    (('prove', 'of'), 1.023710109145508)
    (('bank', 'of'), 1.0237101091455045)
    (('a', 'matter'), 1.0236613719179655)
    (('an', 'all'), 1.021810576230024)
    (('the', 'day'), 1.0201420259674947)
    (('it', 'can'), 1.0187765161403917)
    (('county', 'court'), 1.0180189970676494)
    (('.', 'i'), 1.0168834637222197)
    (('following', '.'), 1.0168834637222197)
    (('which', 'we'), 1.015981497427827)
    (('as', 'as'), 1.0155937282974463)
    (('should', 'as'), 1.0155937282974463)
    (('that', 'bond'), 1.014702254633134)
    (('that', 'equity'), 1.014702254633134)
    (('that', 'pay'), 1.014702254633134)
    (('proper', 'in'), 1.0135112530115364)
    (('show', 'in'), 1.0135112530115364)
    (('legislature', 'to'), 1.0130311083755537)
    (('.', 'principle'), 1.012979009266303)
    (('peace', '.'), 1.012979009266303)
    (('of', 'record'), 1.0114317793670686)
    (('do', 'this'), 1.0110452310250455)
    (('judgment', 'at'), 1.010773501221653)
    (('and', 'same'), 1.0097620810956016)
    (('necessary', 'be'), 1.009596596461229)
    (('3.', 'the'), 1.009001468362296)
    (('the', 'circumstances'), 1.009001468362296)
    (('the', 'missouri'), 1.009001468362296)
    (('property', 'of'), 1.0079427932871887)
    (('on', 'this'), 1.0054857056831459)
    (('if', 'a'), 1.0052593999661745)
    (('have', 'with'), 1.0050944309781578)
    (('filed', 'the'), 1.0034632848208638)
    (('a', 'question'), 1.0021282103683262)
    (('authority', '.'), 1.0018813003829727)
    (('dollars', '.'), 1.0018813003829727)
    (('true', 'of'), 1.0016838028155064)
    (('case', 'was'), 1.0009352033802053)
    (('give', 'to'), 1.0006473839197199)
    (('to', 'possession'), 1.0006473839197199)
    (('on', 'question'), 1.000277414680383)
    (('very', 'in'), 0.9984043606213291)
    (('if', 'at'), 0.9976861802613755)
    (('&', ','), 0.9970544771444558)
    (('such', 'judgment'), 0.9965760982552361)
    (('court', 'proceedings'), 0.9959926907376513)
    (('no', 'or'), 0.9956328520360245)
    (('had', 'such'), 0.9945017700173757)
    (('or', 'whether'), 0.9945017700173757)
    (('defendant', 'was'), 0.994355888522648)
    (('an', 'made'), 0.9943298398079161)
    (('who', 'is'), 0.9939391655154495)
    (('.', 'contended'), 0.9936136843993708)
    (('in', 'state'), 0.9929499305292246)
    (('require', 'the'), 0.9920283897585804)
    (('the', 'money'), 0.9920283897585804)
    (('give', 'in'), 0.9919780914618954)
    (('by', 'court'), 0.9916274981762498)
    (('him', 'for'), 0.990152507575969)
    (('were', 'not'), 0.9900382301755464)
    (('which', 'not'), 0.9900382301755464)
    (('and', 'no'), 0.9894134357037991)
    (('to', 'what'), 0.988369054141284)
    (('does', 'that'), 0.988230043271944)
    (('of', 'duties'), 0.9871842331203915)
    (('the', 'indictment'), 0.986975162032298)
    (('been', 'at'), 0.9867566201031153)
    (('were', 'to'), 0.986147814224605)
    (('been', 'from'), 0.985677969555006)
    (('statute', 'not'), 0.9848015818093963)
    (('and', 'very'), 0.9837668725626578)
    (('are', 'to'), 0.9835738705607788)
    (('that', 'after'), 0.9835535560624677)
    (('aside', 'the'), 0.9831228365366904)
    (('they', 'to'), 0.9829453821862622)
    (('.', 'john'), 0.9821180455615437)
    (('the', 'land'), 0.9800887764470545)
    (('s', ','), 0.979243201027991)
    (('paper', 'to'), 0.977927307419634)
    (('to', 'done'), 0.977927307419634)
    (('is', 'by'), 0.9776013312879712)
    (('rep.', '.'), 0.977219246148703)
    (('term', 'is'), 0.9767298754832261)
    ((',', 'edward'), 0.976295916977655)
    (('evidence', 'was'), 0.976295916977655)
    (('intestate', ','), 0.976295916977655)
    (('others', ','), 0.976295916977655)
    (('which', 'plaintiff'), 0.9758243709558432)
    (('consideration', 'of'), 0.9756885942825626)
    (('and', 'then'), 0.9753382504920758)
    (('and', 'his'), 0.9749966629349238)
    (('principle', 'of'), 0.9741280802949461)
    (('bank', ','), 0.9721798085054232)
    (('.', 'al.'), 0.9715873780693727)
    (('against', 'and'), 0.9715232768079112)
    (('court', 'could'), 0.9712247855887064)
    ((';', 'that'), 0.9696143651045972)
    (('that', 'received'), 0.9696143651045954)
    (('that', 'writ'), 0.9696143651045954)
    (('only', 'in'), 0.9695232406341994)
    (('great', 'of'), 0.9692623251231289)
    (('of', 'another'), 0.9692623251231289)
    (('testimony', 'of'), 0.9692623251231289)
    (('to', 'inquiry'), 0.9689385241923834)
    (('on', 'the'), 0.9683594838649476)
    (('with', 'and'), 0.9683518201759718)
    (('it', 'whether'), 0.9681504430704244)
    (('in', 'its'), 0.9675250477373574)
    (('form', 'of'), 0.9651579267903934)
    (('rendered', '.'), 0.9650445322026009)
    (('to', 'due'), 0.9641215078946068)
    (('on', 'before'), 0.9639296601495246)
    (('under', 'of'), 0.9633374911660262)
    ((',', '22d'), 0.9616491410132539)
    (('a.', ','), 0.9616491410132539)
    (('assumed', ','), 0.9616491410132539)
    (('detainer', ','), 0.9616491410132539)
    (('giving', ','), 0.9616491410132539)
    (('submission', ','), 0.9616491410132539)
    (('defense', 'to'), 0.9611190197330828)
    (('jurisdiction', 'it'), 0.9598828270868225)
    (('than', 'a'), 0.9587059300771976)
    (('established', 'of'), 0.9586150809236216)
    (('year', 'of'), 0.9586150809236216)
    (('and', 'has'), 0.9584028070742825)
    (('to', 'their'), 0.957578662027835)
    (('justice', 'been'), 0.9568551343892544)
    (('second', 'is'), 0.9559713153164271)
    (('the', 'suit'), 0.9545536843399205)
    (('1821', 'the'), 0.9545536843399169)
    (('4.', 'the'), 0.9545536843399169)
    (('averment', 'the'), 0.9545536843399169)
    (('contained', 'the'), 0.9545536843399169)
    (('discharge', 'the'), 0.9545536843399169)
    (('existed', 'the'), 0.9545536843399169)
    (('fourth', 'the'), 0.9545536843399169)
    (('injury', 'the'), 0.9545536843399169)
    (('inquire', 'the'), 0.9545536843399169)
    (('meaning', 'the'), 0.9545536843399169)
    (('presence', 'the'), 0.9545536843399169)
    (('setting', 'the'), 0.9545536843399169)
    (('shows', 'the'), 0.9545536843399169)
    (('sustained', 'the'), 0.9545536843399169)
    (('the', 'complaint'), 0.9545536843399169)
    (('the', 'contingency'), 0.9545536843399169)
    (('the', 'george'), 0.9545536843399169)
    (('the', 'illinois'), 0.9545536843399169)
    (('the', 'madison'), 0.9545536843399169)
    (('the', 'pleadings'), 0.9545536843399169)
    (('the', 'principles'), 0.9545536843399169)
    (('the', 'statement'), 0.9545536843399169)
    (('the', 'testimony'), 0.9545536843399169)
    (('vested', 'the'), 0.9545536843399169)
    (('in', 'supreme'), 0.9540102412628766)
    (('in', 'his'), 0.9540102412628748)
    (('shall', 'not'), 0.9537746861887726)
    ((')', '.'), 0.9535182788215728)
    (('its', 'and'), 0.9528875596786861)
    (('as', 'are'), 0.9519776290014761)
    (('as', 'no'), 0.9519776290014761)
    (('affidavit', ','), 0.9507608248705175)
    (('that', 'not'), 0.9504420558297504)
    (('a', 'who'), 0.9496607904741907)
    (('a', 'interest'), 0.9496607904741889)
    (('it', '?'), 0.949614491632996)
    (('therefore', 'not'), 0.9493962456781979)
    (('right', 'of'), 0.9483649842298654)
    (('demurrer', 'to'), 0.9481799640255879)
    (('present', 'to'), 0.9481799640255844)
    (('.', 'now'), 0.9481707136382056)
    (('under', 'is'), 0.9480661172310825)
    (('deed', 'is'), 0.9457029798626007)
    (('.', 'second'), 0.9455921695364289)
    ((',', 'whether'), 0.9455694013453915)
    (('which', 'judgment'), 0.9443461394826755)
    (('right', 'in'), 0.9437601445691186)
    (('what', 'it'), 0.9434883888361547)
    (('court', 'not'), 0.9428902818187872)
    (('however', 'that'), 0.9421336286824875)
    (('opinion', 'was'), 0.941289467149744)
    (('has', 'such'), 0.9395934289149857)
    (('such', 'has'), 0.9395934289149857)
    (('or', 'such'), 0.9389284392554202)
    (('on', '”'), 0.9388212115589951)
    (('the', 'courts'), 0.9380655615513511)
    (('to', 'person'), 0.9379116285717579)
    (('governor', 'of'), 0.9375534653957907)
    (('recover', '.'), 0.937030156033007)
    (('into', '.'), 0.9370301560330034)
    (('cause', 'which'), 0.9367548062772517)
    (('record', 'this'), 0.9357571037208103)
    (('$', ','), 0.9356539324803101)
    ((',', 'entirely'), 0.9356539324803101)
    ((',', 'least'), 0.9356539324803101)
    (('averred', ','), 0.9356539324803101)
    (('changed', ','), 0.9356539324803101)
    (('collins', ','), 0.9356539324803101)
    (('devise', ','), 0.9356539324803101)
    (('doing', ','), 0.9356539324803101)
    (('kentucky', ','), 0.9356539324803101)
    (('of', 'matter'), 0.9356539324803101)
    (('therein', ','), 0.9356539324803101)
    (('thereto', ','), 0.9356539324803101)
    ((',', 'or'), 0.9343236465023423)
    (('and', 'issue'), 0.9339183231120955)
    (('law', 'is'), 0.9337413679329636)
    (('below', 'to'), 0.9335331880611868)
    (('to', 'proceedings'), 0.9335331880611832)
    (('to', 'true'), 0.9335331880611832)
    (('that', 'given'), 0.9330884890794842)
    (('of', 'these'), 0.9327364490980159)
    (('an', 'for'), 0.9312588185224016)
    (('first', 'is'), 0.9304362232092913)
    (('not', 'to'), 0.9302134174961374)
    (('smith', ','), 0.9292276633208765)
    (('into', 'the'), 0.926539308170323)
    (('section', 'the'), 0.926539308170323)
    (('it', 'said'), 0.9259354951634862)
    (('and', 'were'), 0.9248731835090922)
    (('and', 'manner'), 0.9248731835090869)
    (('interest', 'and'), 0.9248731835090869)
    (('clearly', '.'), 0.9232243565079763)
    (('demurrer', '.'), 0.9232243565079763)
    (('do', '.'), 0.9232243565079763)
    (('provisions', 'the'), 0.9228448246125822)
    (('money', 'to'), 0.9226448719184468)
    (('by', 'him'), 0.9214184958948461)
    (('verdict', 'a'), 0.9210916382774172)
    (('this', 'will'), 0.9202797671018637)
    (('the', 'act'), 0.9202605067860397)
    (('it', 'due'), 0.9192408425894776)
    (('when', 'a'), 0.9184420598634695)
    (('.', 'action'), 0.9169924027964633)
    (('sheriff', 'to'), 0.9168157931162497)
    (('to', 'laws'), 0.9168157931162497)
    (('judge', 'of'), 0.9167949052289934)
    (('notes', 'of'), 0.9167949052289934)
    (('at', 'a'), 0.9165524786622505)
    (('the', 'contract'), 0.916079536525281)
    (('in', 'which'), 0.9152631253210828)
    (('that', 'rule'), 0.9151665810822216)
    (('and', 'appears'), 0.9146048480552587)
    (('said', 'and'), 0.9136459280858347)
    (('said', 'was'), 0.9134358931390807)
    ((',', 'deemed'), 0.9132861194518576)
    ((',', 'ex'), 0.9132861194518576)
    ((',', 'says'), 0.9132861194518576)
    (('constitutional', ','), 0.9132861194518576)
    (('pleas', ','), 0.9132861194518576)
    (('replevin', ','), 0.9132861194518576)
    (('st.', ','), 0.9132861194518576)
    (('vide', ','), 0.9132861194518576)
    (('by', '”'), 0.912429712667592)
    (('this', 'no'), 0.9115095574741297)
    (('the', 'points'), 0.9114849624480321)
    (('against', 'for'), 0.9107947159626839)
    (('the', 'state'), 0.9096278134583997)
    (('“', 'is'), 0.9082745732851336)
    (('him', 'to'), 0.9075379795282394)
    (('up', 'to'), 0.9075379795282394)
    (('used', 'to'), 0.9075379795282394)
    (('opinion', '.'), 0.9070769339313784)
    (('action', 'is'), 0.9053452422464616)
    (('in', 'fact'), 0.9038867618370379)
    (('lands', ','), 0.9032324547879327)
    (('court', 'have'), 0.9025417796477129)
    (('of', 'suit'), 0.9021481292645923)
    (('than', 'to'), 0.9020835494361314)
    (('to', 'than'), 0.9020835494361314)
    (('that', 'when'), 0.9018697584687594)
    ((',', 'who'), 0.9006069853811098)
    (('which', 'said'), 0.9000568633378805)
    (('that', 'bill'), 0.8974645793487603)
    (('or', 'act'), 0.896348650516579)
    (('it', 'should'), 0.8953941006351087)
    (('assigned', 'in'), 0.8951165522093092)
    (('return', ','), 0.893833756785682)
    (('.', '’'), 0.8933087786036857)
    (('was', 'on'), 0.8930175219458683)
    (('state', 'are'), 0.8917601061673697)
    (('state', 'no'), 0.8917601061673697)
    (('whether', 'the'), 0.8909885555624513)
    (('to', 'public'), 0.8904644661692949)
    (('is', 'that'), 0.8902126537499377)
    (('in', 'general'), 0.8898799038431591)
    ((',', 'think'), 0.8893602802063754)
    (('bill', 'and'), 0.8892492737783648)
    (('upon', 'was'), 0.8892391827084332)
    (('then', 'for'), 0.8886144811139065)
    (('instrument', 'of'), 0.8876485595694774)
    (('of', 'instrument'), 0.8876485595694774)
    (('of', 'justices'), 0.8876485595694774)
    (('of', 'nature'), 0.8876485595694774)
    (('came', 'the'), 0.8874394884813803)
    (('complete', 'the'), 0.8874394884813803)
    (('copy', 'the'), 0.8874394884813803)
    (('leave', 'the'), 0.8874394884813803)
    (('the', '22d'), 0.8874394884813803)
    (('the', 'count'), 0.8874394884813803)
    (('the', 'distinction'), 0.8874394884813803)
    (('the', 'establishing'), 0.8874394884813803)
    (('urged', 'the'), 0.8874394884813803)
    (('been', 'in'), 0.8871552665066247)
    (('below', 'for'), 0.8868646991639473)
    (('that', 'case'), 0.8866780241225243)
    (('bill', 'by'), 0.8857945861641259)
    (('as', 'a'), 0.885172064022477)
    (('of', 'defendants'), 0.8828475733056571)
    (('.', 'rule'), 0.8825823720106278)
    (('.', 'two'), 0.8825823720106278)
    (('appeal', '.'), 0.8825823720106278)
    (('error', '.'), 0.8825823720106278)
    (('therefore', '.'), 0.8825823720106278)
    (('debt', 'to'), 0.8810657681670477)
    (('is', 'point'), 0.8806831880121901)
    (('which', 'law'), 0.8800653881504132)
    (('that', 'decision'), 0.8799771526199187)
    (('and', 'defendant'), 0.8794302127479199)
    (('the', 'verdict'), 0.8783180985128318)
    (('which', 'at'), 0.8781758069491943)
    (('general', 'is'), 0.8771942019323085)
    ((',', 'contrary'), 0.876760243426741)
    ((',', 'look'), 0.876760243426741)
    ((',', 'put'), 0.876760243426741)
    ((',', 'rules'), 0.876760243426741)
    ((',', 'thompson'), 0.876760243426741)
    (('clear', ','), 0.876760243426741)
    (('heirs', ','), 0.876760243426741)
    (('hubbard', ','), 0.876760243426741)
    (('thus', ','), 0.876760243426741)
    (('at', 'he'), 0.8767338327752885)
    (('he', 'at'), 0.8767338327752885)
    (('he', 'is'), 0.876473395002062)
    (('constable', 'of'), 0.8761529207316485)
    (('grounds', 'of'), 0.8761529207316485)
    (('of', 'pleadings'), 0.8761529207316485)
    (('can', 'as'), 0.8759417379689367)
    ((')', 'action'), 0.875704194305829)
    (('only', 'that'), 0.875638216895581)
    (('that', 'only'), 0.875638216895581)
    (('after', 'the'), 0.8753834709063071)
    (('general', 'and'), 0.8752424157844878)
    (('does', 'this'), 0.8735417072751108)
    (('verdict', '.'), 0.8719351278111205)
    (('we', 'it'), 0.8719351278111205)
    (('declaration', 'not'), 0.8713937336769249)
    (('not', 'evidence'), 0.8713937336769249)
    (('whether', 'by'), 0.8707924228248771)
    (('on', 'an'), 0.8698074827849744)
    (('as', 'to'), 0.8691916678787557)
    (('land', 'in'), 0.8691213436763654)
    (('decree', 'in'), 0.8691213436763618)
    (('in', 'bond'), 0.8691213436763618)
    (('may', 'in'), 0.8691213436763618)
    (('nor', 'in'), 0.8691213436763618)
    (('not', 'having'), 0.8683596735872925)
    (('verdict', 'not'), 0.8683596735872925)
    (('by', 'party'), 0.8683071594352842)
    (('the', 'point'), 0.8663265013281816)
    (('of', 'making'), 0.8661688321590262)
    (('law', 'not'), 0.8657100951733447)
    (('the', 'execution'), 0.864916471855242)
    (('the', 'party'), 0.864916471855242)
    (('.', 'bill'), 0.8648803702771701)
    (('fact', '.'), 0.8648803702771701)
    (('one', 'was'), 0.8643274853348579)
    (('this', 'we'), 0.8642038426957761)
    (('right', 'as'), 0.8631176976113526)
    (('by', 'which'), 0.8625248068412787)
    (('who', 'by'), 0.8625248068412787)
    (('or', 'before'), 0.8621761613487919)
    (('administration', 'the'), 0.8614442799484365)
    (('assignment', 'the'), 0.8614442799484365)
    (('continued', 'the'), 0.8614442799484365)
    (('default', 'the'), 0.8614442799484365)
    (('disclosed', 'the'), 0.8614442799484365)
    (('examination', 'the'), 0.8614442799484365)
    (('letter', 'the'), 0.8614442799484365)
    (('reversal', 'the'), 0.8614442799484365)
    (('session', 'the'), 0.8614442799484365)
    (('still', 'the'), 0.8614442799484365)
    (('the', 'appellate'), 0.8614442799484365)
    (('the', 'appointing'), 0.8614442799484365)
    (('the', 'kentucky'), 0.8614442799484365)
    (('the', 'least'), 0.8614442799484365)
    (('the', 'letter'), 0.8614442799484365)
    (('the', 'paymaster-general'), 0.8614442799484365)
    (('the', 'powers'), 0.8614442799484365)
    (('the', 'presume'), 0.8614442799484365)
    (('the', 'rule'), 0.8614442799484365)
    (('the', 'subject'), 0.8614442799484365)
    (('the', 'twenty'), 0.8614442799484365)
    (('thereto', 'the'), 0.8614442799484365)
    (('bill', 'not'), 0.8613049160533421)
    (('has', 'a'), 0.8611998896607744)
    (('person', 'be'), 0.8610001044594284)
    (('of', 'said'), 0.8586000270589764)
    (('no', 'such'), 0.8581293282860898)
    (('when', 'this'), 0.8575707503633261)
    ((',', 'but'), 0.8573949185598124)
    ((',', 'nor'), 0.8569969886053137)
    (('under', 'the'), 0.8567063609417751)
    (('indictment', ','), 0.8547339370967428)
    (('and', 'pay'), 0.8544838556176906)
    (('clerk', 'and'), 0.8544838556176906)
    (('hubbard', 'and'), 0.8544838556176906)
    (('land', 'and'), 0.8544838556176906)
    (('deed', 'that'), 0.8524308257342579)
    (('no', 'law'), 0.8514962359536433)
    (('may', 'by'), 0.8510291680034499)
    (('premises', '.'), 0.8508735122832913)
    (('r.', '.'), 0.8508735122832913)
    (('.', 'here'), 0.8508735122832896)
    (('decide', 'of'), 0.8496807093704568)
    (('of', 'errors'), 0.8496807093704568)
    (('of', 'means'), 0.8496807093704568)
    (('of', 'original'), 0.8496807093704568)
    (('upon', 'by'), 0.8494686540158334)
    (('this', 'could'), 0.8487738021261677)
    (('to', 'authorities'), 0.8486442904746703)
    (('the', 'right'), 0.8486202395908542)
    ((',', 'either'), 0.848191091229971)
    (('the', 'law'), 0.8481474573349814)
    (('decision', '.'), 0.8473929435483285)
    (('but', 'from'), 0.8473498130070354)
    (('from', 'which'), 0.8473498130070354)
    (('due', '.'), 0.8460564959855148)
    (('the', 'cause'), 0.845943186408249)
    ((',', 'where'), 0.8455415128160233)
    (('a', 'entered'), 0.8453241306594528)
    (('motion', 'a'), 0.8453241306594528)
    (('is', ':'), 0.844463000099438)
    (('not', 'party'), 0.8438174893245005)
    (('if', 'are'), 0.8437959772720589)
    ((';', 'said'), 0.8434733349715131)
    (('on', 'said'), 0.8434733349715131)
    (('said', ';'), 0.8434733349715131)
    (('the', 'of'), 0.8434653473047931)
    (('.', 'only'), 0.8430540078239908)
    (('can', 'but'), 0.8408947908697364)
    (('he', 'be'), 0.8406801200798313)
    (('in', 'other'), 0.8399749980168458)
    (('were', 'in'), 0.8399749980168458)
    (('so', 'not'), 0.8394785536001663)
    (('court', 'after'), 0.8393131100831255)
    (('it', 'has'), 0.8392415205242578)
    (('found', 'the'), 0.8390764669199839)
    (('special', 'the'), 0.8390764669199839)
    (('the', 'attorney'), 0.8390764669199839)
    (('the', 'failed'), 0.8390764669199839)
    (('the', 'filing'), 0.8390764669199839)
    (('the', 'provision'), 0.8390764669199839)
    (('the', 'purchase'), 0.8390764669199839)
    (('variance', 'the'), 0.8390764669199839)
    (('his', 'are'), 0.8383208472059067)
    (('being', 'this'), 0.8365474997929088)
    (('had', 'to'), 0.836153346291475)
    (('age', ','), 0.8361182589293925)
    (('contained', ','), 0.8361182589293925)
    (('fee', ','), 0.8361182589293925)
    (('goods', ','), 0.8361182589293925)
    (('immaterial', ','), 0.8361182589293925)
    (('be', 'for'), 0.8361015854820586)
    (('and', 'costs'), 0.8358681774503438)
    (('plea', 'of'), 0.8351811396753419)
    (('now', 'in'), 0.8351740117530255)
    (('the', 'time'), 0.8342594506222056)
    (('on', 'a'), 0.8341835730542542)
    (('the', 'construction'), 0.8332473881177833)
    (('made', 'a'), 0.8331750479933362)
    (('to', 'office'), 0.831770471910275)
    (('only', 'this'), 0.8313392087901477)
    (('this', 'only'), 0.8313392087901477)
    (('rule', 'be'), 0.8306264554159064)
    (('jurisdiction', '.'), 0.8301149521164959)
    (('pleaded', '.'), 0.8301149521164923)
    (('and', 'before'), 0.8299615364836228)
    (('laws', ','), 0.8294545286483839)
    (('if', 'was'), 0.8291606309036652)
    (('state', 'of'), 0.8291193868881379)
    (('whether', 'a'), 0.8286453895128254)
    (('rule', 'in'), 0.8284793591790169)
    (('were', 'and'), 0.8253375099581746)
    (('certain', ','), 0.8252299427866596)
    (('declaration', ','), 0.8252299427866596)
    (('the', 'said'), 0.8247463370090635)
    (('this', 'are'), 0.8240467162237906)
    (('the', 'most'), 0.8233091510616646)
    (('entered', 'and'), 0.8205365236943507)
    (('the', 'trial'), 0.8202525926283251)
    (('county', ','), 0.8201767150603736)
    (('but', 'that'), 0.8197470160035358)
    (('the', 'consideration'), 0.8196241042538084)
    ((',', 'until'), 0.8172592316780793)
    (('the', 'exceptions'), 0.8170501605899823)
    (('the', 'proceedings'), 0.8170501605899823)
    (('has', 'by'), 0.815884081722487)
    (('is', 'rule'), 0.8157936572681663)
    (('.', 'exceptions'), 0.8154681761520912)
    (('exceptions', '.'), 0.8154681761520912)
    (('given', 'in'), 0.813979789483902)
    (('case', 'are'), 0.8139366885789521)
    (('or', 'for'), 0.8135637758526464)
    (('such', 'if'), 0.8135637758526446)
    (('to', 'fact'), 0.81302038074395)
    (('*', 'of'), 0.8126865018882548)
    (('that', 'legislature'), 0.8120730881181153)
    ((',', 'he'), 0.8114065054326396)
    (('is', 'a'), 0.8112889579162861)
    (('as', 'they'), 0.8111714825786969)
    (('by', 'an'), 0.8109945062011974)
    (('any', 'on'), 0.8103064710363128)
    (('’', ','), 0.8081641973583942)
    (('of', 'error'), 0.8079814180900691)
    (('according', 'of'), 0.806611987478572)
    (('either', 'of'), 0.806611987478572)
    (('of', 'public'), 0.806611987478572)
    (('upon', 'or'), 0.8065281317988742)
    ((',', 'she'), 0.8063709155353429)
    ((',', 'unnecessary'), 0.8063709155353429)
    (('bond', ','), 0.8063709155353429)
    (('edward', ','), 0.8063709155353429)
    (('itself', ','), 0.8063709155353429)
    (('that', 'having'), 0.8049836633317966)
    (('.', 'first'), 0.8045798600093548)
    (('by', 'or'), 0.8037234532250928)
    (('party', 'is'), 0.8033243053059529)
    (('the', 'authorities'), 0.8025505908948709)
    (('the', 'intended'), 0.8025505908948709)
    (('avail', 'the'), 0.8025505908948674)
    (('conclusion', 'the'), 0.8025505908948674)
    (('countersign', 'the'), 0.8025505908948674)
    (('four', 'the'), 0.8025505908948674)
    (('regarded', 'the'), 0.8025505908948674)
    (('signed', 'the'), 0.8025505908948674)
    (('support', 'the'), 0.8025505908948674)
    (('the', 'contrary'), 0.8025505908948674)
    (('the', 'last'), 0.8025505908948674)
    (('the', 'recovery'), 0.8025505908948674)
    (('the', 'relied'), 0.8025505908948674)
    (('the', 'repeal'), 0.8025505908948674)
    (('the', 'rights'), 0.8025505908948674)
    (('the', 'thus'), 0.8025505908948674)
    (('the', 'use'), 0.8025505908948674)
    (('proceedings', 'in'), 0.8020071478178252)
    (('construction', 'to'), 0.8013385756963132)
    (('further', 'to'), 0.8013385756963132)
    (('to', 'such'), 0.8013385756963132)
    (('reversed', 'as'), 0.8006536106646998)
    (('by', 'power'), 0.8004030949334808)
    (('defense', 'a'), 0.7991011138988089)
    (('decisions', 'of'), 0.7981504087303755)
    (('was', 'said'), 0.7979586757191441)
    (('not', 'plea'), 0.7973931522331483)
    (('and', 'their'), 0.7967683577614011)
    (('their', 'and'), 0.7967683577614011)
    ((',', '“'), 0.7953326440828974)
    ((',', 'coles'), 0.7942980832347679)
    (('coles', ','), 0.7942980832347679)
    (('appears', 'the'), 0.7922822554410409)
    (('the', 'officer'), 0.7922822554410409)
    (('questions', 'to'), 0.7920607621083029)
    (('to', 'into'), 0.7920607621083029)
    (('to', 'pleas'), 0.7920607621083029)
    (('is', 'to'), 0.7915398395386859)
    (('because', 'the'), 0.7910549520570385)
    (('by', 'the'), 0.7908805977102347)
    (('which', 'action'), 0.7902687177997674)
    (('which', 'note'), 0.7902687177997674)
    (('too', '.'), 0.7894729676191474)
    (('in', 'court'), 0.7893296054238625)
    (('.', 'must'), 0.7878041472585302)
    (('the', 'paper'), 0.78744369850466)
    (('and', 'proceedings'), 0.787369659759154)
    (('plaintiff', 'an'), 0.7867905465658271)
    (('as', 'he'), 0.786459729081944)
    (('suit', 'be'), 0.7862323360574557)
    (('out', 'the'), 0.7861561526441996)
    (('justice', 'case'), 0.7859223124093511)
    (('of', 'writing'), 0.7855503719507411)
    (('general', 'that'), 0.7839220478039657)
    (('statute', 'that'), 0.7839220478039657)
    ((',', 'scates'), 0.7836508390352606)
    (('be', 'them'), 0.7833207406375529)
    (('as', 'if'), 0.7818811944841748)
    (('contract', ','), 0.7808358234282053)
    (('not', 'there'), 0.7804430833032114)
    (('whether', 'is'), 0.7798143601626002)
    ((',', 'they'), 0.7797357890318786)
    (('a', 'time'), 0.7797357890318786)
    (('a', 'justice'), 0.779735789031875)
    (('three', '.'), 0.7794888790465251)
    (('warrant', '.'), 0.7794888790465251)
    (('and', 'out'), 0.7791957283134536)
    (('out', 'and'), 0.7791957283134536)
    (('is', 'evidence'), 0.7784331297642417)
    (('court', 'that'), 0.7782309357261141)
    (('et', ','), 0.7781126554938282)
    (('presented', 'the'), 0.7776759222558383)
    (('the', 'recover'), 0.7776759222558383)
    (('words', 'of'), 0.7766172471807309)
    (('upon', 'an'), 0.7758313345759582)
    (('was', 'without'), 0.7753224792761131)
    (('within', 'and'), 0.7743135069337086)
    ((',', 'lockwood'), 0.7741905097861945)
    (('this', 'suit'), 0.774006033724195)
    (('established', 'the'), 0.7739814386980974)
    (('the', 'replication'), 0.7739814386980974)
    (('taken', 'court'), 0.7736002694012036)
    (('that', 'it'), 0.772922454099497)
    (('case', 'they'), 0.7726662157070869)
    (('that', 'been'), 0.7722086272401789)
    (('to', 'whether'), 0.7720230088717592)
    (('be', 'time'), 0.7717327663623408)
    (('by', 'against'), 0.7716713763901666)
    (('for', 'an'), 0.7707941463291554)
    (('to', 'declaration'), 0.7706417784733972)
    (('of', 'sale'), 0.7699535168997222)
    (('of', 'sheriff'), 0.7699535168997222)
    (('of', 'them'), 0.7699535168997222)
    ((',', 'being'), 0.7693767080531408)
    (('so', 'to'), 0.7684739417906883)
    (('to', 'amount'), 0.7684739417906847)
    (('have', 'by'), 0.7684665730562479)
    (('the', 'demurrer'), 0.7677851727341896)
    (('can', 'for'), 0.7677600862395195)
    (('.', 'bank'), 0.7671051545906948)
    (('.', 'presented'), 0.7671051545906948)
    (('therefore', 'to'), 0.7661821302826972)
    (('necessary', 'a'), 0.7660608521347996)
    (('not', 'opinion'), 0.7659979559576158)
    ((',', 'admit'), 0.765728931037998)
    ((',', 'commencement'), 0.765728931037998)
    ((',', 'complainants'), 0.765728931037998)
    ((',', 'inquire'), 0.765728931037998)
    ((',', 'issuing'), 0.765728931037998)
    ((',', 'patent'), 0.765728931037998)
    ((',', 'per'), 0.765728931037998)
    ((',', 'refusal'), 0.765728931037998)
    (('consequently', ','), 0.765728931037998)
    (('contingency', ','), 0.765728931037998)
    (('contracts', ','), 0.765728931037998)
    (('issuing', ','), 0.765728931037998)
    (('judicial', ','), 0.765728931037998)
    (('material', ','), 0.765728931037998)
    (('named', ','), 0.765728931037998)
    (('noble', ','), 0.765728931037998)
    (('themselves', ','), 0.765728931037998)
    (('clerk', 'of'), 0.7647918117839438)
    (('of', 'clerk'), 0.7647918117839438)
    (('to', 'issue'), 0.7645800256861968)
    (('.', 'it'), 0.7644507011447139)
    (('and', 'i'), 0.763718391694507)
    (('jury', 'and'), 0.763718391694507)
    (('not', 'error'), 0.7635297003668668)
    (('same', 'be'), 0.7635122595573698)
    (('that', 'therefore'), 0.763163487637172)
    (('therefore', 'that'), 0.763163487637172)
    (('plea', 'that'), 0.7631634876371685)
    (('the', 'case'), 0.7630595398776983)
    (('appoint', 'the'), 0.7619086063975189)
    (('enter', 'the'), 0.7619086063975189)
    (('executing', 'the'), 0.7619086063975189)
    (('the', 'age'), 0.7619086063975189)
    (('the', 'appoint'), 0.7619086063975189)
    (('the', 'claimed'), 0.7619086063975189)
    (('the', 'executing'), 0.7619086063975189)
    (('no', 'had'), 0.7616995656029975)
    (('question', 'for'), 0.7613338170800859)
    (('no', 'to'), 0.7611814492243312)
    (('county', 'of'), 0.760675703311712)
    (('recover', 'of'), 0.760675703311712)
    (('a', 'settled'), 0.7606269660841729)
    (('appointment', 'a'), 0.7606269660841729)
    (('considered', 'a'), 0.7606269660841729)
    (('or', 'at'), 0.7604807642794391)
    (('against', 'judgment'), 0.7603833046400812)
    (('decree', '.'), 0.7597256242250943)
    (('but', 'this'), 0.7595064640290836)
    (('this', 'were'), 0.7595064640290801)
    (('”', 'it'), 0.7582489659171721)
    (('by', 'said'), 0.7581881470265408)
    (('court', 'where'), 0.7576293124894207)
    (('not', 'have'), 0.7565769397250968)
    (('dollars', ','), 0.7557448424653757)
    (('not', 'but'), 0.7555729765385237)
    (('to', 'argument'), 0.7555348860831863)
    (('and', 'would'), 0.7549481820667765)
    (('due', 'and'), 0.7549481820667765)
    (('4', '.'), 0.7548268248122554)
    (('a', 'action'), 0.7546448080690453)
    (('”', 'is'), 0.7543374541467784)
    (('a', 'consideration'), 0.7537405804989312)
    (('consideration', 'a'), 0.7537405804989312)
    (('general', ','), 0.7531288942583636)
    (('of', 'circuit'), 0.7521642034561928)
    ((',', 'both'), 0.7512293613428831)
    (('recess', ','), 0.7512293613428831)
    (('act', 'which'), 0.7510092873525416)
    (('in', 'affidavit'), 0.7504768471777439)
    (('notice', 'in'), 0.7504768471777439)
    (('has', 'to'), 0.7495361454617466)
    ((')', 'an'), 0.7487920818236091)
    (('the', 'judgment'), 0.7485019478463713)
    ((',', 'i'), 0.7480269293045403)
    (('a', 'property'), 0.7480269293045367)
    (('here', 'a'), 0.7480269293045367)
    (('consideration', '.'), 0.7476527919245193)
    (('that', 'without'), 0.7472219437681495)
    (('upon', 'this'), 0.7464503112036347)
    (('execution', 'and'), 0.7462309649656689)
    (('the', 'payment'), 0.7459670625285035)
    (('note', 'in'), 0.7444946891626181)
    (('defendant', 'to'), 0.7443076306599359)
    (('defendants', 'to'), 0.7443076306599359)
    (('to', 'entered'), 0.7443076306599359)
    (('and', 'verdict'), 0.7443009378672691)
    (('should', 'it'), 0.7433910071900591)
    (('plea', 'a'), 0.743209913006762)
    (('doubt', 'the'), 0.7430495791462057)
    (('until', 'the'), 0.7430495791462057)
    (('do', 'to'), 0.7417290865581592)
    (('evidence', 'it'), 0.7417026570372869)
    (('question', 'this'), 0.7415845560318175)
    (('such', 'be'), 0.7415005649429247)
    (('of', 'the'), 0.7413979532750474)
    ((';', 'is'), 0.7409584243455782)
    (('upon', 'a'), 0.7402074248452379)
    (('not', 'from'), 0.7398851156384865)
    ((',', 'assignee'), 0.7392567196768063)
    ((',', 'leave'), 0.7392567196768063)
    ((',', 'permit'), 0.7392567196768063)
    ((',', 'rejected'), 0.7392567196768063)
    (('brown', ','), 0.7392567196768063)
    (('hands', ','), 0.7392567196768063)
    (('king', ','), 0.7392567196768063)
    (('oath', ','), 0.7392567196768063)
    (('and', 'without'), 0.739006638197754)
    (('here', 'in'), 0.7378768103981095)
    (('jurors', 'to'), 0.7376129780859273)
    (('to', 'another'), 0.7376129780859273)
    (('to', 'testimony'), 0.7376129780859273)
    (('”', '('), 0.7362226595871739)
    (('considered', 'and'), 0.7358393591190726)
    (('evidence', 'court'), 0.7335656075989512)
    ((',', '('), 0.7321672948904876)
    (('1.', 'the'), 0.7321612630034693)
    (('article', 'the'), 0.7321612630034693)
    (('the', 'existence'), 0.7321612630034693)
    (('the', 'judge'), 0.7321612630034693)
    (('the', 'others'), 0.7321612630034693)
    (('the', 'return'), 0.7321612630034693)
    (('is', 'where'), 0.7321075067633132)
    (('then', 'to'), 0.7319993716224005)
    (('i', 'it'), 0.7316138394137042)
    (('state', ','), 0.7312954339741218)
    (('in', 'the'), 0.7312414701505787)
    (('counsel', ','), 0.731082788231106)
    (('.', '3'), 0.7305792785655782)
    (('.', 'authorities'), 0.7305792785655782)
    (('.', 'granted'), 0.7305792785655782)
    (('3', '.'), 0.7305792785655782)
    (('entitled', '.'), 0.7305792785655782)
    (('granted', '.'), 0.7305792785655782)
    (('its', '.'), 0.7305792785655782)
    (('sentence', '.'), 0.7305792785655782)
    (('shown', '.'), 0.7305792785655782)
    (('or', 'not'), 0.730324182633364)
    (('secretary', 'to'), 0.7299997939760487)
    (('to', 'considered'), 0.7299997939760487)
    (('for', '”'), 0.7296249573527476)
    (('states', 'and'), 0.7289529735338327)
    (('from', 'an'), 0.7287053165084139)
    (('?', 'was'), 0.7285282677971736)
    ((',', 'when'), 0.7281845621778622)
    (('bill', 'that'), 0.7275395779064482)
    (('will', 'that'), 0.7275395779064482)
    (('he', 'state'), 0.7262422062478393)
    ((',', 'if'), 0.7245539678053277)
    (('of', 'both'), 0.7241498272865989)
    (('of', 'declared'), 0.7241498272865989)
    (('of', 'maker'), 0.7241498272865989)
    (('of', 'third'), 0.7241498272865989)
    (('since', 'of'), 0.7241498272865989)
    (('third', 'of'), 0.7241498272865989)
    (('issued', 'of'), 0.7241498272865954)
    (('a', 'bank'), 0.7231522606655094)
    (('to', 'decided'), 0.7231134083908124)
    (('rule', 'that'), 0.7225215031398236)
    (('upon', 'the'), 0.7223802422108854)
    (('a', '1'), 0.7219594577526749)
    (('this', 'one'), 0.7215386138300595)
    ((',', 'corporation'), 0.7206410415094595)
    ((',', 'lieutenant'), 0.7206410415094595)
    (('missouri', ','), 0.7206410415094595)
    (('satisfied', ','), 0.7206410415094595)
    (('substance', ','), 0.7206410415094595)
    (('duties', 'the'), 0.7200884307028943)
    (('it', 'by'), 0.7184784372245687)
    (('no', 'for'), 0.718265095188201)
    (('jurisdiction', 'in'), 0.7171182502313123)
    (('and', 'does'), 0.7169803318677559)
    (('character', 'the'), 0.7163939471451535)
    (('reverse', 'the'), 0.7163939471451535)
    (('the', 'estate'), 0.7163939471451535)
    (('the', 'property'), 0.7163939471451535)
    (('in', 'where'), 0.7156468630146442)
    (('process', ','), 0.7156054516121628)
    (('him', '.'), 0.7154723861753709)
    (('governor', ','), 0.7146547464437276)
    (('execution', 'it'), 0.7141264126848625)
    (('good', 'in'), 0.7130021417590804)
    (('shall', 'or'), 0.7129314127461548)
    (('that', 'note'), 0.7125374145672012)
    (('and', 'can'), 0.7124648507452633)
    (('court', 'has'), 0.7124599745873006)
    (('set', 'the'), 0.7123527819232898)
    (('had', 'his'), 0.7116588831034019)
    (('exceptions', 'to'), 0.7111407667247356)
    (('argument', 'the'), 0.709441186503387)
    (('security', 'the'), 0.709441186503387)
    (('the', 'absence'), 0.709441186503387)
    (('the', 'consent'), 0.709441186503387)
    (('the', 'intention'), 0.709441186503387)
    (('the', 'refused'), 0.709441186503387)
    (('not', 'an'), 0.7078950013940464)
    ((',', 'heirs'), 0.7068352419844288)
    ((',', 'lie'), 0.7068352419844288)
    ((',', 'neither'), 0.7068352419844288)
    (('morrison', ','), 0.7068352419844288)
    (('wilson', ','), 0.7068352419844288)
    (('(', 'if'), 0.7062924535221278)
    (('this', 'must'), 0.7057000203332855)
    (('court', 'right'), 0.7056346748511579)
    (('against', 'on'), 0.7049534708900822)
    (('his', 'and'), 0.7049074995671809)
    ((',', 'does'), 0.7044913015161285)
    (('in', 'their'), 0.7044906419035648)
    (('opinion', 'of'), 0.7042502698488953)
    (('to', '?'), 0.7034463749347353)
    (('statute', 'of'), 0.703088211758768)
    (('the', 'nature'), 0.7030149173439533)
    (('case', 'has'), 0.7027557112654499)
    (('and', 'that'), 0.7027519709413497)
    (('as', 'it'), 0.7027490226927142)
    (('against', 'or'), 0.7027356456402654)
    (('and', 'do'), 0.702480762172641)
    (('their', '.'), 0.7020101263688083)
    (('evidence', 'a'), 0.701733277030602)
    (('done', ','), 0.7011886788432875)
    (('rule', 'to'), 0.7010871020608107)
    (('is', 'from'), 0.6993919316199353)
    (('taken', 'the'), 0.698994399068269)
    (('term', '.'), 0.69887041883824)
    (('shall', 'for'), 0.6985980617321239)
    (('recover', 'and'), 0.6983646537004091)
    (('entered', 'the'), 0.6982139310801294)
    (('before', 'judgment'), 0.6974313990121566)
    (('this', 'power'), 0.6973847521212839)
    (('the', 'writ'), 0.6970574620475567)
    (('can', 'it'), 0.6968484212530299)
    ((',', 'nothing'), 0.6961879977849215)
    ((',', 'seem'), 0.6961879977849215)
    (('hobson', ','), 0.6961879977849215)
    (('circuit', 'of'), 0.6955806750898255)
    (('parties', 'in'), 0.6946236124442251)
    (('would', 'for'), 0.6942196212215492)
    (('an', 'of'), 0.693453030063683)
    (('he', 'such'), 0.6926114283665576)
    (('or', 'he'), 0.6926114283665576)
    (('officer', 'and'), 0.692212426718811)
    (('of', 'two'), 0.6917283495942215)
    (('apparent', 'the'), 0.6915192785061244)
    (('appointed', 'the'), 0.6915192785061244)
    (('assembly', 'the'), 0.6915192785061244)
    (('commencement', 'the'), 0.6915192785061244)
    (('consequently', 'the'), 0.6915192785061244)
    (('consider', 'the'), 0.6915192785061244)
    (('contracts', 'the'), 0.6915192785061244)
    (('material', 'the'), 0.6915192785061244)
    (('prosecuted', 'the'), 0.6915192785061244)
    (('refusal', 'the'), 0.6915192785061244)
    (('rendering', 'the'), 0.6915192785061244)
    (('service', 'the'), 0.6915192785061244)
    (('the', 'aforesaid'), 0.6915192785061244)
    (('the', 'became'), 0.6915192785061244)
    (('the', 'brothers'), 0.6915192785061244)
    (('the', 'february'), 0.6915192785061244)
    (('the', 'lord'), 0.6915192785061244)
    (('the', 'meaning'), 0.6915192785061244)
    (('the', 'objections'), 0.6915192785061244)
    (('the', 'possession'), 0.6915192785061244)
    (('the', 'proof'), 0.6915192785061244)
    (('to', 'action'), 0.6911030134881884)
    (('and', 'not'), 0.6901278221067173)
    (('trial', ','), 0.686657360046798)
    (('an', 'will'), 0.6866263846403964)
    (('common', 'a'), 0.6866263846403964)
    (('costs', ','), 0.6858756233487817)
    (('pleaded', 'to'), 0.6851455581917918)
    (('the', 'matter'), 0.6839060943962494)
    (('the', 'proper'), 0.6839060943962494)
    (('or', 'by'), 0.6834292195073814)
    (('case', 'of'), 0.6832775458710714)
    (('he', 'to'), 0.6831263905551381)
    (('court', 'jury'), 0.6828348054780218)
    ((',', 'then'), 0.6821128125731981)
    ((',', 'we'), 0.6818973402345243)
    (('making', 'the'), 0.681535189933502)
    (('the', 'authority'), 0.681535189933502)
    (('doubt', 'of'), 0.6797557079281447)
    (('of', 'below'), 0.6797557079281447)
    (('the', 'decree'), 0.6796938431093338)
    (('issue', 'that'), 0.679099222848695)
    (('necessary', 'that'), 0.679099222848695)
    (('that', 'issue'), 0.679099222848695)
    (('with', 'the'), 0.67891924172649)
    ((';', 'has'), 0.6787768483310117)
    (('would', 'an'), 0.6778561750126642)
    (('authorized', 'the'), 0.6770197088110095)
    (('the', 'manner'), 0.6770197088110095)
    (('the', 'offense'), 0.6770197088110095)
    (('the', 'requires'), 0.6770197088110095)
    (('have', 'to'), 0.6765835446883699)
    (('plaintiffs', 'in'), 0.6764762657339674)
    (('same', 'in'), 0.6764762657339674)
    (('against', 'it'), 0.6763843186933123)
    (('term', ','), 0.6751263822570905)
    (('.', 'pleas'), 0.6739957501992109)
    (('effect', '.'), 0.6739957501992109)
    (('of', 'note'), 0.6735237542166317)
    (('making', 'of'), 0.6735237542166281)
    (('of', 'authority'), 0.6735237542166281)
    (('evidence', 'by'), 0.6734909824512609)
    (('.', 'are'), 0.6728637807092923)
    (('his', 'he'), 0.6728029472863781)
    (('to', 'all'), 0.6717665283905916)
    (('we', 'this'), 0.6715587647533781)
    (('.', 'cases'), 0.6710782668169166)
    (('and', 'there'), 0.670883917278303)
    (('passed', ','), 0.6703093659593158)
    (('that', 'can'), 0.670054083245688)
    (('the', 'issue'), 0.669576725625447)
    (('be', '?'), 0.6683550265170304)
    (('to', 'circuit'), 0.6680720448328508)
    (('real', 'to'), 0.6680720448328472)
    (('to', 'real'), 0.6680720448328472)
    (('were', 'it'), 0.6677020755935104)
    (('and', 'if'), 0.6673139077414483)
    (('declares', 'the'), 0.6650470671449327)
    (('determined', 'the'), 0.6650470671449327)
    (('establishing', 'the'), 0.6650470671449327)
    (('render', 'the'), 0.6650470671449327)
    (('terms', 'the'), 0.6650470671449327)
    (('the', 'charge'), 0.6650470671449327)
    (('the', 'consequences'), 0.6650470671449327)
    (('the', 'copy'), 0.6650470671449327)
    (('the', 'davidson'), 0.6650470671449327)
    (('the', 'ordinary'), 0.6650470671449327)
    (('opinion', 'on'), 0.6648760315147477)
    (('is', 'given'), 0.6637905638231167)
    (('record', 'was'), 0.6635084759467631)
    (('of', 'such'), 0.6630383129832111)
    (('under', 'a'), 0.6627796426860293)
    (('record', 'that'), 0.6618801517999877)
    (('and', 'second'), 0.6618387776752925)
    (('him', 'and'), 0.6618387776752925)
    (('for', 'such'), 0.6615606824075968)
    (('to', 'new'), 0.6615587378741736)
    (('equity', '.'), 0.6601899506741802)
    (('of', 'facts'), 0.6600194898668832)
    (('term', 'of'), 0.6600194898668832)
    (('was', 'to'), 0.6598344696626057)
    (('said', 'that'), 0.6588268278224341)
    (('the', 'jurisdiction'), 0.6581606815596928)
    (('against', 'the'), 0.6565556062512954)
    (('of', 'law'), 0.6564052206507647)
    (('the', 'decision'), 0.656329850043825)
    (('1', 'if'), 0.6562517710225304)
    (('his', 'or'), 0.6560855523414446)
    (('statute', 'in'), 0.6554146502061364)
    (('statute', 'is'), 0.6548017805958608)
    (('in', 'all'), 0.6541084527055148)
    (('which', 'are'), 0.6538962760684832)
    (('no', 'but'), 0.6538962760684814)
    (('correct', 'of'), 0.6537604993952009)
    (('suit', 'was'), 0.6537604993952009)
    (('judgment', 'on'), 0.6532973574792855)
    (('plaintiff', 'and'), 0.65237844842623)
    (('peace', ','), 0.6521999854075844)
    (('trial', 'by'), 0.6519578209016181)
    ((',', 'can'), 0.6518609664797168)
    (('first', ','), 0.6512005430116119)
    (('will', 'a'), 0.6510024749096743)
    (('.', 'although'), 0.6504089298815927)
    (('.', 'objection'), 0.6504089298815927)
    (('had', 'by'), 0.6484000014884295)
    (('consideration', 'court'), 0.648069387317344)
    (('provision', 'the'), 0.6464313889775859)
    (('the', 'corporation'), 0.6464313889775859)
    (('the', 'demand'), 0.6464313889775859)
    (('the', 'while'), 0.6464313889775859)
    (('affidavit', 'of'), 0.6461473152853259)
    (('first', 'of'), 0.6461473152853259)
    (('matter', 'of'), 0.6461473152853259)
    (('of', 'proper'), 0.6461473152853259)
    (('settled', 'of'), 0.6461473152853259)
    (('take', 'of'), 0.6461473152853259)
    ((',', 'that'), 0.6451162567404936)
    (('to', ':'), 0.644026570866199)
    (('bill', 'in'), 0.6408523560032471)
    (('fact', 'in'), 0.6408523560032453)
    (('been', 'for'), 0.6407436762664993)
    (('law', ','), 0.6399573791661304)
    (('below', ','), 0.6397210461258922)
    (('that', 'have'), 0.6384084566292216)
    (('without', 'a'), 0.6382633630789947)
    (('.', 'those'), 0.6374698741740978)
    (('paper', '.'), 0.6374698741740978)
    (('plaintiffs', '.'), 0.6374698741740978)
    (('subject', '.'), 0.6374698741740978)
    (('with', 'a'), 0.6369956169157938)
    (('have', 'at'), 0.6357257332714941)
    (('to', 'making'), 0.6345194851218228)
    (('to', 'warrant'), 0.6345194851218228)
    (('at', 'the'), 0.6345126989968186)
    (('state', 'and'), 0.6343580412531864)
    (('been', 'as'), 0.6334481105759764)
    (('made', 'as'), 0.6334481105759764)
    (('appeared', 'the'), 0.6326255894525552)
    (('the', 'commission'), 0.6326255894525552)
    (('use', 'the'), 0.6326255894525552)
    (('court', 'to'), 0.6303481994061606)
    (('be', 'a'), 0.6302884004905849)
    (('point', '.'), 0.6292959427283975)
    (('of', 'deed'), 0.6289925942462595)
    (('to', 'cases'), 0.6286786065327625)
    (('bill', 'is'), 0.6281666540923965)
    (('is', 'bill'), 0.6281666540923965)
    (('himself', 'a'), 0.6277326955868254)
    (('judgment', 'law'), 0.6273729265380759)
    (('entered', '.'), 0.6262426187508439)
    (('and', 'process'), 0.626214867944574)
    (('and', 'they'), 0.6262148679445723)
    (('sufficient', 'that'), 0.6256599638872355)
    (('that', 'any'), 0.6256599638872338)
    (('that', 'does'), 0.6256599638872338)
    (('execution', 'be'), 0.6255120255112949)
    (('order', 'of'), 0.6246141537356813)
    (('in', 'execution'), 0.6233649292744055)
    (('the', 'error'), 0.6233477758645449)
    (('his', '.'), 0.6228230982096967)
    (('according', 'the'), 0.6219783452530478)
    (('appearance', 'the'), 0.6219783452530478)
    (('bar', 'the'), 0.6219783452530478)
    (('year', 'the'), 0.6219783452530478)
    (('be', 'but'), 0.6197296729172912)
    (('trial', '.'), 0.6195479661768353)
    (('for', 'the'), 0.6195382684922812)
    (('of', 'court'), 0.6181081898762741)
    (('to', 'his'), 0.6180313623332552)
    (('has', 'or'), 0.6176653340276239)
    (('point', 'is'), 0.6176487821783976)
    (('not', 'before'), 0.6175926076211695)
    (('himself', 'in'), 0.6175825766803982)
    (('appeal', 'the'), 0.6175186970623479)
    (('executed', 'the'), 0.6175186970623479)
    (('yet', 'the'), 0.6175186970623479)
    (('court', 'then'), 0.6169206887466778)
    (('pay', 'a'), 0.6162370567489983)
    (('to', 'officer'), 0.6159835336843962)
    (('under', 'was'), 0.6154141877457171)
    (('be', 'from'), 0.6146890562167613)
    (('that', 'under'), 0.6137858635989417)
    ((',', 'according'), 0.6137258375929484)
    ((',', 'did'), 0.6137258375929484)
    ((',', 'granting'), 0.6137258375929484)
    ((',', 'reasons'), 0.6137258375929484)
    ((',', 'recovery'), 0.6137258375929484)
    ((',', 'regarded'), 0.6137258375929484)
    ((',', 'regulating'), 0.6137258375929484)
    ((',', 'repeal'), 0.6137258375929484)
    ((',', 'since'), 0.6137258375929484)
    ((',', 'will'), 0.6137258375929484)
    (('answer', ','), 0.6137258375929484)
    (('arrest', ','), 0.6137258375929484)
    (('became', ','), 0.6137258375929484)
    (('children', ','), 0.6137258375929484)
    (('consent', ','), 0.6137258375929484)
    (('conviction', ','), 0.6137258375929484)
    (('everett', ','), 0.6137258375929484)
    (('face', ','), 0.6137258375929484)
    (('fraud', ','), 0.6137258375929484)
    (('future', ','), 0.6137258375929484)
    (('judgments', ','), 0.6137258375929484)
    (('name', ','), 0.6137258375929484)
    (('non', ','), 0.6137258375929484)
    (('objections', ','), 0.6137258375929484)
    (('prisoner', ','), 0.6137258375929484)
    (('raised', ','), 0.6137258375929484)
    (('settled', ','), 0.6137258375929484)
    (('wife', ','), 0.6137258375929484)
    (('†', ','), 0.6137258375929484)
    ((',', 'agreed'), 0.6137258375929449)
    ((',', 'fee'), 0.6137258375929449)
    ((',', 'setting'), 0.6137258375929449)
    ((',', 'took'), 0.6137258375929449)
    (('above', ','), 0.6137258375929449)
    (('alone', ','), 0.6137258375929449)
    (('collateral', ','), 0.6137258375929449)
    (('existence', ','), 0.6137258375929449)
    (('facto', ','), 0.6137258375929449)
    (('hand', ','), 0.6137258375929449)
    (('importance', ','), 0.6137258375929449)
    (('my', ','), 0.6137258375929449)
    (('second', ','), 0.6137258375929449)
    (('to', 'plaintiff'), 0.6136242608104716)
    (('sworn', 'the'), 0.6135167665048513)
    (('take', 'the'), 0.6135167665048513)
    (('the', 'go'), 0.6135167665048513)
    (('the', 'purpose'), 0.6135167665048513)
    (('law', 'by'), 0.6126657897552192)
    (('made', 'to'), 0.6125769322411436)
    (('.', 'affidavit'), 0.6119347820669567)
    (('think', '.'), 0.6119347820669567)
    (('any', 'case'), 0.6118929126343033)
    (('the', 'county'), 0.6116659708169081)
    (('and', 'did'), 0.6112127046053253)
    (('.', 'but'), 0.6102850448478669)
    (('or', 'state'), 0.6099891377519882)
    (('to', 'out'), 0.6098574308875548)
    (('and', 'shall'), 0.6097502665268415)
    (('”', 'v.'), 0.6086726098666624)
    (('the', 'discharge'), 0.6066303809196114)
    (('the', 'notes'), 0.6066303809196114)
    (('trial', 'as'), 0.6064810629757105)
    (('in', 'consideration'), 0.6060869378425693)
    (('pay', 'in'), 0.6060869378425693)
    (('states', 'in'), 0.6060869378425693)
    (('does', 'to'), 0.6042255628082245)
    (('to', 'one'), 0.6042255628082245)
    (('his', 'but'), 0.6038555935688876)
    (('of', 'who'), 0.6038555935688876)
    (('against', '.'), 0.6031999720893495)
    (('and', 'himself'), 0.6029450886217269)
    (('jury', '.'), 0.6018459644433776)
    (('by', 'a'), 0.6013411966954259)
    (('point', 'in'), 0.6011881384297268)
    (('exercise', 'the'), 0.6009167297252169)
    (('facts', 'the'), 0.6009167297252169)
    (('relation', 'the'), 0.6009167297252169)
    (('the', 'exercise'), 0.6009167297252169)
    (('of', 'brought'), 0.600767411781316)
    (('be', 'an'), 0.6006208499695838)
    (('that', 'second'), 0.59966475535429)
    (('case', 'upon'), 0.598836759808858)
    (('of', 'subject'), 0.5986189452027411)
    (('in', 'first'), 0.5984737537326943)
    ((';', 'no'), 0.5973127477021158)
    (('no', ';'), 0.5973127477021158)
    (('judgment', 'been'), 0.5972474688604095)
    (('.', 'is'), 0.5969766901555502)
    (('had', 'he'), 0.5961816656834671)
    (('contended', 'the'), 0.5960997134274422)
    (('was', 'in'), 0.5934869010629349)
    (('decided', 'is'), 0.5934012359317187)
    (('doubt', '.'), 0.5930757548156436)
    (('form', '.'), 0.5930757548156436)
    (('case', 'from'), 0.5923527262386585)
    (('and', 'held'), 0.591449449783898)
    (('be', 'no'), 0.5911605207205213)
    (('its', 'be'), 0.5911605207205177)
    (('that', 'judgment'), 0.5906413747088202)
    ((',', 'rep.'), 0.5890637833586787)
    (('be', 'against'), 0.5877699315197447)
    ((',', 'would'), 0.5872536262317567)
    (('a', 'due'), 0.5870907110894805)
    (('of', 'courts'), 0.5866463035366642)
    (('of', 'much'), 0.5866463035366642)
    (('to', 'every'), 0.5856098846408777)
    (('bond', 'to'), 0.5856098846408742)
    (('construction', 'the'), 0.5853198746741981)
    (('is', 'action'), 0.5834171473590963)
    ((',', 'shall'), 0.5826989419723247)
    (('where', 'the'), 0.5786867823417516)
    (('as', 'in'), 0.578628942335822)
    (('a', 'statute'), 0.5781019278622281)
    (('amount', 'the'), 0.5779903328757108)
    (('appear', 'the'), 0.5760420610861878)
    (('the', 'presented'), 0.5760420610861878)
    (('as', 'statute'), 0.5747722032483722)
    (('the', 'process'), 0.574281603221749)
    (('principle', ','), 0.5741974734063113)
    (('in', 'opinion'), 0.5741145481042906)
    (('had', 'in'), 0.573382772769861)
    (('not', 'such'), 0.5727829056468821)
    (('into', 'a'), 0.571149167220458)
    (('declaration', 'this'), 0.570472639639064)
    (('this', 'ought'), 0.570472639639064)
    (('.', 'declaration'), 0.5701146063723321)
    (('no', 'been'), 0.5698320112800062)
    (('suit', ','), 0.5693317182344941)
    (('on', 'all'), 0.5692983715325184)
    (('the', 'plea'), 0.5680853372578447)
    (('general', 'in'), 0.5679518089557973)
    (('for', 'of'), 0.5678738873561358)
    (('and', 'only'), 0.5678626294662799)
    ((',', 'there'), 0.5676294230034955)
    (('being', 'to'), 0.5672313553260224)
    (('.', '1.'), 0.5670805462826998)
    (('.', 'can'), 0.5670805462826998)
    (('.', 'stated'), 0.5670805462826998)
    (('our', '.'), 0.5670805462826998)
    (('present', '.'), 0.5670805462826998)
    (('stated', '.'), 0.5670805462826998)
    (('and', 'person'), 0.5666815446349531)
    (('not', 'law'), 0.5661498133144356)
    (('that', 'the'), 0.5661215416432448)
    (('on', 'judgment'), 0.5658345162289464)
    (('or', 'but'), 0.564770385595498)
    (('all', 'be'), 0.5631461445509238)
    (('three', ','), 0.5630997645229776)
    (('he', 'an'), 0.5623789575927276)
    (('after', 'of'), 0.5612112561603944)
    ((',', 'justice'), 0.5609843755209738)
    (('court', 'had'), 0.5609174382222619)
    (('execution', ','), 0.5606145011333865)
    (('then', 'and'), 0.5603007512132301)
    (('does', 'it'), 0.5593448975030952)
    (('to', 'any'), 0.559137673279686)
    (('record', 'not'), 0.5586062926460329)
    (('be', 'whether'), 0.5576079610094951)
    (('whether', 'be'), 0.5576079610094951)
    (('and', 'legal'), 0.5575021178605581)
    (('and', 'now'), 0.5575021178605581)
    (('motion', 'and'), 0.5575021178605581)
    (('of', 'yet'), 0.5542248258442868)
    (('of', 'great'), 0.5542248258442832)
    (('of', 'title'), 0.5542248258442832)
    (('of', 'words'), 0.5542248258442832)
    ((',', 'as'), 0.5533532196134665)
    (('brought', 'in'), 0.5530938502286844)
    (('this', 'not'), 0.5525060931096135)
    (('was', 'no'), 0.5517980884497433)
    (('to', 'said'), 0.5516625527175414)
    (('motion', 'to'), 0.5516625527175378)
    (('to', 'legal'), 0.5516625527175378)
    ((',', 'could'), 0.5509900822449865)
    (('circuit', 'be'), 0.5505185362231728)
    (('.', 'according'), 0.5500070329237552)
    (('liable', '.'), 0.5500070329237552)
    (('public', '.'), 0.5500070329237552)
    (('here', ','), 0.5495955001732327)
    (('inquiry', ','), 0.5495955001732327)
    (('premises', ','), 0.5495955001732327)
    (('property', ','), 0.5495955001732327)
    (('writing', ','), 0.5495955001732327)
    (('form', 'a'), 0.5491228608904599)
    (('to', 'given'), 0.5490840086157611)
    (('cases', 'is'), 0.549007116573268)
    (('action', 'by'), 0.5488643279375154)
    ((',', 'therefore'), 0.5481374959653706)
    (('and', 'received'), 0.5463615602553595)
    (('writ', 'and'), 0.5463615602553595)
    (('is', 'sheriff'), 0.5460955211533651)
    (('taken', 'of'), 0.5448261278420361)
    (('case', 'it'), 0.5439495221033894)
    (('a', 'than'), 0.5436684307983519)
    (('of', 'people'), 0.5435775816447794)
    (('law', 'it'), 0.5433739405913087)
    (("'", ','), 0.5433365097015503)
    (('trespass', ','), 0.5433365097015503)
    (('do', 'that'), 0.5407710663007208)
    (('county', 'it'), 0.5407292193357485)
    ((',', 'which'), 0.5397252561491719)
    (('been', 'or'), 0.5395998098605936)
    (('become', 'the'), 0.5395161850610748)
    (('both', 'the'), 0.5395161850610748)
    (('consent', 'the'), 0.5395161850610748)
    (('decided', 'the'), 0.5395161850610748)
    (('dismissed', 'the'), 0.5395161850610748)
    (('end', 'the'), 0.5395161850610748)
    (('everett', 'the'), 0.5395161850610748)
    (('face', 'the'), 0.5395161850610748)
    (('given', 'the'), 0.5395161850610748)
    (('interest', 'the'), 0.5395161850610748)
    (('justified', 'the'), 0.5395161850610748)
    (('name', 'the'), 0.5395161850610748)
    (('necessity', 'the'), 0.5395161850610748)
    (('notice', 'the'), 0.5395161850610748)
    (('pleas', 'the'), 0.5395161850610748)
    (('prove', 'the'), 0.5395161850610748)
    (('reasoning', 'the'), 0.5395161850610748)
    (('reasons', 'the'), 0.5395161850610748)
    (('rendered', 'the'), 0.5395161850610748)
    (('requires', 'the'), 0.5395161850610748)
    (('rules', 'the'), 0.5395161850610748)
    (('showing', 'the'), 0.5395161850610748)
    (('the', 'application'), 0.5395161850610748)
    (('the', 'arrest'), 0.5395161850610748)
    (('the', 'believe'), 0.5395161850610748)
    (('the', 'claim'), 0.5395161850610748)
    (('the', 'common'), 0.5395161850610748)
    (('the', 'did'), 0.5395161850610748)
    (('the', 'effect'), 0.5395161850610748)
    (('the', 'granting'), 0.5395161850610748)
    (('the', 'ground'), 0.5395161850610748)
    (('the', 'grounds'), 0.5395161850610748)
    (('the', 'lands'), 0.5395161850610748)
    (('the', 'opportunity'), 0.5395161850610748)
    (('the', 'public'), 0.5395161850610748)
    (('the', 'reasoning'), 0.5395161850610748)
    (('the', 'supposed'), 0.5395161850610748)
    (('the', 'vacancy'), 0.5395161850610748)
    (('the', 'whose'), 0.5395161850610748)
    (('the', 'witness'), 0.5395161850610748)
    (('vacant', 'the'), 0.5395161850610748)
    (('agreed', 'the'), 0.5395161850610712)
    (('agreement', 'the'), 0.5395161850610712)
    (('authorize', 'the'), 0.5395161850610712)
    (('believed', 'the'), 0.5395161850610712)
    (('directed', 'the'), 0.5395161850610712)
    (('except', 'the'), 0.5395161850610712)
    (('founded', 'the'), 0.5395161850610712)
    (('irregularity', 'the'), 0.5395161850610712)
    (('rankin', 'the'), 0.5395161850610712)
    (('the', 'acting'), 0.5395161850610712)
    (('the', 'clair'), 0.5395161850610712)
    (('the', 'enter'), 0.5395161850610712)
    (('the', 'goods'), 0.5395161850610712)
    (('the', 'irregularity'), 0.5395161850610712)
    (('the', 'scire'), 0.5395161850610712)
    (('the', 'sued'), 0.5395161850610712)
    (('be', 'at'), 0.5385998462424091)
    (('to', 'further'), 0.5383041698625206)
    (('we', 'to'), 0.5383041698625206)
    (('court', 'judgment'), 0.5381170978210932)
    (('part', '.'), 0.5379342006231802)
    (('which', 'an'), 0.5378257232336878)
    (('not', 'one'), 0.5370328806976801)
    (('are', 'as'), 0.5369401297226339)
    (('error', 'that'), 0.5366549578284889)
    (('award', ','), 0.5357233255916753)
    (('mere', ','), 0.5357233255916753)
    (('sworn', ','), 0.5357233255916753)
    (('view', ','), 0.5357233255916753)
    (('witnesses', ','), 0.5357233255916753)
    (('court', 'is'), 0.5354538867224399)
    (('case', 'had'), 0.5352716310313923)
    (('i', 'that'), 0.5348944999640501)
    (('a', 'which'), 0.534623291195345)
    (('person', 'it'), 0.5345769923541503)
    (('to', 'second'), 0.5331424647467422)
    (('is', 'as'), 0.5330286179522403)
    (('assigned', ','), 0.5328058422093811)
    (('not', 'after'), 0.5323563934882074)
    (('not', 'then'), 0.5323563934882074)
    (('them', 'in'), 0.5296348774046962)
    (('defendant', 'not'), 0.5295577601355355)
    (('.', 'statute'), 0.5289454173959278)
    (('a', 'has'), 0.5286245505739018)
    (('right', 'is'), 0.5281097756800008)
    (('and', 'being'), 0.5279830309405042)
    (('and', 'parties'), 0.5279830309405042)
    ((',', '1'), 0.5277526144830951)
    (('which', 'be'), 0.5266202685258072)
    (('administrator', ','), 0.5262629963426093)
    (('certainly', ','), 0.5262629963426093)
    (('days', ','), 0.5262629963426093)
    (('hundred', ','), 0.5262629963426093)
    (('replication', ','), 0.5262629963426093)
    (('was', 'a'), 0.5256345079680926)
    (('made', 'for'), 0.5252664588465628)
    (('state', 'for'), 0.5252664588465628)
    (('and', 'courts'), 0.5243352539253614)
    (('the', 'ought'), 0.5242494284077637)
    ((';', 'was'), 0.5237837122801494)
    (('by', 'of'), 0.5233874219216936)
    (('of', 'which'), 0.5232371333606025)
    (('most', 'of'), 0.5225159661169485)
    (('of', 'term'), 0.5225159661169485)
    (('reverse', 'of'), 0.5225159661169485)
    (('parties', 'it'), 0.5223506900208932)
    (('court', 'would'), 0.52206150240524)
    (('on', 'of'), 0.5219609766733306)
    (('or', '”'), 0.5215658870303308)
    (('not', 'a'), 0.5202679982182765)
    ((',', 'it'), 0.5193102357008428)
    (('to', 'form'), 0.5184956887823375)
    (('give', 'a'), 0.5167013831980825)
    (('for', 'which'), 0.5162213192435594)
    (('”', 'in'), 0.5154843890616618)
    (('was', 'when'), 0.5149327946978843)
    (('of', 'defense'), 0.5146964616576497)
    ((',', 'wilson'), 0.5141901640420308)
    (('important', ','), 0.5141901640420308)
    (('last', ','), 0.5141901640420308)
    (('to', 'jury'), 0.5134600988850408)
    (('not', 'power'), 0.5128790189889045)
    (('an', 'upon'), 0.5127969287421656)
    (('below', 'is'), 0.5124812405481514)
    ((',', 'after'), 0.512187811130886)
    (('that', 'court'), 0.5114443950312122)
    (('.', 's'), 0.5111341412794665)
    (('suit', 'this'), 0.5109716278904024)
    (('have', 'but'), 0.5104046824789492)
    (('law', 'this'), 0.5096474469430206)
    (('defendants', ','), 0.509389177778214)
    (('.', 'clearly'), 0.5081868572291306)
    (('.', 'hubbard'), 0.5081868572291306)
    (('on', 'such'), 0.5081868572291306)
    (('his', '('), 0.5081722455135775)
    (('of', 'this'), 0.507831920359834)
    (('contract', 'to'), 0.5076073726396046)
    (('evidence', 'to'), 0.5076073726396046)
    (('notice', 'to'), 0.5076073726396011)
    (('think', 'to'), 0.5076073726396011)
    (('to', 'affidavit'), 0.5076073726396011)
    (('to', 'notice'), 0.5076073726396011)
    (('an', ';'), 0.5072374034002642)
    (('an', 'on'), 0.5072374034002642)
    (('next', ','), 0.5068106336764338)
    (('action', 'a'), 0.5067172946254601)
    (('in', 'these'), 0.5065512642916552)
    (('make', 'in'), 0.5065512642916552)
    (('smith', 'in'), 0.5065512642916552)
    (('state', 'which'), 0.5052917590852957)
    (('a', 'where'), 0.5034045605846238)
    (('been', 'a'), 0.503026446301007)
    (('recover', 'the'), 0.5020414796424113)
    (('he', 'was'), 0.5017574059501513)
    (('pay', 'of'), 0.5017574059501513)
    (('was', 'any'), 0.5017574059501513)
    (('of', '2.'), 0.5017574059501477)
    (('return', 'of'), 0.5017574059501477)
    (('second', 'of'), 0.5017574059501477)
    (('be', 'in'), 0.5003990373100606)
    (('is', 'constitution'), 0.5002918315402383)
    (('the', 'principle'), 0.4999878208744377)
    (('in', 'such'), 0.499887534010643)
    (('a', 'will'), 0.49899938146462475)
    (('jury', 'a'), 0.49899938146462475)
    ((',', 'satisfied'), 0.49824862017301186)
    (('created', ','), 0.49824862017301186)
    (('demand', ','), 0.49824862017301186)
    (('purchase', ','), 0.49824862017301186)
    (('plaintiff', 'to'), 0.49814704339053506)
    (('before', 'of'), 0.4963439097637874)
    (('other', '.'), 0.4961140249285556)
    (('case', 'been'), 0.49540717015345237)
    (('“', 'or'), 0.4949307605268629)
    (('record', 'in'), 0.49427293451321574)
    (('a', 'might'), 0.49398130669799656)
    (('therefore', 'is'), 0.4938655623808046)
    (('.', 'below'), 0.4935400812647295)
    (('to', 'those'), 0.49250048024939375)
    (('plea', 'in'), 0.4920516945965403)
    (('record', '.'), 0.4917924189784628)
    (('or', 'if'), 0.49163568096528465)
    (('of', '?'), 0.49148907049632484)
    (('if', 'an'), 0.4906862271364183)
    (('has', 'this'), 0.49047331757384427)
    (('this', 'has'), 0.49047331757384427)
    (('shall', 'to'), 0.49045265160053475)
    ((',', 'should'), 0.49034342208766546)
    ((',', 'no'), 0.4897371203174927)
    (('proved', ','), 0.488194955509087)
    (('of', 'supreme'), 0.4871106299857466)
    (('until', 'of'), 0.4871106299857466)
    (('that', 'law'), 0.48683225918991724)
    (('when', 'that'), 0.48683225918991724)
    (('this', 'shall'), 0.4862037227414575)
    (('question', 'to'), 0.4860742110899636)
    (('to', 'trial'), 0.4860742110899636)
    (('be', 'under'), 0.48485161857417935)
    (('of', 'its'), 0.48468389259121025)
    (('opinion', 'it'), 0.4843037858729282)
    (('constitution', 'in'), 0.4838311877915693)
    (('in', 'constitution'), 0.4838311877915693)
    (('his', 'to'), 0.4837302706216633)
    (('into', 'of'), 0.4831417277828045)
    (('of', 'section'), 0.4831417277828045)
    (('of', ';'), 0.48314172778280096)
    (('judgment', 'has'), 0.4826679150850417)
    (('declaration', '.'), 0.48265176512199304)
    (('estate', ','), 0.48248130431469605)
    (('v.', '’'), 0.4811828747447464)
    (('justices', 'the'), 0.4806224960075056)
    (('reason', 'the'), 0.4806224960075056)
    (('required', 'the'), 0.4806224960075056)
    (('shown', 'the'), 0.4806224960075056)
    (('the', 'instrument'), 0.4806224960075056)
    (('“', 'for'), 0.48059740951283203)
    (('below', 'in'), 0.4800790529304635)
    (('would', 'in'), 0.4800790529304635)
    (('if', 'this'), 0.479920881931438)
    ((')', 'is'), 0.4779240185117857)
    (('county', 'is'), 0.4779240185117821)
    (('to', 'the'), 0.4773271285454648)
    (('from', 'this'), 0.4769318717996427)
    ((',', 'b'), 0.4762223138430137)
    ((',', 'brown'), 0.4762223138430137)
    ((',', 'charge'), 0.4762223138430137)
    ((',', 'count'), 0.4762223138430137)
    ((',', 'declares'), 0.4762223138430137)
    ((',', 'delivered'), 0.4762223138430137)
    ((',', 'objected'), 0.4762223138430137)
    ((',', 'terms'), 0.4762223138430137)
    ((',', 'virtue'), 0.4762223138430137)
    (('appointment', 'of'), 0.4762223138430137)
    (('assumpsit', ','), 0.4762223138430137)
    (('claim', ','), 0.4762223138430137)
    (('courts', ','), 0.4762223138430137)
    (('entry', ','), 0.4762223138430137)
    (('errors', ','), 0.4762223138430137)
    (('favor', ','), 0.4762223138430137)
    (('imprisonment', ','), 0.4762223138430137)
    (('interested', ','), 0.4762223138430137)
    (('merely', ','), 0.4762223138430137)
    (('objected', ','), 0.4762223138430137)
    (('provides', ','), 0.4762223138430137)
    (('show', 'of'), 0.4762223138430137)
    (('territory', ','), 0.4762223138430137)
    (('urged', ','), 0.4762223138430137)
    (('the', 'several'), 0.47538584764135905)
    (('with', 'as'), 0.47523652969745456)
    (('and', 'it'), 0.4738490604709895)
    (('”', 'not'), 0.4729025432779643)
    (('had', 'a'), 0.4725015792875489)
    (('constitution', ','), 0.47170683272052116)
    (('which', 'such'), 0.471660981204014)
    (('issue', ','), 0.47076788375090217)
    (('good', 'to'), 0.47013266722094116)
    (('to', 'appear'), 0.47013266722094116)
    (('and', 'might'), 0.46919369973289804)
    (('notes', 'the'), 0.4691268571696767)
    (('pay', 'the'), 0.4691268571696767)
    (('the', "'"), 0.4691268571696767)
    (('at', 'of'), 0.4686490941382111)
    (('it', '“'), 0.4685794335799116)
    (('“', 'it'), 0.4685794335799116)
    (('that', 'question'), 0.46770760411099843)
    (('county', '.'), 0.4675448727317857)
    (('duties', '.'), 0.4675448727317857)
    (('ground', '.'), 0.4675448727317857)
    (('.', 'held'), 0.46754487273178214)
    (('executed', '.'), 0.46754487273178214)
    (('in', 'new'), 0.4670229001050181)
    (('new', 'in'), 0.4670229001050181)
    (('further', ','), 0.4668844492636737)
    (('any', 'of'), 0.46635206981895294)
    (('may', 'a'), 0.46423396330394695)
    (('be', 'if'), 0.46413280809510127)
    (('not', 'therefore'), 0.4639694185079577)
    (('may', 'on'), 0.46238316761600373)
    ((',', 'am'), 0.4617227441478988)
    (('offense', ','), 0.4617227441478988)
    (('p.', ','), 0.4617227441478988)
    (('affidavit', 'the'), 0.46151367305980173)
    (('duty', 'the'), 0.46151367305980173)
    (('go', 'the'), 0.46151367305980173)
    (('view', 'the'), 0.46151367305980173)
    (('all', 'in'), 0.4614633747631167)
    (('are', 'which'), 0.46125119812608517)
    (('of', 'whom'), 0.4611154214528064)
    (('rule', 'of'), 0.4611154214528064)
    (('to', 'rule'), 0.4600790025570163)
    (('he', 'no'), 0.45980922395218116)
    (('*', 'is'), 0.45954548919692684)
    (('cause', 'to'), 0.4590774787119436)
    (('to', 'an'), 0.4586977721586578)
    (('opinion', 'the'), 0.458216082959229)
    (('act', 'in'), 0.4573077452718053)
    (('making', ','), 0.4561845606064665)
    (('warrant', ','), 0.4561845606064665)
    (('and', 'he'), 0.4561637512837784)
    (('than', 'the'), 0.45545192027259773)
    (('act', 'it'), 0.4549598242907855)
    (('term', 'to'), 0.4543653513626218)
    (('of', 'jury'), 0.454060663918856)
    (('a', '?'), 0.4539656278501205)
    (('whether', 'it'), 0.45357727024066463)
    (('secretary', ','), 0.4532611653997023)
    (('but', 'court'), 0.4531141486877459)
    (('them', 'that'), 0.4528233670250188)
    (('and', 'upon'), 0.45238541204634686)
    (('language', 'the'), 0.4520533438107357)
    (('paid', 'the'), 0.4520533438107357)
    (('register', 'the'), 0.4520533438107357)
    (('the', 'subsequent'), 0.4520533438107357)
    (('why', 'the'), 0.4520533438107357)
    (('have', 'such'), 0.4516033288627632)
    (('deed', ','), 0.4514544086940724)
    (('officer', ','), 0.4514544086940724)
    (('party', 'it'), 0.45109200685107)
    (('not', 'where'), 0.45067259589449904)
    (('where', 'not'), 0.45067259589449904)
    (('decision', 'the'), 0.44987897257639986)
    (('execution', 'the'), 0.44987897257639986)
    (('proceedings', 'a'), 0.44958718733954584)
    (('only', 'by'), 0.4489307244321026)
    (('to', 'supreme'), 0.448106360890943)
    (('to', 'sustained'), 0.448106360890943)
    (('to', 'until'), 0.448106360890943)
    (('time', 'in'), 0.447657575238086)
    (('constitution', 'the'), 0.4464067806695944)
    (('remanded', 'the'), 0.4464067806695944)
    (('whether', 'court'), 0.44544022080232537)
    (('it', 'but'), 0.44530965425706626)
    (('state', 'that'), 0.4446339692116723)
    (('of', 'either'), 0.4440419080938618)
    (('appears', 'in'), 0.44381550894369326)
    (('could', 'in'), 0.44381550894369326)
    (('united', 'in'), 0.44381550894369326)
    ((',', 'consider'), 0.44380083615063626)
    ((',', 'president'), 0.44380083615063626)
    (('again', ','), 0.44380083615063626)
    (('constable', ','), 0.44380083615063626)
    (('criminal', ','), 0.44380083615063626)
    (('rule', ','), 0.44380083615063626)
    (('say', ','), 0.44380083615063626)
    (('jurors', ','), 0.4438008361506327)
    (('title', ','), 0.4438008361506327)
    (('words', ','), 0.4438008361506327)
    (('against', 'a'), 0.44376986074423286)
    (('court', 'in'), 0.4426790811252701)
    (('to', 'been'), 0.44265193079883147)
    (('.', 'assigned'), 0.441072661370594)
    (('al.', '.'), 0.441072661370594)
    (('by', '’'), 0.44029180615823194)
    (('account', 'the'), 0.43998051151015716)
    (('exception', 'the'), 0.43998051151015716)
    (('neither', 'the'), 0.43998051151015716)
    (('the', 'order'), 0.43998051151015716)
    (('and', 'consideration'), 0.43944635633884843)
    (('in', 'proceedings'), 0.43943706843311503)
    (('be', 'on'), 0.43761526246706595)
    (('have', 'for'), 0.43726997784873234)
    (('execution', 'of'), 0.4365732371900144)
    (('of', 'execution'), 0.4365732371900144)
    (('the', 'motion'), 0.43517952524633685)
    (('.', 'trial'), 0.4351233950394082)
    (('against', 'in'), 0.4336197418378056)
    (('due', 'to'), 0.4336067911958281)
    (('to', 'which'), 0.4336067911958281)
    (('which', 'to'), 0.4336067911958281)
    (('who', 'to'), 0.4336067911958281)
    (('to', 'common'), 0.43360679119582457)
    (('now', '.'), 0.4335975408084458)
    (('error', 'court'), 0.43305649634649157)
    (('same', 'that'), 0.4330148859448393)
    (('ever', 'the'), 0.43260098114456014)
    (('return', 'the'), 0.43260098114456014)
    (('the', 'gave'), 0.43260098114456014)
    (('his', 'have'), 0.43179484778258015)
    (('point', ','), 0.4315225063721968)
    ((':', 'is'), 0.42942550082059583)
    (('and', 'officer'), 0.42917802088501844)
    (('in', 'an'), 0.42854875229038214)
    (('on', 'defendant'), 0.4284358356926674)
    (('.', 'ante'), 0.42801650854514506)
    (('new', '.'), 0.42801650854514506)
    (('but', 'case'), 0.4274683414968763)
    (('question', 'it'), 0.4273877462598037)
    (('for', 'against'), 0.4253678887924437)
    (('a', '('), 0.4251341436150078)
    (('all', 'the'), 0.42403896764113824)
    (('commenced', 'the'), 0.42403896764113824)
    (('created', 'the'), 0.42403896764113824)
    (('equally', 'the'), 0.42403896764113824)
    (('filing', 'the'), 0.42403896764113824)
    (('lieutenant', 'the'), 0.42403896764113824)
    (('recovered', 'the'), 0.42403896764113824)
    (('satisfied', 'the'), 0.42403896764113824)
    (('the', 'constitutional'), 0.42403896764113824)
    (('the', 'permitted'), 0.42403896764113824)
    (('the', 'questions'), 0.42403896764113824)
    (('the', 'variance'), 0.42403896764113824)
    (('value', 'the'), 0.42403896764113824)
    (('while', 'the'), 0.42403896764113824)
    (('and', 'such'), 0.4238495012878296)
    (('plaintiffs', 'a'), 0.42359197880660204)
    (('of', 'statute'), 0.42298029256603087)
    (('and', 'are'), 0.42237284297990385)
    (('with', 'or'), 0.4220302134794167)
    (('absence', ','), 0.42108075965055036)
    (('maker', ','), 0.42108075965055036)
    (('supposed', ','), 0.42108075965055036)
    (('it', 'for'), 0.41948066440520293)
    (('of', 'consideration'), 0.4192952457581782)
    (('in', 'any'), 0.41867850826632136)
    (('although', 'the'), 0.41752566068246466)
    (('defense', 'the'), 0.41752566068246466)
    (('objection', 'the'), 0.41752566068246466)
    (('such', 'is'), 0.4168125042083979)
    (('as', 'one'), 0.4165101193316403)
    (('error', ','), 0.4157864599810388)
    (('but', 'by'), 0.4150658298700556)
    (('sheriff', ','), 0.41441702936953817)
    (('admitted', 'the'), 0.41398530297721337)
    (('the', 'whom'), 0.41398530297721337)
    (('used', 'the'), 0.41398530297721337)
    (('and', 'ought'), 0.4139112642317109)
    (('first', 'and'), 0.4139112642317109)
    ((',', 'are'), 0.4137882670841968)
    (('legislature', 'a'), 0.4136078902339797)
    (('error', 'was'), 0.41275239989140644)
    (('that', 'circuit'), 0.4126662405530368)
    (('a', 'case'), 0.4122639730050892)
    (('state', '.'), 0.4120497601400821)
    (('what', 'the'), 0.4117606378626988)
    (('to', 'same'), 0.41158048486582643)
    (('may', 'not'), 0.4115019986138222)
    (('time', '.'), 0.4086511836782165)
    ((',', 'order'), 0.40727496012551967)
    (('as', 'such'), 0.40717225475230023)
    (('constitution', 'that'), 0.4070196774118955)
    (('question', 'in'), 0.40701559074073757)
    (('jury', 'is'), 0.4057742327559488)
    (('either', 'to'), 0.40503763899905465)
    (('case', 'court'), 0.4045679393867978)
    (('law', 'that'), 0.4043700989979442)
    (('opinion', 'a'), 0.4036924213688984)
    (('statute', '.'), 0.40341453531206994)
    (('a', 'for'), 0.40259489751155897)
    (('not', 'for'), 0.4023303527155697)
    (('two', 'of'), 0.4022217323992372)
    (('upon', 'of'), 0.4022217323992372)
    (('are', 'of'), 0.40222173239923364)
    (('more', 'of'), 0.40222173239923364)
    (('of', 'entitled'), 0.40222173239923364)
    (('acted', 'the'), 0.4020126613111401)
    (('assumpsit', 'the'), 0.4020126613111401)
    (('bring', 'the'), 0.4020126613111401)
    (('building', 'the'), 0.4020126613111401)
    (('charge', 'the'), 0.4020126613111401)
    (('decide', 'the'), 0.4020126613111401)
    (('favor', 'the'), 0.4020126613111401)
    (('fill', 'the'), 0.4020126613111401)
    (('knowledge', 'the'), 0.4020126613111401)
    (('permit', 'the'), 0.4020126613111401)
    (('rejected', 'the'), 0.4020126613111401)
    (('sufficient', 'the'), 0.4020126613111401)
    (('the', 'contains'), 0.4020126613111401)
    (('the', 'course'), 0.4020126613111401)
    (('the', 'credit'), 0.4020126613111401)
    (('the', 'delivered'), 0.4020126613111401)
    (('the', 'errors'), 0.4020126613111401)
    (('the', 'form'), 0.4020126613111401)
    (('the', 'render'), 0.4020126613111401)
    (('true', 'the'), 0.4020126613111401)
    (('virtue', 'the'), 0.4020126613111401)
    (('person', 'not'), 0.40123366315999576)
    (('time', 'to'), 0.40118531350345066)
    (('for', ';'), 0.4007441018236193)
    (('.', 'form'), 0.4004306768732455)
    (('case', 'no'), 0.39889918930010637)
    (('and', 'him'), 0.39880437184149997)
    (('then', 'that'), 0.39859105534131345)
    (('be', 'that'), 0.39839919483954844)
    (('will', 'to'), 0.3979828814651043)
    (('*', ','), 0.3974079306661835)
    (('to', 'this'), 0.3973063538837103)
    (('office', ','), 0.3969144485198015)
    (('case', ','), 0.3966966010235957)
    (('evidence', 'the'), 0.39592533143222397)
    (('and', 'one'), 0.39505223698039416)
    (('and', 'taken'), 0.39505223698039416)
    (('has', 'an'), 0.3943234588623099)
    (('nor', 'a'), 0.3938446354125489)
    (('plaintiffs', 'to'), 0.39296480669847966)
    (('to', 'plaintiffs'), 0.39296480669847966)
    (('further', 'the'), 0.39267479673180006)
    (('the', 'further'), 0.39267479673180006)
    (('be', '('), 0.3922564523064409)
    ((',', "'"), 0.39133341625650075)
    ((',', 'do'), 0.39133341625650075)
    ((',', 'injunction'), 0.39133341625650075)
    ((',', 'offered'), 0.39133341625650075)
    ((',', 'trespass'), 0.39133341625650075)
    (('equity', ','), 0.39133341625650075)
    ((',', 'complainant'), 0.3913334162564972)
    ((',', 'forth'), 0.3913334162564972)
    ((',', 'full'), 0.3913334162564972)
    ((',', 'march'), 0.3913334162564972)
    ((',', 'nil'), 0.3913334162564972)
    (('article', ','), 0.3913334162564972)
    (('awarded', ','), 0.3913334162564972)
    (('full', ','), 0.3913334162564972)
    (('obligation', ','), 0.3913334162564972)
    (('present', ','), 0.3913334162564972)
    (('tobin', ','), 0.3913334162564972)
    (('costs', 'in'), 0.39107404687171865)
    (('considered', '.'), 0.3895423607305091)
    (('notice', '.'), 0.3895423607305091)
    (('in', 'of'), 0.3892909604491024)
    (('did', 'and'), 0.38882028326887763)
    (('certificate', 'the'), 0.3875130916160252)
    (('committed', 'the'), 0.3875130916160252)
    (('rights', 'the'), 0.3875130916160252)
    (('the', 'thereof'), 0.3875130916160252)
    (('the', 'void'), 0.3875130916160252)
    (('the', 'want'), 0.3875130916160252)
    (('that', 'error'), 0.38465186438343935)
    (('and', 'judgment'), 0.3841999589327294)
    (('which', 'it'), 0.38390910959292057)
    (('at', 'was'), 0.38376019655169813)
    (('was', 'at'), 0.38376019655169813)
    (('power', 'it'), 0.38318794234926656)
    (('as', 'by'), 0.3829985811107228)
    (('from', 'of'), 0.3826815460035924)
    (('authority', 'the'), 0.3819749080745929)
    (('it', 'under'), 0.38082092780534893)
    (('to', 'party'), 0.3804954547362627)
    (('of', 'proceedings'), 0.3801954260692355)
    (('.', 'no'), 0.3800820314814466)
    (('by', 's'), 0.38004526372137093)
    (('be', 'jury'), 0.379965046406344)
    ((',', 'interest'), 0.37926058395592577)
    (('common', ','), 0.37926058395592577)
    (('a', 'affirmed'), 0.37919785944814777)
    (('before', 'on'), 0.3789671594283668)
    (('in', 'record'), 0.3787957170932792)
    (('what', 'in'), 0.3787957170932792)
    (('that', 'be'), 0.3784996374018448)
    (('a', 'when'), 0.3778736785007659)
    (('and', 'for'), 0.37780729054646045)
    (('as', 'an'), 0.37647545752938427)
    (('land', 'of'), 0.37622652386628985)
    (('judgment', 'a'), 0.3750402339744916)
    (('.', 'very'), 0.3744354683403053)
    (('done', '.'), 0.3744354683403053)
    (('maker', '.'), 0.3744354683403053)
    ((',', 'so'), 0.37389882292047716)
    ((',', 'not'), 0.37336686972705024)
    (('said', ','), 0.37332762820218335)
    (('court', 'should'), 0.37268387836700967)
    (('it', 'if'), 0.3721749496268494)
    (('from', 'that'), 0.3716545238545663)
    (('it', 'to'), 0.37135451274276576)
    ((',', 'may'), 0.37115553431886994)
    (('jurisdiction', 'is'), 0.371008814595271)
    (('be', 'to'), 0.3708710358478662)
    (('not', 'him'), 0.3708600141164773)
    (('should', '.'), 0.3696975493336403)
    (('acts', 'the'), 0.36959118361876264)
    (('give', 'the'), 0.36959118361876264)
    (('lands', 'the'), 0.36959118361876264)
    (('proof', 'the'), 0.36959118361876264)
    (('the', 'two'), 0.36959118361876264)
    (('possession', 'the'), 0.3695911836187591)
    (('the', 'great'), 0.3695911836187591)
    (('suit', 'in'), 0.3690477405417205)
    (('make', '.'), 0.36800919918086805)
    (('their', 'court'), 0.3679614681246086)
    (('in', 'when'), 0.36772355959433867)
    (('fact', 'of'), 0.3665978226685169)
    (('that', 'parties'), 0.36627333506858406)
    ((',', 'require'), 0.3657983241493632)
    (('penalty', ','), 0.3657983241493632)
    (('purpose', ','), 0.3657983241493632)
    (('be', 'had'), 0.364962883067097)
    (('a', 'jurisdiction'), 0.36469828975303287)
    (('it', 'shall'), 0.36465199091184175)
    (('that', 'is'), 0.3636668392541047)
    (('clerk', 'to'), 0.36321746330443005)
    (('jurisdiction', 'to'), 0.36321746330443005)
    (('more', 'to'), 0.36321746330443005)
    (('to', 'clerk'), 0.36321746330443005)
    (('to', 'land'), 0.36321746330443005)
    (('to', 'present'), 0.36321746330443005)
    (('nor', 'to'), 0.3632174633044265)
    (('pay', 'to'), 0.3632174633044265)
    (('to', 'decree'), 0.3632174633044265)
    (('received', 'the'), 0.3626384229769961)
    (('decision', 'be'), 0.36247761967750236)
    (('of', 'executed'), 0.36157974790188874)
    (('by', 'record'), 0.3607035414203672)
    (('sale', ','), 0.35996924534716257)
    (('if', 'the'), 0.35857819089634546)
    (('new', ','), 0.3584687823508723)
    (('state', 'is'), 0.35733387769819913)
    (('himself', 'the'), 0.35509161392364774)
    (('passed', 'the'), 0.35509161392364774)
    (('the', 'reason'), 0.35509161392364774)
    (('the', 'sentence'), 0.35509161392364774)
    (('by', 'this'), 0.3546033418839514)
    (('do', 'in'), 0.35454817084660206)
    (('in', 'demurrer'), 0.35454817084660206)
    (('be', 'such'), 0.35447744183367647)
    (('and', 'suit'), 0.35441025248304925)
    (('case', 'for'), 0.3543336368666594)
    (('its', 'is'), 0.35393530123633)
    (('cause', 'a'), 0.3536431012440353)
    (('law', 'and'), 0.3530860715356674)
    (('was', 'under'), 0.35237978191192454)
    (('without', '.'), 0.35206765531184914)
    (('so', 'a'), 0.3516421369275875)
    (('it', 'have'), 0.3512514204720354)
    ((',', 'acting'), 0.3506914317591523)
    ((',', 'alleged'), 0.3506914317591523)
    ((',', 'collateral'), 0.3506914317591523)
    ((',', 'cornelius'), 0.3506914317591523)
    ((',', 'fully'), 0.3506914317591523)
    ((',', 'how'), 0.3506914317591523)
    ((',', 'irregularity'), 0.3506914317591523)
    ((',', 'provided'), 0.3506914317591523)
    (('alleged', ','), 0.3506914317591523)
    (('description', ','), 0.3506914317591523)
    (('facias', ','), 0.3506914317591523)
    (('sued', ','), 0.3506914317591523)
    (('to', 'right'), 0.35039342294684417)
    (('court', 'we'), 0.34922490554302144)
    (('state', 'on'), 0.3491725571680142)
    ((',', 'must'), 0.3490226113985351)
    (('a', 'without'), 0.3487567458840104)
    (('error', 'not'), 0.34849220108802115)
    (('to', 'at'), 0.34803105948975244)
    (('question', 'of'), 0.34777394837685804)
    (('s', '.'), 0.34763540899658807)
    (('in', 'at'), 0.346974951141803)
    (('absence', 'the'), 0.34687110711867675)
    (('directly', 'the'), 0.34687110711867675)
    (('done', 'the'), 0.34687110711867675)
    (('final', 'the'), 0.34687110711867675)
    (('the', 'final'), 0.34687110711867675)
    (('the', 'relief'), 0.34687110711867675)
    (('the', 'security'), 0.34687110711867675)
    (('by', 'he'), 0.3465096598376114)
    (('to', 'its'), 0.3461439499454855)
    (('or', 'which'), 0.34613009912015613)
    (('effect', 'of'), 0.3456382040328698)
    (('found', 'of'), 0.3456382040328698)
    (('of', 'taking'), 0.3456382040328698)
    (('a', 'from'), 0.34515810335738806)
    (('shall', 'in'), 0.3442798353927756)
    (('be', 'upon'), 0.3436391142580497)
    (('said', 'by'), 0.34315064774769866)
    (('the', 'set'), 0.34311897225757093)
    (('cases', 'of'), 0.34272072065057557)
    (('to', 'question'), 0.34168430175478903)
    (('a', 'brought'), 0.340851547798664)
    (('filed', ','), 0.34070734318652995)
    (('court', 'on'), 0.33994709195501116)
    (('and', 'other'), 0.33991068278793435)
    (('and', 'all'), 0.3399106827879308)
    (('of', 'person'), 0.33948597705127526)
    (('this', 'from'), 0.339428348049708)
    (('on', 'as'), 0.33900075211072433)
    (('.', 'will'), 0.33881155860958145)
    (('be', 'trial'), 0.33877335908623607)
    (('the', 'relation'), 0.33788232389142436)
    (('writing', 'the'), 0.33788232389142436)
    (('a', 'of'), 0.3378388296870902)
    (('is', 'said'), 0.3370614826719347)
    (('his', 'by'), 0.33645599517369007)
    (('.', 'was'), 0.3363003394535333)
    (('.', 'facts'), 0.3363003394535298)
    (('estate', '.'), 0.3363003394535298)
    (('not', 'record'), 0.33621387130958524)
    (('for', '.'), 0.33551647904799964)
    (('issue', 'the'), 0.3351576865548864)
    (('as', 'is'), 0.3350892403403307)
    (('himself', 'to'), 0.3340711176449105)
    (('to', 'plea'), 0.3340711176449105)
    (('not', 'that'), 0.3337706953812578)
    (('parties', '.'), 0.33368912599699385)
    (('its', ','), 0.3336179184002148)
    ((',', 'certainly'), 0.33361791840021127)
    (('appearance', ','), 0.33361791840021127)
    (('bar', ','), 0.33361791840021127)
    (('false', ','), 0.33361791840021127)
    (('over', ','), 0.33361791840021127)
    (('public', ','), 0.33361791840021127)
    (('their', ','), 0.33361791840021127)
    (('well', ','), 0.33361791840021127)
    (('the', 'due'), 0.33306530759364605)
    (('the', 'entitled'), 0.33306530759364605)
    (('at', 'judgment'), 0.33270159610901473)
    (('of', 'may'), 0.33183240450783913)
    (('therefore', ','), 0.33132610689222375)
    (('was', 'and'), 0.3309218995606784)
    (('”', 'and'), 0.3309218995606784)
    (('have', 'a'), 0.3301410677166672)
    (('decide', '.'), 0.330041348981851)
    (('land', ','), 0.32993287159235507)
    (('such', 'by'), 0.32979226489268143)
    (('to', 'defendants'), 0.3292701313810902)
    ((',', 'appear'), 0.3283236187306997)
    (('have', 'on'), 0.32829027202872396)
    (('such', 'it'), 0.32761461158730754)
    ((')', 'in'), 0.327508918094253)
    (('being', 'of'), 0.32725967471801454)
    (('justice', 'of'), 0.3262728791659377)
    (('as', 'will'), 0.32574465540845665)
    (('the', 'had'), 0.3253913797082255)
    (('one', 'on'), 0.3248796438660726)
    (('necessary', '.'), 0.324586918889743)
    (('note', 'it'), 0.32429425329570094)
    ((',', 'claim'), 0.32421922039796414)
    ((',', 'during'), 0.32421922039796414)
    (('means', ','), 0.32421922039796414)
    (('his', ','), 0.3242192203979606)
    (('rule', 'a'), 0.3240563052556844)
    (('judgment', 'of'), 0.3240270020190401)
    (('that', 'such'), 0.3235403500800551)
    (('case', 'can'), 0.322863298267567)
    (('to', 'third'), 0.3225754788070816)
    (('after', 'a'), 0.3220539523445396)
    (('then', 'a'), 0.3220539523445396)
    (('it', 'when'), 0.32098151925486107)
    (('power', 'is'), 0.32038274152530377)
    (('that', 'verdict'), 0.3195568361615564)
    ((',', 'possession'), 0.31826995406677483)
    (('testimony', ','), 0.31826995406677483)
    (('them', 'a'), 0.31739257497467577)
    (('clearly', 'the'), 0.31712376372462714)
    (('does', 'the'), 0.31712376372462714)
    (('injunction', 'the'), 0.31712376372462714)
    (('judge', 'the'), 0.31712376372462714)
    (('object', 'the'), 0.31712376372462714)
    (('practice', 'the'), 0.31712376372462714)
    (('the', 'personal'), 0.31712376372462714)
    (('erroneous', 'the'), 0.3171237637246236)
    (('guilty', 'the'), 0.3171237637246236)
    (('issued', 'the'), 0.3171237637246236)
    (('next', 'the'), 0.3171237637246236)
    (('remedy', 'the'), 0.3171237637246236)
    (('the', 'forth'), 0.3171237637246236)
    (('the', 'nil'), 0.3171237637246236)
    (('the', 'pleaded'), 0.3171237637246236)
    (('the', 'remedy'), 0.3171237637246236)
    (('as', 'his'), 0.3169744457807262)
    (('will', 'it'), 0.316576340134862)
    (('of', 'general'), 0.31606508864951977)
    (('and', 'under'), 0.3160639408335655)
    (('reversed', ','), 0.31604528895226025)
    ((',', 'has'), 0.31554634547819305)
    ((',', 'without'), 0.3153845630232013)
    ((',', 'very'), 0.31416555573403926)
    (('him', ','), 0.31416555573403926)
    (('did', 'to'), 0.3125913902344628)
    (('they', 'that'), 0.3125020786276025)
    (('execution', 'a'), 0.31158695329347097)
    (('had', 'as'), 0.3107424920692097)
    (('plaintiff', ','), 0.3105343052871703)
    (('maker', 'of'), 0.30911232800775323)
    (('paper', 'of'), 0.30911232800775323)
    (('received', 'a'), 0.3081147613866655)
    (('would', 'it'), 0.30780613050713157)
    (('not', 'by'), 0.30760113250150667)
    ((',', 'officer'), 0.3070644993588978)
    (('statute', 'be'), 0.3070644993588978)
    (('manner', 'the'), 0.30505093142405215)
    (('action', 'in'), 0.3039220977766348)
    (('and', 'which'), 0.3033848067628213)
    (('party', 'in'), 0.3014368343870437)
    (('to', 'evidence'), 0.3011564951721759)
    (('point', 'the'), 0.3007293254739558)
    (('to', 'a'), 0.299802487526204)
    (('or', 'have'), 0.29960023541771363)
    (('then', 'is'), 0.29921813152726173)
    (('1825', 'the'), 0.29850808555728037)
    (('that', 'governor'), 0.29849522063372547)
    (('a', 'before'), 0.29835579492433695)
    (('and', 'the'), 0.2972663453794411)
    (('after', 'and'), 0.2972663453794375)
    (('make', 'the'), 0.2955906021749861)
    (('upon', 'it'), 0.29474997768168265)
    (('the', 'must'), 0.2947125163043651)
    (('a', 'may'), 0.2943089618616348)
    ((')', 'such'), 0.29406205187628487)
    (('made', 'the'), 0.29272242023449024)
    (('and', 'them'), 0.2926049680095737)
    (('and', 'we'), 0.2926049680095737)
    (('sheriff', 'and'), 0.2926049680095737)
    (('defendant', 'court'), 0.29219396050664415)
    (('in', '?'), 0.2918124154986437)
    ((',', 'authorized'), 0.29179774270558667)
    ((',', 'declared'), 0.29179774270558667)
    ((',', 'follows'), 0.29179774270558667)
    ((',', 'himself'), 0.29179774270558667)
    ((',', 'ought'), 0.29179774270558667)
    ((',', 'required'), 0.29179774270558667)
    ((',', 'sold'), 0.29179774270558667)
    (('contended', ','), 0.29179774270558667)
    (('damages', ','), 0.29179774270558667)
    (('granted', ','), 0.29179774270558667)
    (('instrument', ','), 0.29179774270558667)
    (('justices', ','), 0.29179774270558667)
    (('nature', ','), 0.29179774270558667)
    (('other', ','), 0.29179774270558667)
    (('overruled', ','), 0.29179774270558667)
    (('oyer', ','), 0.29179774270558667)
    (('sold', ','), 0.29179774270558667)
    ((',', 'also'), 0.2917977427055831)
    ((',', 'otherwise'), 0.2917977427055831)
    ((',', 'reynolds'), 0.2917977427055831)
    ((',', 'use'), 0.2917977427055831)
    (('averment', ','), 0.2917977427055831)
    (('due', ','), 0.2917977427055831)
    (('executive', ','), 0.2917977427055831)
    (('force', ','), 0.2917977427055831)
    (('lie', ','), 0.2917977427055831)
    (('principles', ','), 0.2917977427055831)
    (('relied', ','), 0.2917977427055831)
    (('brought', '.'), 0.29169503733236724)
    (('considered', 'the'), 0.2915886716174896)
    (('seal', 'the'), 0.2915886716174896)
    (('the', 'witnesses'), 0.2915886716174896)
    (('of', 'testimony'), 0.29119042001049067)
    (('as', 'may'), 0.29097923724777885)
    (('was', 'plaintiff'), 0.28876368261595076)
    (('not', 'as'), 0.28811958310853925)
    (('counsel', 'to'), 0.2879293360001931)
    (('to', 'point'), 0.2879293360001931)
    ((')', 'law'), 0.28742895954383485)
    (('.', 'id.'), 0.28697262708996263)
    (('declaration', 'a'), 0.2866957777517598)
    (('record', ','), 0.2856716399087418)
    (('be', '.'), 0.28451488366611244)
    (('.', 'where'), 0.2843230486760149)
    (('been', 'to'), 0.2839541847797733)
    (('error', 'the'), 0.28386130959287925)
    (('are', 'this'), 0.283478334861087)
    (('into', 'and'), 0.283327154421567)
    (('motion', 'the'), 0.28317643180128726)
    (('new', 'to'), 0.2830471146204445)
    (('principle', 'to'), 0.2830471146204445)
    (('to', 'objection'), 0.2830471146204445)
    ((',', 'having'), 0.28115049850607576)
    ((';', 'court'), 0.281053402901442)
    (('on', 'he'), 0.2797917543375341)
    ((')', 'if'), 0.279728700862254)
    (('of', 'between'), 0.2793649846137036)
    (('of', 'nor'), 0.2793649846137036)
    (('there', 'on'), 0.27878322927661614)
    (('be', 'same'), 0.27808543238712957)
    (('that', 'demurrer'), 0.27773666046692824)
    (('received', 'to'), 0.2774875892785431)
    (('to', 'received'), 0.2774875892785431)
    (('and', 'could'), 0.27717492743996885)
    (('evidence', 'in'), 0.27654565884533255)
    (('the', 'appeal'), 0.27648177922728223)
    (('acting', 'the'), 0.2764817792272787)
    (('alleged', 'the'), 0.2764817792272787)
    (('clair', 'the'), 0.2764817792272787)
    (('description', 'the'), 0.2764817792272787)
    (('due', 'the'), 0.2764817792272787)
    (('erred', 'the'), 0.2764817792272787)
    (('how', 'the'), 0.2764817792272787)
    (('performance', 'the'), 0.2764817792272787)
    (('proved', 'the'), 0.2764817792272787)
    (('provided', 'the'), 0.2764817792272787)
    (('punishment', 'the'), 0.2764817792272787)
    (('receive', 'the'), 0.2764817792272787)
    (('the', 'decided'), 0.2764817792272787)
    (('the', 'demurred'), 0.2764817792272787)
    (('the', 'directed'), 0.2764817792272787)
    (('the', 'facias'), 0.2764817792272787)
    (('the', 'fee'), 0.2764817792272787)
    (('the', 'gatewood'), 0.2764817792272787)
    (('the', 'setting'), 0.2764817792272787)
    (('took', 'the'), 0.2764817792272787)
    (('a', 'error'), 0.275693283694288)
    (('of', 'cause'), 0.2756893264703031)
    (('are', 'on'), 0.2753846528147541)
    (('at', 'or'), 0.2750539371091989)
    (('i', 'of'), 0.2734884182770365)
    ((',', 'secretary'), 0.2726889197578828)
    (('for', 'it'), 0.27263927607593175)
    (('justice', 'is'), 0.27147314104435694)
    (('against', ','), 0.271333640145869)
    (('assigned', '.'), 0.2711476599282818)
    (('set', '.'), 0.2711476599282818)
    (('in', 'issue'), 0.2704839060581321)
    (('in', 'than'), 0.2704839060581321)
    (('made', 'in'), 0.2704839060581321)
    (('necessary', 'in'), 0.2704839060581321)
    (('error', 'this'), 0.269963528386608)
    (('true', ','), 0.269771436375585)
    (('can', 'on'), 0.26973808967360924)
    (('.', 'case'), 0.2691313143297762)
    (('the', 'only'), 0.2686622747679799)
    (('jurisdiction', 'the'), 0.2682141632436803)
    ((',', 'is'), 0.26816799632068467)
    (('are', 'in'), 0.26708532959626297)
    (('persons', 'the'), 0.26649769065465634)
    (('the', 'filed'), 0.26649769065465634)
    (('pleaded', ','), 0.2658025341726393)
    (('action', '.'), 0.26491570621676885)
    (('of', 'form'), 0.2647182086493025)
    (('statutes', 'of'), 0.26471820864929896)
    ((':', 'the'), 0.26450913756120187)
    (('.', 'an'), 0.2640114786466512)
    (('term', 'the'), 0.26388174244764784)
    (('to', 'time'), 0.26368178975351597)
    (('to', 'these'), 0.2636817897535124)
    (('to', 'two'), 0.2636817897535124)
    (('and', 'whether'), 0.26328940118501976)
    (('for', 'he'), 0.2632405780736846)
    (('be', 'before'), 0.26178362005802924)
    (('that', 'are'), 0.2606631471079872)
    (('reverse', ','), 0.2600888829782484)
    (('seem', 'the'), 0.25940826586833765)
    (('the', 'agent'), 0.25940826586833765)
    (('the', 'hundred'), 0.25940826586833765)
    (('tried', 'the'), 0.25940826586833765)
    (('of', 'a'), 0.25822166162372184)
    (('so', '.'), 0.25809150710283646)
    (('the', 'therefore'), 0.25711645436035013)
    (('declaration', 'it'), 0.25627582986704667)
    (('bill', ','), 0.2561738329748664)
    (('made', 'and'), 0.25584641799946084)
    (('as', 'has'), 0.2558341509668196)
    (('not', 'on'), 0.25538279669654074)
    (('to', 'decision'), 0.2549645726524048)
    (('costs', 'the'), 0.2541139661988261)
    (('having', 'to'), 0.25303454555400506)
    ((',', 'constable'), 0.2511557582082382)
    ((',', 'far'), 0.2511557582082382)
    ((',', 'grounds'), 0.2511557582082382)
    ((',', 'shows'), 0.2511557582082382)
    ((',', 'some'), 0.2511557582082382)
    (('acts', ','), 0.2511557582082382)
    (('however', 'to'), 0.250006852856437)
    ((')', 'case'), 0.2498694121691436)
    (('then', ','), 0.2491534052970934)
    (('it', 'at'), 0.2487026101622476)
    (('that', 'plea'), 0.24859031480740867)
    (('that', 'who'), 0.24859031480740867)
    (('or', 'of'), 0.248000813704369)
    (('but', 'the'), 0.24615724237047942)
    (('court', 'only'), 0.24597094374599848)
    (('court', 'so'), 0.24597094374599848)
    (('hubbard', '.'), 0.24515245139533803)
    (('more', '.'), 0.24515245139533803)
    (('taken', '.'), 0.24515245139533803)
    (('case', 'that'), 0.24513199503499905)
    (('of', 'action'), 0.2446804554127553)
    (('appointment', 'to'), 0.24457296680580853)
    (('award', 'to'), 0.24457296680580853)
    (('matter', 'to'), 0.24457296680580853)
    (('such', ','), 0.24449202792722957)
    (('whole', ','), 0.24449202792722602)
    (('title', 'the'), 0.24406030153490121)
    (('to', 'suit'), 0.24165548342351428)
    (('him', 'it'), 0.24116893747683932)
    (('said', 'this'), 0.24013230493550353)
    (('maker', 'the'), 0.23995590320216564)
    (('brought', 'to'), 0.2398350477991471)
    (('jurisdiction', ','), 0.23933032281144762)
    (('no', 'which'), 0.23885877678963752)
    (('of', 'ever'), 0.23872300011635517)
    (('of', 'mandamus'), 0.23872300011635517)
    (('of', 'our'), 0.23872300011635517)
    (('a', 'cause'), 0.23816588382409876)
    ((',', 'under'), 0.23757735170769934)
    (('case', 'not'), 0.23754148393635433)
    (('“', 'a'), 0.23596497563083219)
    (('being', 'the'), 0.2357354368839708)
    (('and', 'said'), 0.2355740229731964)
    (('that', 'must'), 0.23542585560896256)
    (('ex', ','), 0.2352142143392193)
    (('failed', ','), 0.2352142143392193)
    (('lieutenant', ','), 0.2352142143392193)
    (('payment', ','), 0.2352142143392193)
    (('recover', ','), 0.2352142143392193)
    (('taking', ','), 0.2352142143392193)
    (('on', 'they'), 0.23411417994288897)
    (('will', 'on'), 0.23411417994288897)
    (('“', 'on'), 0.23411417994288897)
    (('all', 'that'), 0.23264877093838976)
    (('is', 'upon'), 0.23194477685771986)
    (('it', 'had'), 0.23118484890421698)
    (('cases', '.'), 0.23050567543093337)
    (('have', 'in'), 0.22979313983865524)
    (('judgment', 'is'), 0.22934766537168016)
    ((',', 'appears'), 0.22906198735762473)
    (('to', 'opinion'), 0.2286753399256014)
    (('no', 'be'), 0.22859044133581108)
    (('be', 'they'), 0.2279619529612944)
    (('they', 'be'), 0.2279619529612944)
    (('would', 'a'), 0.22719476600310173)
    ((':', 'a'), 0.22719476600309818)
    (('a', 'one'), 0.22719476600309818)
    (('he', 'a'), 0.22719476600309818)
    (('page', 'a'), 0.22719476600309818)
    (('set', 'a'), 0.22719476600309818)
    (('that', 'power'), 0.22711058739695744)
    (('trial', 'court'), 0.22660561887906638)
    (('and', 'in'), 0.22611823072213255)
    (('power', 'the'), 0.22585570615731498)
    (('question', '.'), 0.223619289845697)
    (('shall', 'and'), 0.22272714341759325)
    (('constitution', '.'), 0.2224323748952557)
    ((',', 'v.'), 0.2220388255914827)
    (('of', 'their'), 0.22164948675741414)
    (('1', 'this'), 0.22146501069541813)
    ((',', 'notes'), 0.2214084148141886)
    (('notes', ','), 0.2214084148141886)
    (('a', 'on'), 0.21884742328957785)
    (('it', 'error'), 0.21880112444838318)
    (('or', 'been'), 0.2176717149732319)
    (('sold', 'the'), 0.21758809017371306)
    (('the', 'assigned'), 0.21758809017371306)
    (('the', 'both'), 0.21758809017371306)
    (('the', 'committed'), 0.21758809017371306)
    (('the', 'damages'), 0.21758809017371306)
    (('the', 'given'), 0.21758809017371306)
    (('the', 'overruled'), 0.21758809017371306)
    (('thereof', 'the'), 0.21758809017371306)
    (('time', 'the'), 0.21758809017371306)
    (('coles', 'the'), 0.2175880901737095)
    (('otherwise', 'the'), 0.2175880901737095)
    (('relied', 'the'), 0.2175880901737095)
    (('the', 'force'), 0.2175880901737095)
    (('validity', 'the'), 0.2175880901737095)
    (('wilson', 'the'), 0.2175880901737095)
    (('any', 'in'), 0.21704464709667093)
    (('an', 'been'), 0.21672226114436555)
    (('them', 'to'), 0.21637607497515887)
    (('plea', '.'), 0.21600610573581847)
    (('of', ','), 0.21569476361979412)
    (('a', 'should'), 0.21532066571480613)
    ((',', 'warrant'), 0.21517646110267208)
    (('legislature', ','), 0.21517646110267208)
    (('of', 'new'), 0.21513617979874056)
    (('case', 'have'), 0.21476561121322746)
    (('been', 'court'), 0.21293068198199094)
    (('and', 'plaintiff'), 0.2118058570402468)
    ((',', 'objection'), 0.21162739402160113)
    (('objection', ','), 0.21162739402160113)
    ((';', 'which'), 0.21084440062004006)
    (('cause', 'by'), 0.2099235892447595)
    (('below', '.'), 0.2097471152641397)
    (('manner', 'of'), 0.20957665445683915)
    ((',', 'smith'), 0.20933558251361362)
    (('for', 'his'), 0.20879279405130902)
    (('person', 'is'), 0.208737385696395)
    (('or', 'on'), 0.20862657537022145)
    ((',', 'necessary'), 0.2077334779171096)
    (('error', 'to'), 0.2070982613871486)
    (('to', 'good'), 0.2070982613871486)
    (('should', 'in'), 0.20517054680837887)
    (('did', 'the'), 0.2050971459905142)
    (('.', 'yet'), 0.20451046689798957)
    (('some', '.'), 0.20451046689798957)
    (('by', 'for'), 0.20442760148990757)
    (('so', 'in'), 0.203988494271222)
    (('.', 'the'), 0.2037995015481293)
    (('court', 'for'), 0.2031016819734539)
    (('must', '.'), 0.20284164653737236)
    (('or', 'has'), 0.2026278347487782)
    (('who', 'in'), 0.20254507740155603)
    (('affirmed', 'and'), 0.20240715903799966)
    (('below', 'and'), 0.20240715903799966)
    (('taken', 'and'), 0.2024071590379961)
    (('of', '’'), 0.20238115305263804)
    (('a', 'second'), 0.2011995574701544)
    (('of', '”'), 0.20058787122958321)
    (('be', ';'), 0.20057606516621362)
    (('in', 'a'), 0.20027976461726738)
    (('this', 'defendant'), 0.19949032043815507)
    (('interest', ','), 0.19868833831410626)
    ((',', 'become'), 0.1986883383141027)
    ((',', 'directly'), 0.1986883383141027)
    ((',', 'might'), 0.1986883383141027)
    ((',', 'too'), 0.1986883383141027)
    ((',', 'whom'), 0.1986883383141027)
    (('become', ','), 0.1986883383141027)
    (('examination', ','), 0.1986883383141027)
    (('intention', ','), 0.1986883383141027)
    (('relief', ','), 0.1986883383141027)
    (('security', ','), 0.1986883383141027)
    (('used', ','), 0.1986883383141027)
    (('years', ','), 0.1986883383141027)
    (('appointment', 'the'), 0.19847926722600917)
    (('county', 'in'), 0.1984289689293206)
    (('before', 'it'), 0.19839491378654728)
    (('when', 'the'), 0.1975966149862458)
    (('suit', 'it'), 0.19677481811838504)
    (('supreme', 'the'), 0.19556178384371137)
    (('the', 'sustained'), 0.19556178384371137)
    (('court', 'was'), 0.1948967591517281)
    (('be', 'and'), 0.1947127672479958)
    (('the', 'person'), 0.1943806990123882)
    (('such', 'not'), 0.19427128239315294)
    (('of', 'writ'), 0.19363511058782024)
    (('law', 'was'), 0.1930046998105226)
    (('the', '’'), 0.19224524205925064)
    (('is', 'in'), 0.19209071976438707)
    (('on', 'that'), 0.19200678644104485)
    (('stated', 'the'), 0.1915928816407657)
    (('it', 'made'), 0.19132038802627704)
    (('is', 'so'), 0.1913027923603714)
    (('present', 'in'), 0.19104943856372714)
    (('of', 'any'), 0.19071762720552599)
    (('.', 'p.'), 0.19001089720287467)
    (('be', 'cause'), 0.18952087665721606)
    (('cause', 'be'), 0.18952087665721606)
    (('to', 'parties'), 0.1887197320722933)
    (('was', '“'), 0.18859952069052)
    (('“', 'was'), 0.18859952069052)
    (('authority', 'of'), 0.18809692704638792)
    (('filed', 'of'), 0.18809692704638792)
    (('three', 'of'), 0.18809692704638792)
    (('the', 'new'), 0.18804381455969832)
    ((',', 'day'), 0.18746108289085228)
    ((',', 'motion'), 0.18746108289085228)
    (('motion', ','), 0.18746108289085228)
    (('court', 'upon'), 0.18707725469243286)
    (('as', 'the'), 0.18620767553393947)
    (('an', 'had'), 0.1861973941309074)
    (('before', 'and'), 0.18610534670889933)
    (('the', 'inquiry'), 0.18587923044637478)
    ((',', 'hubbard'), 0.18488253878907202)
    (('clerk', ','), 0.18488253878907202)
    (('in', 'question'), 0.1846231694042899)
    (('?', 'be'), 0.18292819934679017)
    (('from', 'and'), 0.1828669726423513)
    (('action', 'and'), 0.18236940580145244)
    (('it', 'were'), 0.18227524842327014)
    (('who', 'it'), 0.18227524842327014)
    (('another', ','), 0.18076643031684014)
    (('counsel', 'the'), 0.1804350917562445)
    ((',', 'were'), 0.17932301344717416)
    (('no', 'judgment'), 0.17881139311969818)
    (('would', 'that'), 0.17820098691601416)
    (('as', 'law'), 0.1781467410834061)
    (('or', 'be'), 0.17759967974959778)
    (('referred', 'the'), 0.1769461056763646)
    (('some', 'the'), 0.1769461056763646)
    (('the', 'acts'), 0.1769461056763646)
    (('the', 'per'), 0.1769461056763646)
    (('third', 'the'), 0.1769461056763646)
    (('second', 'and'), 0.17641195050505232)
    (('(', 'an'), 0.17535583448347936)
    (('their', 'in'), 0.1739759252047861)
    (('jury', 'of'), 0.17395274472611888)
    (('of', 'fact'), 0.17395274472611888)
    (('of', 'process'), 0.17395274472611888)
    (('process', 'of'), 0.17395274472611888)
    ((',', 'considered'), 0.17315324620696515)
    ((',', 'purple'), 0.17315324620696515)
    ((',', 'show'), 0.17315324620696515)
    (('appointment', ','), 0.17315324620696515)
    (('particular', ','), 0.17315324620696515)
    (('think', ','), 0.17315324620696515)
    (('decree', 'the'), 0.17273385438945255)
    (('there', 'in'), 0.17094823250721447)
    (('trial', 'of'), 0.17089618629277936)
    (('interest', 'to'), 0.170572385362032)
    (('land', 'to'), 0.170572385362032)
    (('same', 'to'), 0.170572385362032)
    (('to', 'manner'), 0.170572385362032)
    ((':', 'it'), 0.17030260675719333)
    (('v.', '('), 0.17030260675719333)
    (('a', 'after'), 0.17005085889949)
    (('court', ','), 0.1693616670445941)
    (('have', 'his'), 0.1687604419487876)
    (('note', 'is'), 0.16837964808025419)
    (('court', '“'), 0.16826163264826022)
    (('with', '.'), 0.16637533801121762)
    (('a', 'law'), 0.1663695733070547)
    ((',', 'every'), 0.16626686062172524)
    ((',', 'ground'), 0.16626686062172524)
    (('being', 'and'), 0.16541295155579405)
    (('defendant', 'on'), 0.16540142985887485)
    (('.', 'court'), 0.16536071314283518)
    (('cause', ','), 0.16526533677665256)
    (('demurrer', 'the'), 0.165120670279574)
    (('defense', '.'), 0.1649821027113525)
    (('case', 'but'), 0.16443393566308373)
    (('which', 'case'), 0.16443393566308373)
    (('is', 'this'), 0.1640896056707568)
    (('of', 'counsel'), 0.16343487281211821)
    (('which', 'can'), 0.16282288575709813)
    (('said', 'judgment'), 0.1619375745553029)
    (('not', 'only'), 0.16140664848752806)
    ((',', 'writ'), 0.16121363289544277)
    (('circumstances', 'the'), 0.16100456180734568)
    (('even', 'the'), 0.16100456180734568)
    (('justice', 'the'), 0.16100456180734568)
    (('manifest', 'the'), 0.16100456180734568)
    (('pass', 'the'), 0.16100456180734568)
    (('taking', 'the'), 0.16100456180734568)
    (('the', 'answer'), 0.16100456180734568)
    (('the', 'pleas'), 0.16100456180734568)
    (('the', 'says'), 0.16100456180734568)
    (('the', 'taking'), 0.16100456180734568)
    (('york', 'the'), 0.16100456180734568)
    (('this', 'it'), 0.15978061063618654)
    (('writ', '.'), 0.1594225773694511)
    (('that', 'an'), 0.15909216396831027)
    (('the', 'cases'), 0.15808707842505143)
    (('the', 'was'), 0.15696653853113318)
    (('was', 'the'), 0.15696653853113318)
    (('bill', '.'), 0.1543869874721544)
    ((',', 'much'), 0.15429421895565198)
    (('cases', ','), 0.15429421895565198)
    (('proceedings', ','), 0.15429421895564843)
    (('their', 'it'), 0.15370609622650022)
    (('case', '.'), 0.15365409690983967)
    (('the', 'rendered'), 0.15249306195182655)
    (('he', 'it'), 0.15168692858984656)
    (('it', 'he'), 0.15168692858984656)
    (('circuit', ','), 0.1507538612504007)
    (('but', 'be'), 0.1502443896160699)
    (('cause', 'of'), 0.1501584443864452)
    (('and', 'by'), 0.14913236599565138)
    (('by', 'will'), 0.14882899199791666)
    (('be', 'one'), 0.14880241544216588)
    (('1', '('), 0.1482763004271952)
    (('has', 'for'), 0.14765249923740242)
    (("'", 'the'), 0.147198762282315)
    (('consideration', 'the'), 0.147198762282315)
    (('held', 'the'), 0.147198762282315)
    (('john', 'the'), 0.147198762282315)
    (('personal', 'the'), 0.147198762282315)
    (('the', 'grant'), 0.147198762282315)
    (('should', 'to'), 0.1467256434076667)
    (('that', 'statute'), 0.14649212718867588)
    (('ill.', 'a'), 0.14605800327769103)
    (('give', '.'), 0.1456167778444204)
    (('law', '.'), 0.14230404380358763)
    (('trial', 'not'), 0.1420413236205924)
    (('he', 'been'), 0.14127970552668145)
    (('the', 'warrant'), 0.14096680857079846)
    (('legislature', 'in'), 0.14042336549375634)
    (('that', 'one'), 0.14023313671699356)
    (('that', 'taken'), 0.14023313671699356)
    ((',', 'passed'), 0.13979464926053708)
    (('reason', ','), 0.13979464926053708)
    (('commission', 'of'), 0.13918732656544108)
    (('costs', 'of'), 0.13918732656544108)
    (('of', 'appeal'), 0.13918732656544108)
    (('of', 'commission'), 0.13918732656544108)
    (('some', 'of'), 0.13918732656544108)
    (('if', 'court'), 0.13897134455373816)
    (('then', 'the'), 0.1384178767401032)
    (('and', 'general'), 0.13827682161828037)
    (('no', 'was'), 0.13676058917090117)
    (('right', 'not'), 0.13564355242244375)
    (('or', 'it'), 0.13496953364491304)
    (('people', '.'), 0.13496953364491304)
    (('bill', 'to'), 0.13494847563131174)
    (('jury', 'to'), 0.13494847563131174)
    (('to', 'following'), 0.13494847563131174)
    (('that', 'this'), 0.13494778896212978)
    (('for', 'judgment'), 0.13424584068625478)
    (('and', 'must'), 0.1341011456470902)
    (('that', 'if'), 0.13363543448257076)
    (('the', 'should'), 0.13234080355520206)
    ((',', 'said'), 0.13231952869838892)
    (('?', 'a'), 0.13203753296275877)
    (('action', 'it'), 0.1316491753533029)
    (('one', 'the'), 0.13071063949374562)
    (('an', 'or'), 0.13062406336895194)
    (('be', 'error'), 0.13018673727481556)
    (('have', 'or'), 0.12967523397540148)
    (('act', 'is'), 0.1291202176330266)
    (('any', 'judgment'), 0.1287707106201026)
    ((',', '1.'), 0.12829901042270464)
    ((',', 'founded'), 0.12829901042270464)
    ((',', 'guilty'), 0.12829901042270464)
    (('doctrine', ','), 0.12829901042270464)
    (('mandamus', ','), 0.12829901042270464)
    (('our', ','), 0.12829901042270464)
    (('stated', ','), 0.12829901042270464)
    (('a', 'would'), 0.1276590924521841)
    (('below', 'a'), 0.1276590924521841)
    (('it', 'a'), 0.12754251574484599)
    (('.', 'considered'), 0.1265079548967165)
    (('award', '.'), 0.1265079548967165)
    (('a', 'point'), 0.12591143016591744)
    (('a', 'reversed'), 0.12591143016591744)
    (('a', 'see'), 0.12591143016591744)
    (('out', 'a'), 0.12591143016591744)
    (('rep.', 'a'), 0.12591143016591744)
    (('state', 'or'), 0.12456231058174794)
    (('paper', 'the'), 0.12447868578222909)
    (('plaintiffs', 'the'), 0.12447868578222909)
    (('properly', 'the'), 0.12447868578222909)
    (('the', 'diligence'), 0.12447868578222909)
    (('the', 'examination'), 0.12447868578222909)
    (('the', 'properly'), 0.12447868578222909)
    (('up', 'the'), 0.12447868578222909)
    (('very', 'the'), 0.12447868578222909)
    (('.', 'affirmed'), 0.12359047151442226)
    (('suit', '.'), 0.12359047151442226)
    ((',', 'what'), 0.12340021100986576)
    (('was', 'he'), 0.12324578269642217)
    ((',', 'was'), 0.12201274757527258)
    (('two', ','), 0.12187274126327452)
    (('to', 'by'), 0.12181307344217629)
    (('to', 'act'), 0.12132886634218565)
    (('cause', 'the'), 0.1208030276388321)
    (('was', 'defendant'), 0.11988677060650588)
    (('is', 'but'), 0.1194700475993109)
    (('be', 'record'), 0.11790840749637965)
    (('execution', 'to'), 0.11746104890247011)
    (('judgment', 'as'), 0.11689661033181054)
    ((',', 'plaintiffs'), 0.11622617812212965)
    (('subject', ','), 0.11622617812212965)
    (('in', 'point'), 0.11576131125948663)
    ((',', 'all'), 0.11491998062150444)
    (('the', 'action'), 0.11449459720960675)
    (('the', 'could'), 0.11421035032840621)
    (('can', 'by'), 0.11406357383724242)
    (('governor', '.'), 0.11390791811708567)
    (('&', '.'), 0.11390791811708212)
    (('but', 'a'), 0.1131595227570692)
    (('plea', 'to'), 0.11167869630846283)
    ((',', 'agent'), 0.11122549706376361)
    ((',', 'appearance'), 0.11122549706376361)
    ((',', 'bar'), 0.11122549706376361)
    ((',', 'plaintiff'), 0.11122549706376361)
    ((',', 'rendered'), 0.11122549706376361)
    ((',', 'well'), 0.11122549706376361)
    (('year', ','), 0.11122549706376361)
    (('reynolds', 'the'), 0.1106728862571984)
    (('of', 'decree'), 0.10943998317139148)
    (('from', 'judgment'), 0.10923052422445778)
    (('a', 'at'), 0.10919755660464858)
    (('were', '.'), 0.10909090181930736)
    ((';', 'case'), 0.10785040729671636)
    (('court', 'any'), 0.10702400312639426)
    (('an', 'in'), 0.1066206574030204)
    (('declaration', 'in'), 0.1066206574030204)
    (('testimony', 'the'), 0.10655677778496653)
    (('said', 'on'), 0.10650774080530567)
    (('action', ','), 0.10624208954950731)
    (('entered', 'of'), 0.10523999464210476)
    (('of', 'motion'), 0.10523999464210476)
    (('so', ','), 0.10471219010509003)
    (('said', 'or'), 0.10428991555548706)
    (('*', 'that'), 0.1032389292347915)
    (('being', 'that'), 0.1032389292347915)
    (('is', 'reversed'), 0.10307560934863602)
    (('record', 'is'), 0.10307560934863602)
    (('law', 'to'), 0.10282777872619775)
    (('judgment', 'can'), 0.1027755020871588)
    (('not', 'upon'), 0.10251295943395888)
    (('judgment', 'in'), 0.10185570923427179)
    (('case', 'be'), 0.10169818031512179)
    (('also', 'a'), 0.1016638839192403)
    (('and', 'what'), 0.10112382320081537)
    (('point', 'and'), 0.10112382320081537)
    (('record', 'and'), 0.10112382320081537)
    (('of', 'upon'), 0.09965896237880756)
    ((',', 'application'), 0.09915266476318862)
    ((',', 'never'), 0.09915266476318862)
    ((',', 'overruled'), 0.09915266476318862)
    (('am', ','), 0.09915266476318862)
    (('committed', ','), 0.09915266476318862)
    (('decided', ','), 0.09915266476318862)
    (('declared', ','), 0.09915266476318862)
    (('the', 'evidence'), 0.09894359367509509)
    (('the', 'take'), 0.09894359367509153)
    (('witnesses', 'the'), 0.09894359367509153)
    (('all', 'and'), 0.09890258328413637)
    (('not', 'whether'), 0.09784151971005883)
    (('record', 'to'), 0.09528425805779506)
    (('legal', ','), 0.09435167849937187)
    (('this', ','), 0.0942163983604054)
    (('and', 'party'), 0.0941542683859744)
    (('that', 'made'), 0.09413672212754065)
    (('on', 'or'), 0.0931493579502849)
    (('order', 'the'), 0.09205720808985163)
    (('of', 'whole'), 0.09188161178708398)
    (('.', 'when'), 0.09167797073361683)
    (('him', 'in'), 0.09151376501281305)
    (('in', 'due'), 0.0915137650128095)
    (('is', 'its'), 0.09090089540253743)
    (('this', 'been'), 0.0904796985194487)
    (('this', 'made'), 0.0904796985194487)
    ((',', 'most'), 0.09016388153593624)
    (('character', ','), 0.09016388153593624)
    (('facts', ','), 0.09016388153593624)
    (('presented', '.'), 0.08903324947805658)
    (('power', 'of'), 0.08856125349547384)
    (('to', '”'), 0.0875830206910031)
    (('of', 'present'), 0.08671990667130558)
    (('state', 'to'), 0.0865081205735585)
    (('the', 'has'), 0.0860584674684084)
    (('has', 'in'), 0.08551502439136627)
    (('which', 'the'), 0.08548455416636713)
    (('which', 'by'), 0.0849172281777264)
    (('be', '”'), 0.08467207802245014)
    (('no', 'a'), 0.08459037056029928)
    (('“', 'not'), 0.0836973373897898)
    ((',', 'questions'), 0.08321112089416971)
    (('writ', ','), 0.08321112089416971)
    (('no', 'on'), 0.08273957487235606)
    (('made', ','), 0.08220259583325173)
    (('one', 'is'), 0.08150219740028675)
    (('which', 'this'), 0.08143455891644535)
    (('a', 'with'), 0.08060226839140938)
    (('of', 'time'), 0.08029363751187191)
    (('assigned', 'the'), 0.08008456642377837)
    (('entry', 'the'), 0.08008456642377837)
    (('form', 'the'), 0.08008456642377837)
    (('means', 'the'), 0.08008456642377837)
    (('oath', 'the'), 0.08008456642377837)
    (('the', 'means'), 0.08008456642377837)
    (('the', 'oath'), 0.08008456642377837)
    (('proceedings', 'the'), 0.08008456642377482)
    (('the', 'true'), 0.08008456642377482)
    (('to', 'on'), 0.07894190989646432)
    (('law', 'in'), 0.0782169423993544)
    ((',', 'making'), 0.07767293735273739)
    (('right', 'the'), 0.07688922726396896)
    (('and', 'also'), 0.07687627695413823)
    (('of', 'and'), 0.07687627695413823)
    (('rule', 'and'), 0.07687627695413823)
    (('after', 'is'), 0.07682571019081408)
    (('is', 'after'), 0.07682571019081408)
    (('to', 'so'), 0.07659623715301933)
    (('”', 'that'), 0.07610279929727781)
    (('of', 's'), 0.07586139119661084)
    (('act', 'the'), 0.07523516676237918)
    (('one', ','), 0.07485975146124346)
    (('.', 'on'), 0.07453367978294168)
    (('a', 'made'), 0.07418314749713062)
    (('defendant', 'is'), 0.07402707683814214)
    (('office', 'is'), 0.07402707683814214)
    ((',', '1819'), 0.07315745623024483)
    (('those', ','), 0.07315745623024483)
    (('?', 'the'), 0.07239017463377806)
    (('the', '?'), 0.07239017463377806)
    (('of', 'his'), 0.07207313070690446)
    (('before', 'was'), 0.07041809434220525)
    (('demurrer', ','), 0.06940532136913546)
    (('of', 'i'), 0.06703754080960778)
    (('act', ','), 0.06698265910227974)
    (('not', '.'), 0.0667212326726414)
    ((',', 'proper'), 0.06623804229045405)
    ((',', 'take'), 0.06623804229045405)
    (('money', ','), 0.06623804229045405)
    (('entered', 'to'), 0.06623572554729762)
    (('said', 'to'), 0.06623572554729762)
    (('are', 'for'), 0.06618839860850656)
    (('’', 'v.'), 0.06614537546590071)
    (('authorities', 'the'), 0.06558499672866347)
    (('granted', 'the'), 0.06558499672866347)
    (('the', 'were'), 0.06558499672866347)
    (('be', 'circuit'), 0.0650917090529326)
    (('be', 'their'), 0.0650917090529326)
    (('parties', ','), 0.06483259157931087)
    (('trial', 'this'), 0.06351265091917924)
    (('v.', '.'), 0.06256103811686486)
    (('only', 'a'), 0.06213551973260323)
    (('statute', 'to'), 0.06204792858386554)
    (('form', ','), 0.061184814564171575)
    (('of', 'first'), 0.06118481456416802)
    (('of', 'notice'), 0.06118481456416802)
    (('without', ','), 0.06012750778112519)
    (('statute', ','), 0.05979023249241422)
    (('however', 'the'), 0.05952324394145947)
    (('appears', ','), 0.05913698591531258)
    (('not', 'he'), 0.058985583893036875)
    (('and', 'was'), 0.058842354123875396)
    ((',', 'have'), 0.05833645225513706)
    (('as', 'this'), 0.05758781696207649)
    (('in', 'office'), 0.05756643308947318)
    (('not', 'at'), 0.056465591914520274)
    (('cause', 'it'), 0.055742842494339584)
    (('from', 'a'), 0.055651486162403785)
    ((';', 'on'), 0.054725198702758604)
    (('on', ';'), 0.054725198702758604)
    (('a', 'them'), 0.054358169140883206)
    (('laws', 'a'), 0.054358169140883206)
    (('cited', 'the'), 0.054089357890831025)
    (('existence', 'the'), 0.054089357890831025)
    (('johns.', 'the'), 0.054089357890831025)
    (('the', 'debet'), 0.054089357890831025)
    (('the', 'edward'), 0.054089357890831025)
    (('the', 'hand'), 0.054089357890831025)
    (('the', 'issued'), 0.054089357890831025)
    (('the', 'tobin'), 0.054089357890831025)
    (('and', 'brought'), 0.05302953499977292)
    (('ill.', '.'), 0.05250737345294354)
    (('interest', '.'), 0.05250737345293999)
    (('1', ','), 0.05241460493643757)
    (('this', 'law'), 0.05021582830572413)
    (('that', 'a'), 0.049467672793809925)
    (('record', 'the'), 0.04919055847799214)
    (('his', 'not'), 0.048931919229112)
    (('the', 's'), 0.0487690259575686)
    (('note', ','), 0.04852659169322138)
    (('as', 'been'), 0.04848560985482209)
    (('case', 'as'), 0.04747778931723445)
    (('taken', ','), 0.04737901503913733)
    (('to', 'as'), 0.04718996985674906)
    (('affirmed', 'in'), 0.04711964565435878)
    (('be', 'we'), 0.04635514647134542)
    (('for', 'that'), 0.04617259323223166)
    (('argument', 'of'), 0.04607792217396067)
    (('this', 'they'), 0.04581064918572153)
    (('case', 'an'), 0.04578943916446221)
    (('it', 'one'), 0.044771724673335456)
    (('plaintiff', '.'), 0.04304704420387395)
    (('who', 'a'), 0.04277019486567113)
    (('of', 'does'), 0.042325787312851304)
    (('of', 'one'), 0.042325787312851304)
    ((',', 'had'), 0.04114706132762436)
    (('same', 'is'), 0.04086021290294184)
    (('that', 'his'), 0.04069746316607947)
    (('that', 'suit'), 0.04069746316607592)
    (('can', 'in'), 0.03904634511867755)
    (('note', 'not'), 0.03894783065648966)
    (('upon', 'and'), 0.03734791276750471)
    (('days', 'the'), 0.037015844531889996)
    (('hobson', 'the'), 0.037015844531889996)
    (('points', 'the'), 0.037015844531889996)
    (('well', 'the'), 0.037015844531889996)
    ((',', 'upon'), 0.03654068746351058)
    (('legislature', 'of'), 0.03609383360133833)
    (('courts', 'to'), 0.03306886161209732)
    (('form', 'to'), 0.03306886161209732)
    (('it', 'as'), 0.0328976243850434)
    (('but', 'in'), 0.03262007595924388)
    (('suit', 'and'), 0.03248215759568751)
    (('.', 'a'), 0.031990356112427065)
    (('this', 'statute'), 0.03180379119184451)
    (('is', 'we'), 0.03152234832360534)
    (('this', 'be'), 0.02934604776799077)
    ((',', 'with'), 0.02876333687179411)
    (('make', ','), 0.02876333687179411)
    ((',', 'appeared'), 0.028763336871790557)
    ((',', 'ascertain'), 0.028763336871790557)
    ((',', 'competent'), 0.028763336871790557)
    ((',', 'each'), 0.028763336871790557)
    ((',', 'erred'), 0.028763336871790557)
    ((',', 'his'), 0.028763336871790557)
    ((',', 'object'), 0.028763336871790557)
    ((',', 'private'), 0.028763336871790557)
    ((',', 'proof'), 0.028763336871790557)
    ((',', 'recover'), 0.028763336871790557)
    ((',', 'support'), 0.028763336871790557)
    (('commission', ','), 0.028763336871790557)
    (('doubt', ','), 0.028763336871790557)
    (('far', ','), 0.028763336871790557)
    (('granting', ','), 0.028763336871790557)
    (('john', ','), 0.028763336871790557)
    (('judge', ','), 0.028763336871790557)
    (('otherwise', ','), 0.028763336871790557)
    (('personal', ','), 0.028763336871790557)
    (('pleadings', ','), 0.028763336871790557)
    (('possession', ','), 0.028763336871790557)
    (('presented', ','), 0.028763336871790557)
    (('question', ','), 0.028763336871790557)
    (('referred', ','), 0.028763336871790557)
    (('sustained', ','), 0.028763336871790557)
    (('third', ','), 0.028763336871790557)
    (('witness', ','), 0.028763336871790557)
    (('contract', 'the'), 0.028554265783693467)
    (('the', 'taken'), 0.027617146529639314)
    (('contract', '.'), 0.02697228134580243)
    (('not', 'under'), 0.025085177274746684)
    (('court', 'with'), 0.024971757709419506)
    (('also', 'the'), 0.02494301223131501)
    (('declared', 'the'), 0.02494301223131501)
    (('the', 'also'), 0.02494301223131501)
    (('the', 'part'), 0.02494301223131501)
    (('before', 'is'), 0.02455840057384151)
    (('received', 'of'), 0.023710109145508085)
    (('of', 'all'), 0.023710109145504532)
    (('of', 'presented'), 0.023710109145504532)
    (('presented', 'of'), 0.023710109145504532)
    (('a', 'an'), 0.023661371917967244)
    (('.', 'decree'), 0.022760030058886827)
    (('this', 'but'), 0.02254086986287618)
    (('the', 'entered'), 0.020142025967494703)
    (('before', 'a'), 0.018247875731599805)
    (('court', 'it'), 0.018018997067649423)
    (('plea', 'and'), 0.01798258790057261)
    (('question', 'the'), 0.017289439687878883)
    (('trial', 'the'), 0.017289439687878883)
    (('was', 'one'), 0.016330578779907512)
    (('inquiry', 'the'), 0.015954229004062626)
    (('the', 'here'), 0.015954229004062626)
    (('governor', 'the'), 0.015954229004059073)
    (('”', 'the'), 0.015954229004059073)
    (('not', 'it'), 0.015916862001152055)
    (('to', 'for'), 0.014296445431568827)
    (('plaintiff', 'a'), 0.014201042668897657)
    (('of', 'rule'), 0.013656444481583208)
    (('to', 'from'), 0.013528675216452513)
    (('upon', 'that'), 0.013141740645515654)
    (('whether', 'to'), 0.013031108375553657)
    (('no', '.'), 0.01235024698095799)
    (('trial', 'it'), 0.01235024698095799)
    (('in', ','), 0.012116670939782637)
    (('or', ','), 0.01204594192685704)
    (('record', 'of'), 0.011431779367068629)
    ((',', '*'), 0.010384807556935272)
    (('answer', 'the'), 0.009001468362296094)
    (('effect', 'the'), 0.009001468362296094)
    (('questions', 'the'), 0.009001468362296094)
    (('court', 'of'), 0.008483634920850136)
    (('trial', 'that'), 0.008275985473702008)
    (('might', ','), 0.0060432603717082145)
    (('states', 'the'), 0.005179757409887742)
    (('“', 'and'), 0.004726491198301375)
    (('but', 'which'), 0.0043935231526148755)
    (('warrant', 'the'), 0.0034632848208637768)
    ((',', 'from'), 0.0026504965526612523)
    (('court', '.'), 0.002422142016630602)
    (('.', 'did'), 0.0018813003829727393)
    (('sustained', 'of'), 0.0016838028155063967)
    (('brought', 'the'), 0.0010962702769496957)
    (('with', 'to'), 0.0006473839197198572)
    (('his', 'an'), -0.00021573009997410963)
    (('.', 'decision'), -0.0006039630066219104)
    (('might', 'in'), -0.00159563937867091)
    (('is', 'for'), -0.0025742575118528066)
    (('than', 'of'), -0.0037706272766016014)
    (('in', 'law'), -0.004245217792618661)
    (('more', 'the'), -0.004804331162738151)
    (('been', 'the'), -0.005182445446354933)
    ((',', 'now'), -0.0051839950515457645)
    (('entered', ','), -0.0051839950515457645)
    (('now', ','), -0.0051839950515457645)
    (('been', 'an'), -0.005670160192082108)
    ((';', 'the'), -0.007118197000984594)
    (('part', ','), -0.007762539153322479)
    (('to', 'defendant'), -0.007764855896478906)
    (('and', 'is'), -0.008959892525165714)
    (('court', 'there'), -0.00946173935445671)
    (('fact', 'the'), -0.010680897499408815)
    (('v.', 'state'), -0.010723387918368132)
    ((',', 'within'), -0.01076502731484652)
    (('opinion', 'is'), -0.011460822227139289)
    (('in', 'et'), -0.01157972795129325)
    (('was', '.'), -0.011622963966772204)
    (('been', '.'), -0.012448068387829636)
    (('common', 'of'), -0.012815766879608503)
    (('interest', 'of'), -0.012815766879608503)
    (('of', 'due'), -0.012815766879608503)
    (('a', ';'), -0.013813333500696245)
    (('(', 'as'), -0.015600893306146446)
    (('have', 'the'), -0.015873200276736554)
    (('constitution', 'and'), -0.016233127437342176)
    (('bond', '.'), -0.01788195443845808)
    (('construction', ','), -0.018542377906566543)
    (('them', ','), -0.018542377906566543)
    (('for', 'had'), -0.019831580996655163)
    (('second', 'the'), -0.0199112235529455)
    (('the', 'can'), -0.0199112235529455)
    (('reversed', 'the'), -0.021198769413405927)
    (('opinion', ','), -0.021509869609431576)
    (('be', 'all'), -0.02181635617023403)
    (('might', 'to'), -0.022072692580366038)
    (('the', 'laws'), -0.02236270254704209)
    ((',', 'ever'), -0.023704083022344946)
    (('an', 'was'), -0.023704083022344946)
    (('consideration', ','), -0.023704083022344946)
    (('issued', ','), -0.023704083022344946)
    ((';', 'in'), -0.023963452407123498)
    (('and', 'after'), -0.024661749507924213)
    (('and', 'to'), -0.0250665120280118)
    (('and', 'had'), -0.026217216009964517)
    (('it', 'upon'), -0.027178117205679087)
    (('was', 'there'), -0.029765835809545393)
    ((',', 'aside'), -0.030130352181775066)
    ((',', 'contended'), -0.030130352181775066)
    ((',', 'oyer'), -0.030130352181775066)
    ((',', 'reason'), -0.030130352181775066)
    (('were', ','), -0.030130352181775066)
    (('declaration', 'the'), -0.030339423269872157)
    (('be', 'when'), -0.030667273752499113)
    (('law', 'be'), -0.030667273752499113)
    (('property', 'to'), -0.03106147580761842)
    (('to', 'property'), -0.03106147580761842)
    (('so', 'is'), -0.031089628976076256)
    (('defendant', ','), -0.031179203584489557)
    (('and', 'with'), -0.03164817982402823)
    (('.', 'in'), -0.03174080299884352)
    (('a', 'being'), -0.03219186281555153)
    (('was', 'from'), -0.03235595327525331)
    (('at', ','), -0.03296565287991626)
    (('can', 'a'), -0.03326569616686825)
    (('the', 'under'), -0.0336691478837281)
    ((';', 'this'), -0.0340426585034912)
    (('this', ';'), -0.0340426585034912)
    (('circuit', '.'), -0.03495546779739911)
    ((',', 'governor'), -0.03536700054792519)
    (('power', 'that'), -0.03592381843683512)
    (('court', 'trial'), -0.03642878695472618)
    (('question', 'court'), -0.03642878695472618)
    (('is', 'on'), -0.03664915431797411)
    (('brought', 'of'), -0.03666250883397382)
    (('a', 'upon'), -0.03740015381831441)
    ((',', 'same'), -0.03835085898674606)
    (('debt', ','), -0.03835085898674606)
    (('much', ','), -0.03835085898674606)
    (('original', ','), -0.03835085898674606)
    (('same', ','), -0.03835085898674606)
    (('without', 'and'), -0.03860094046579832)
    (('had', 'this'), -0.039580842044923514)
    (('an', 'an'), -0.04025039190223012)
    (('a', 'state'), -0.04129406992280238)
    (('of', 'having'), -0.041384919076378424)
    (('of', 'unless'), -0.041384919076378424)
    (('well', 'of'), -0.041384919076378424)
    (('being', 'in'), -0.042341981721982336)
    ((',', 'process'), -0.04338644888404275)
    (('jury', ','), -0.04338644888404275)
    (('in', 'note'), -0.044001205643670716)
    (('law', 'of'), -0.044034497490326174)
    (('error', 'it'), -0.044233281385409384)
    (('again', 'the'), -0.04544631566008306)
    (('amendment', 'the'), -0.04544631566008306)
    (('competent', 'the'), -0.04544631566008306)
    (('each', 'the'), -0.04544631566008306)
    (('equity', 'the'), -0.04544631566008306)
    (('grant', 'the'), -0.04544631566008306)
    (('ground', 'the'), -0.04544631566008306)
    (('last', 'the'), -0.04544631566008306)
    (('lie', 'the'), -0.04544631566008306)
    (('pleadings', 'the'), -0.04544631566008306)
    (('say', 'the'), -0.04544631566008306)
    (('subject', 'the'), -0.04544631566008306)
    (('the', 'constable'), -0.04544631566008306)
    (('the', 'costs'), -0.04544631566008306)
    (('the', 'exception'), -0.04544631566008306)
    (('the', 'heirs'), -0.04544631566008306)
    (('the', 'justice'), -0.04544631566008306)
    (('the', 'neither'), -0.04544631566008306)
    (('the', 'used'), -0.04544631566008306)
    (('the', 'years'), -0.04544631566008306)
    (('whom', 'the'), -0.04544631566008306)
    (('years', 'the'), -0.04544631566008306)
    (('they', 'it'), -0.045993739249848176)
    (('also', '.'), -0.047028300097974096)
    (('as', 'not'), -0.04891540416902984)
    (('evidence', ','), -0.04923917512947895)
    ((',', 'a'), -0.049239175129482504)
    ((',', 'sworn'), -0.049239175129482504)
    (('considered', ','), -0.049239175129482504)
    (('duty', ','), -0.049239175129482504)
    (('matter', ','), -0.049239175129482504)
    (('proper', ','), -0.049239175129482504)
    (('seal', ','), -0.049239175129482504)
    (('p.', 'a'), -0.050339209525812834)
    (('for', 'this'), -0.050593834767340695)
    ((',', 'on'), -0.05108997081742572)
    (('and', 'from'), -0.05159828099467134)
    (('states', 'to'), -0.051820035974415646)
    (('.', 'defendants'), -0.0518292863617944)
    (('defendant', '.'), -0.0518292863617944)
    (('motion', '.'), -0.0518292863617944)
    (('in', 'that'), -0.054804623477167524)
    (('to', 'cause'), -0.055495694117816186)
    (('but', 'judgment'), -0.05565386051732446)
    (('judgment', 'which'), -0.05565386051732446)
    (('is', 'with'), -0.05569160220915137)
    (('record', 'it'), -0.05651161116384529)
    (('so', 'be'), -0.05689881532567753)
    (('parties', 'and'), -0.0569794697806536)
    (('of', 'assigned'), -0.05720988623805923)
    ((',', 'defendant'), -0.05765141494568127)
    (('must', 'in'), -0.05771224376166728)
    (('made', 'that'), -0.057866371317508936)
    (('evidence', '.'), -0.05791661624071054)
    (('it', ';'), -0.058732851080524284)
    (('a', '”'), -0.05932799275306522)
    (('on', ','), -0.05991382120443234)
    (('the', 'other'), -0.059945885355194406)
    (('the', 'which'), -0.059945885355194406)
    (('of', 'or'), -0.0601214816579656)
    (('could', 'a'), -0.06060754497963927)
    (('deed', 'to'), -0.062088371428242084)
    (('to', 'deed'), -0.062088371428242084)
    (('general', 'of'), -0.06244653460420935)
    (('was', 'as'), -0.06265771736691761)
    (('that', 'justice'), -0.06280711258778027)
    (('it', '.'), -0.06296984396699656)
    (('by', ','), -0.06347461171494473)
    ((',', 'absence'), -0.06434606751968985)
    ((',', 'consent'), -0.06434606751968985)
    ((',', 'intention'), -0.06434606751968985)
    ((',', 'maker'), -0.06434606751968985)
    ((',', 'refused'), -0.06434606751968985)
    ((',', 'security'), -0.06434606751968985)
    ((',', 'those'), -0.06434606751968985)
    ((',', 'vacancy'), -0.06434606751968985)
    (('diligence', ','), -0.06434606751968985)
    (('directly', ','), -0.06434606751968985)
    (('given', ','), -0.06434606751968985)
    (('paper', ','), -0.06434606751968985)
    (('plaintiffs', ','), -0.06434606751968985)
    ((')', 'a'), -0.06443940657066705)
    (('first', 'the'), -0.06455513860778694)
    (('the', 'in'), -0.06488619648376925)
    (('the', 'opinion'), -0.06534587309778672)
    (('of', 'party'), -0.0659271033391704)
    (('not', ';'), -0.066545298190821)
    (('therefore', 'of'), -0.0672635509019841)
    (('in', '’'), -0.0676848298364412)
    (('he', 'by'), -0.06852783944123075)
    (('its', 'to'), -0.06889354933335667)
    (('evidence', 'is'), -0.06956377679070869)
    (('no', 'in'), -0.06994965768130612)
    (('?', 'in'), -0.07075766388606652)
    (('person', 'in'), -0.07075766388606652)
    ((',', 'give'), -0.07077233667912353)
    ((',', 'make'), -0.07077233667912353)
    (('by', 'at'), -0.07104783141974735)
    (('by', 'from'), -0.07212648196785665)
    (('from', 'by'), -0.07212648196785665)
    (('must', 'and'), -0.07234973182033855)
    (('a', 'under'), -0.07418595148017815)
    ((',', 'legislature'), -0.0743301560923122)
    (('will', 'was'), -0.07443488514327257)
    (('and', 'on'), -0.07512681649091135)
    (('a', 'ought'), -0.07587430163295039)
    ((',', 'other'), -0.07593404179490193)
    (('as', 'on'), -0.07603674716812137)
    (('to', 'contract'), -0.0773551280815532)
    ((',', 'held'), -0.07815186704472055)
    (('a', 'having'), -0.07890836172258275)
    (('plaintiff', 'is'), -0.07902410603977472)
    (('to', 'of'), -0.07997107628856526)
    ((',', 'sufficient'), -0.08017103468137066)
    (('this', '.'), -0.0813483732818483)
    ((',', 'such'), -0.08173620439375995)
    (('from', 'court'), -0.0824411847115627)
    (('because', 'of'), -0.08320509477100657)
    (('decree', 'of'), -0.08320509477100657)
    (('equity', 'of'), -0.08320509477100657)
    (('nor', 'of'), -0.08320509477100657)
    (('of', 'hubbard'), -0.08320509477100657)
    (('of', 'second'), -0.08320509477100657)
    (('and', '”'), -0.08411559971816729)
    (('law', 'as'), -0.08488766475038645)
    (('peace', 'the'), -0.08497467984672014)
    (('?', 'and'), -0.08539515194474134)
    (('and', '?'), -0.08539515194474134)
    ((',', 'commenced'), -0.08671388054814244)
    ((',', 'found'), -0.08671388054814244)
    (('questions', ','), -0.08671388054814244)
    (('good', ','), -0.086713880548146)
    (('verdict', 'in'), -0.08905848062901)
    (('than', '.'), -0.0904505803891027)
    (('him', 'a'), -0.09098119402315774)
    (('when', 'in'), -0.09170805904295776)
    (('as', 'for'), -0.09258792343196731)
    (('trial', 'and'), -0.09304872448817392)
    (('the', 'have'), -0.09317892644418535)
    (('have', 'an'), -0.09366664118991253)
    (('to', 'that'), -0.09392742947625976)
    (('there', 'not'), -0.09402603461293069)
    (('between', ','), -0.09409341091373946)
    ((',', 'for'), -0.094276273584736)
    (('that', 'action'), -0.09481750749040074)
    ((',', 'its'), -0.09522538040366157)
    (('in', 'jury'), -0.09611323816296036)
    (('in', 'they'), -0.09611323816296036)
    (('“', 'in'), -0.09611323816296036)
    (('to', '.'), -0.09748258993631964)
    (('act', 'and'), -0.09789812414956955)
    (('in', ';'), -0.09796403385090002)
    (('his', 'as'), -0.0980630534981195)
    (('and', 'a'), -0.09815068944200078)
    (('the', 'would'), -0.09888557462154424)
    ((',', 'al'), -0.0989922103265819)
    (('he', 'for'), -0.09932950131102558)
    (('and', 'evidence'), -0.10066190859804891)
    (('by', ';'), -0.10094931713360822)
    (('on', 'with'), -0.10182077293835334)
    ((',', 'general'), -0.1024811964064618)
    ((',', 'r.'), -0.1024811964064618)
    (('several', ','), -0.1024811964064618)
    (('the', 'is'), -0.10255076659607099)
    (('people', 'and'), -0.10369596868768127)
    (('instrument', 'the'), -0.10434000471364868)
    (('intended', 'the'), -0.10434000471364868)
    (('nature', 'the'), -0.10434000471364868)
    (('oyer', 'the'), -0.10434000471364868)
    (('the', 'oyer'), -0.10434000471364868)
    (('the', 'passed'), -0.10434000471364868)
    ((',', 'present'), -0.106166243214318)
    (('a', 'only'), -0.10778948170970892)
    (('the', 'appears'), -0.10818207100804145)
    ((',', 'true'), -0.10874018687814413)
    (('decide', ','), -0.10874018687814413)
    (('verdict', 'to'), -0.10953553383070513)
    ((',', 'against'), -0.1097565272096368)
    (('without', 'the'), -0.11029746001004881)
    (('of', '.'), -0.11030723882618076)
    (('and', 'jury'), -0.11075072622163518)
    (('below', 'that'), -0.11130563027897011)
    (('much', 'the'), -0.11256051151861968)
    (('the', 'debt'), -0.11256051151861968)
    (('can', 'to'), -0.11322058063855778)
    (('in', 'decision'), -0.11360066489180198)
    (('in', 'party'), -0.11360066489180198)
    ((',', 'issue'), -0.11419461697025213)
    (('they', 'by'), -0.1142054138358759)
    (('will', 'by'), -0.1142054138358759)
    (('“', 'by'), -0.1142054138358759)
    (('an', 'which'), -0.11425097334600665)
    (('legal', 'of'), -0.11715242669434289)
    (('of', 'defendant'), -0.11715242669434289)
    ((',', 'party'), -0.11745740397925175)
    (('if', 'by'), -0.11750049339745416)
    (('the', 'will'), -0.11759610141591992)
    (('is', 'it'), -0.11768572640887953)
    (('which', 'court'), -0.1180425525083777)
    ((')', 'on'), -0.11875762215274221)
    (('of', 'under'), -0.11912466902594687)
    (('judgment', 'the'), -0.11999326902513374)
    (('*', 'the'), -0.12040837334130572)
    (('a', 'do'), -0.12072853741720735)
    (('demurrer', 'a'), -0.12072853741720735)
    (('a', 'or'), -0.1207285374172109)
    (('not', 'question'), -0.12099308221320015)
    (('decree', 'to'), -0.12220936386581371)
    (('as', 'case'), -0.1224472121250777)
    (('court', 'defendant'), -0.12284353877220155)
    ((',', 'an'), -0.12323975657325903)
    ((',', 'be'), -0.12323975657325903)
    ((',', 'called'), -0.12323975657325903)
    ((',', 'demurrer'), -0.12323975657325903)
    ((',', 'entitled'), -0.12323975657325903)
    ((',', 'thereof'), -0.12323975657325903)
    (('called', ','), -0.12323975657325903)
    (('duties', ','), -0.12323975657325903)
    (('every', ','), -0.12323975657325903)
    (('ground', ','), -0.12323975657325903)
    (('manner', ','), -0.12323975657325903)
    (('rights', ','), -0.12323975657325903)
    (('want', ','), -0.12323975657325903)
    (('c.', 'the'), -0.12344882766135612)
    (('matter', 'the'), -0.12344882766135612)
    (('mere', 'the'), -0.12344882766135612)
    (('settled', 'the'), -0.12344882766135612)
    (('show', 'the'), -0.12344882766135612)
    (('the', 'view'), -0.12344882766135612)
    (('not', 'in'), -0.12375422189025542)
    (('that', 'so'), -0.12436178310441903)
    (('statute', 'and'), -0.1247575842155122)
    (('to', 'it'), -0.12547707769972405)
    (('with', ','), -0.12556480951950277)
    (('.', 'defendant'), -0.12582986780557093)
    (('on', 'it'), -0.1258470469390609)
    (('one', '.'), -0.1268163259916193)
    (('to', 'see'), -0.1271081632786526)
    (('this', 'upon'), -0.12801880671250743)
    (('and', 'execution'), -0.12823815295047325)
    (('of', 'justice'), -0.12829298429954505)
    (('we', 'the'), -0.1292779064635532)
    (('is', ','), -0.129328921649023)
    (('a', 'by'), -0.12947217048918702)
    (('s', 'not'), -0.12987123355965124)
    (('clerk', 'the'), -0.13033521324659603)
    (('do', 'the'), -0.13033521324659603)
    (('governor', 'to'), -0.1305971493585325)
    (('.', 'law'), -0.13071445060283082)
    ((',', '”'), -0.13222853980051497)
    (('so', 'and'), -0.132577088674811)
    (('error', 'be'), -0.132847668558977)
    (('the', 'are'), -0.13290915691042215)
    (('right', 'a'), -0.13355257777479324)
    (('of', 'power'), -0.13383116784097382)
    (('court', ';'), -0.13398409637740016)
    (('that', 'act'), -0.13407693793762832)
    ((',', 'before'), -0.13408713881025136)
    (('may', 'to'), -0.1342821961663887)
    (('“', 'be'), -0.1346081264234158)
    (('.', 'they'), -0.13511962972282987)
    (('.', '“'), -0.13511962972282987)
    (('will', '.'), -0.13511962972282987)
    (('cases', 'a'), -0.13537531338161202)
    (('cases', 'the'), -0.13564412463166065)
    (('the', 'affirmed'), -0.13564412463166065)
    (('amount', ','), -0.1362959093987044)
    (('defense', ','), -0.1362959093987044)
    (('case', 'the'), -0.13694467014558143)
    (('its', 'a'), -0.13780205077615193)
    (('given', 'of'), -0.13834664896346638)
    (('diligence', 'the'), -0.13855572005156347)
    (('relief', 'the'), -0.13855572005156347)
    (('supposed', 'the'), -0.13855572005156347)
    (('the', 'still'), -0.13855572005156347)
    (('not', 'they'), -0.13869508394665786)
    (('not', 'this'), -0.13937161152805544)
    (('and', 'as'), -0.13961554294262157)
    (('might', '.'), -0.1401377044894545)
    (('that', 'cause'), -0.14097649695531445)
    (('to', 'judgment'), -0.141013765303871)
    (('great', ','), -0.1411616645705216)
    (('are', 'or'), -0.1418706717139102)
    (('the', 'does'), -0.14230785491267284)
    (('to', 'there'), -0.14231056992232283)
    (('be', 'would'), -0.1433783360511498)
    (('this', '('), -0.14353180608383198)
    (('.', 'sufficient'), -0.14388983935056388)
    (('sufficient', '.'), -0.14388983935056388)
    (('therefore', 'the'), -0.14498198921099714)
    (('record', 'be'), -0.1451259983374129)
    (('of', 'ought'), -0.14526606290325716)
    (('consideration', 'and'), -0.14551614438230942)
    (('could', 'that'), -0.1475691742657439)
    (('legislature', 'the'), -0.1485398086241858)
    (('facts', '.'), -0.14912648771671044)
    (('that', 'right'), -0.1501248791694998)
    ((',', ')'), -0.15027900932560811)
    (('proceedings', 'of'), -0.1503192906295432)
    (('are', 'the'), -0.1510565036206799)
    (('to', ';'), -0.15135570952532973)
    (('to', 'demurrer'), -0.15135570952532973)
    (('to', 'himself'), -0.15135570952532973)
    (('to', 'jurisdiction'), -0.15135570952532973)
    ((',', 'established'), -0.15180890877002895)
    ((',', 'her'), -0.15180890877002895)
    ((',', 'points'), -0.15180890877002895)
    ((',', 'subsequent'), -0.15180890877002895)
    ((',', 'why'), -0.15180890877002895)
    (('according', ','), -0.15180890877002895)
    (('established', ','), -0.15180890877002895)
    (('liable', ','), -0.15180890877002895)
    (('paid', ','), -0.15180890877002895)
    (('real', ','), -0.15180890877002895)
    (('seem', ','), -0.15180890877002895)
    (('subsequent', ','), -0.15180890877002895)
    (('tried', ','), -0.15180890877002895)
    (('way', ','), -0.15180890877002895)
    (('the', 'held'), -0.15236151957659416)
    (('and', 'have'), -0.15407815788573487)
    (('defendant', 'a'), -0.15467586934054722)
    ((')', 'this'), -0.15505805946486007)
    (('this', 'any'), -0.15560463838440697)
    ((',', 'by'), -0.15704872702714212)
    (('could', 'the'), -0.15709167148898828)
    (('court', 'be'), -0.1580582313562573)
    (('what', 'of'), -0.15849322207524352)
    (('other', 'in'), -0.16002500198315417)
    (('is', 'if'), -0.1601155344983347)
    ((')', 'the'), -0.16092353308001606)
    (('the', 'prove'), -0.16092353308001606)
    (('he', 'in'), -0.16146697615705818)
    (('before', ','), -0.1615678752323575)
    (('as', 'was'), -0.16219339091783525)
    (('upon', ','), -0.1627681207598961)
    (('of', 'principle'), -0.16337544345498856)
    (('.', 'from'), -0.16343002574620868)
    (('admitted', ','), -0.1638817410706075)
    (('obtained', ','), -0.1638817410706075)
    (('too', ','), -0.1638817410706075)
    (('is', 'against'), -0.16402846079420286)
    (('or', 'to'), -0.1647140923803505)
    (('from', ','), -0.16480624929890197)
    (('is', 'the'), -0.16483504485041323)
    (('and', 'against'), -0.1659802469420235)
    (('state', 'the'), -0.16603645647367316)
    (('them', 'the'), -0.16675261188221668)
    (('to', 'without'), -0.1672972533943522)
    ((',', 'set'), -0.16763387593170975)
    (('the', 'nor'), -0.16830306344561308)
    (('by', '.'), -0.16843291384959613)
    (('be', 'his'), -0.16937354458409004)
    (('.', 'present'), -0.16988504788350767)
    ((';', 'or'), -0.16988504788350767)
    (('can', '.'), -0.16988504788350767)
    (('second', '.'), -0.16988504788350767)
    (('the', ':'), -0.16989368658457238)
    (('court', 'this'), -0.17028453368951801)
    ((',', 'construction'), -0.17054547135161613)
    (('a', '2'), -0.17135461048717815)
    (('to', 'against'), -0.17181981208504737)
    (('plea', ','), -0.1721493570542023)
    (('plaintiff', 'that'), -0.1722962601681175)
    (('for', 'said'), -0.1730778412923364)
    (('cause', '.'), -0.1735607060269082)
    (('the', 'shall'), -0.17377041263562276)
    (('jury', 'the'), -0.1741796297822873)
    (('and', 'who'), -0.17466249004182544)
    (('as', 'had'), -0.17468433510103054)
    ((',', 'notice'), -0.17477005721334038)
    (('to', 'brought'), -0.1752024514796986)
    ((',', ':'), -0.1758543827366772)
    (('no', 'this'), -0.17595328377620945)
    (('be', 'or'), -0.17603727486510223)
    (('be', 'the'), -0.17712942472553905)
    (('a', 'to'), -0.17751929000698397)
    (('one', 'it'), -0.1776206966631122)
    (('the', 'being'), -0.1793020623948749)
    (('must', 'that'), -0.17961164366987958)
    ((',', 'into'), -0.17982328493962285)
    (('their', 'to'), -0.17992486172209965)
    (('officer', '.'), -0.1801533833373341)
    (('land', 'the'), -0.18037589574619162)
    (('out', ','), -0.18145437051855495)
    (('power', 'in'), -0.1815047293936054)
    (('whether', 'in'), -0.1815047293936054)
    (('can', 'this'), -0.18159984691735076)
    (('decided', 'of'), -0.18274076832192065)
    (('of', 'smith'), -0.18274076832192065)
    (('defendant', 'by'), -0.18291816391989002)
    (('exceptions', 'the'), -0.18294983941001774)
    (('the', 'until'), -0.18294983941001774)
    (('that', 'state'), -0.18339725340137036)
    (('error', 'a'), -0.1837383349430084)
    ((',', 'the'), -0.18394058801772317)
    (('to', 'case'), -0.18396037495725537)
    (('as', 'of'), -0.18421969724783338)
    (('proceedings', '.'), -0.18453182384790878)
    (('and', 'rule'), -0.18615812887965433)
    (('is', 'then'), -0.18620869564297848)
    (('to', 'i'), -0.18697961925605)
    (('cause', 'in'), -0.18702173436117064)
    (('an', 'by'), -0.1890054937988026)
    (('opinion', 'be'), -0.18927310202179726)
    (('it', 'with'), -0.18928361418869244)
    (('on', 'any'), -0.18969352896368719)
    (('the', 'and'), -0.1898841319849467)
    (('upon', 'to'), -0.1908840737119668)
    (('been', 'and'), -0.19161255897176233)
    (('a', 'but'), -0.1916950587713515)
    (('a', 'then'), -0.1925192204852202)
    ((',', 'clearly'), -0.1936290844646571)
    ((',', 'decree'), -0.1936290844646571)
    ((',', 'mandamus'), -0.1936290844646571)
    ((',', 'our'), -0.1936290844646571)
    ((',', 'pay'), -0.1936290844646571)
    ((',', 'pleaded'), -0.1936290844646571)
    ((',', 'practice'), -0.1936290844646571)
    ((',', 'return'), -0.1936290844646571)
    (('correct', ','), -0.1936290844646571)
    (('may', ','), -0.1936290844646571)
    (('pay', ','), -0.1936290844646571)
    (('to', 'after'), -0.194000046933823)
    (('any', 'be'), -0.19584575594528175)
    (('had', 'and'), -0.19614221745227667)
    (('power', 'and'), -0.19614221745227667)
    (('all', 'it'), -0.19623637483045897)
    (('made', 'of'), -0.1964157052189961)
    (('to', 'in'), -0.19712973816247725)
    (('was', 'but'), -0.19724033801703555)
    (('the', 'who'), -0.1974494091051291)
    (('am', 'the'), -0.19744940910513264)
    (('entitled', 'the'), -0.19744940910513264)
    (('follows', 'the'), -0.19744940910513264)
    (('offense', 'the'), -0.19744940910513264)
    (('the', 'authorized'), -0.19744940910513264)
    (('the', 'declared'), -0.19744940910513264)
    (('the', 'every'), -0.19744940910513264)
    (('a', 'is'), -0.19829709091290937)
    (('of', 'without'), -0.19868231219094312)
    (('was', 'case'), -0.20069865778944518)
    (('for', 'if'), -0.2007695751613845)
    (('to', 'with'), -0.20098647724993057)
    (('of', 'appointment'), -0.20184959126962454)
    (('of', 'secretary'), -0.20184959126962454)
    (('this', 'justice'), -0.2039676599458069)
    (('his', 'for'), -0.20624470522753668)
    (('to', 'writ'), -0.2079392378916971)
    ((',', 'cases'), -0.2082758604290582)
    (('error', 'and'), -0.20852594190811047)
    (('from', 'on'), -0.209233715359332)
    (('all', 'of'), -0.2107551444915181)
    (('appear', ','), -0.21224476263200387)
    (('he', 'as'), -0.21354027091805605)
    (('could', 'to'), -0.21409146487329167)
    (('the', 'title'), -0.2153713171023952)
    (('this', 'to'), -0.21567052300704148)
    (('very', ','), -0.21634916096473944)
    (('.', 'has'), -0.2165257730022958)
    (('by', 'and'), -0.21699553280224748)
    (('from', 'in'), -0.21753303857781603)
    (('they', '.'), -0.21758178991480293)
    (('any', 'to'), -0.21846990538386635)
    (('act', 'be'), -0.21861706360393995)
    (('this', 'that'), -0.21868916565257024)
    (('note', '.'), -0.22051112095347136)
    (('.', 'had'), -0.22051112095347492)
    (('.', 'whether'), -0.22051112095347492)
    (('this', 'for'), -0.22051883620965285)
    (('of', 'taken'), -0.22070861852094126)
    (('that', 'before'), -0.2207168469995544)
    (('by', 'state'), -0.22153945794719831)
    (('cases', 'that'), -0.22233694266771664)
    (('court', 'said'), -0.22237921232311564)
    (('time', ','), -0.2227754301241731)
    (('state', 'it'), -0.2237171112525651)
    (('it', 'there'), -0.22371711125256866)
    (('made', 'it'), -0.22371711125256866)
    (('of', 'are'), -0.22580949021380547)
    (('this', 'would'), -0.22599396627580504)
    (('false', 'the'), -0.22601856130190257)
    (('nothing', 'the'), -0.22601856130190257)
    (('over', 'the'), -0.22601856130190257)
    (('the', 'certainly'), -0.22601856130190257)
    (('the', 'having'), -0.22601856130190257)
    (('the', 'her'), -0.22601856130190257)
    (('the', 'hobson'), -0.22601856130190257)
    (('the', 'nothing'), -0.22601856130190257)
    (('verdict', 'the'), -0.22601856130190257)
    ((',', 'only'), -0.22649371837028554)
    ((',', 'one'), -0.2267959483998574)
    (('s', 'and'), -0.22745775791848644)
    (('upon', 'court'), -0.22796024458641284)
    (('not', 'the'), -0.22995204879246245)
    (('therefore', 'in'), -0.23041432987455224)
    (('trial', 'in'), -0.23041432987455224)
    (('.', 'opinion'), -0.23042658981855624)
    (('are', ','), -0.23088047976122894)
    (('had', 'the'), -0.23100196881615886)
    (('no', 'is'), -0.23102719948482786)
    (('have', 'as'), -0.23215594908540282)
    (('will', 'the'), -0.23307331883585647)
    (('“', 'the'), -0.23307331883585647)
    (('in', 'for'), -0.2331597066133675)
    ((',', 'acts'), -0.234271068962002)
    ((',', 'executed'), -0.234271068962002)
    ((',', 'witness'), -0.234271068962002)
    (('appeal', ','), -0.234271068962002)
    (('grounds', ','), -0.234271068962002)
    (('judgment', 'and'), -0.23470987371176477)
    (('1', 'as'), -0.2355665772480542)
    (('of', 'case'), -0.23610399392064352)
    (('by', 'but'), -0.2370108667096389)
    (('party', ','), -0.23775163769696306)
    (('obtained', 'the'), -0.2380913936024811)
    (('appear', 'of'), -0.23932429668828803)
    (('“', 'of'), -0.24108475455272327)
    (('in', ':'), -0.2423869715406255)
    (('a', 'in'), -0.24250272514448312)
    (('should', 'a'), -0.2441109529224903)
    (('(', 'v.'), -0.24473489252164882)
    (('whole', 'the'), -0.24475512388348974)
    (('justice', 'and'), -0.2450518179332235)
    (('time', 'and'), -0.2450518179332235)
    (('of', 'could'), -0.2454765236698826)
    (('officer', 'of'), -0.2454765236698826)
    (('and', 'of'), -0.24790581498837483)
    ((';', 'a'), -0.2482785871377189)
    (('.', 'general'), -0.24866216126762453)
    (('the', 'notice'), -0.248979709745214)
    (('the', 'show'), -0.248979709745214)
    (('first', 'in'), -0.24952315282225612)
    (('of', 'must'), -0.2510467817128301)
    (('being', '.'), -0.25127337472416045)
    (('these', 'the'), -0.25189719312750825)
    (('.', 'may'), -0.2523472080754807)
    ((',', 'laws'), -0.2530076315435892)
    (('for', 'to'), -0.2536367598150626)
    (('of', 'on'), -0.25382386638340293)
    (('was', ';'), -0.25382386638340293)
    (('any', 'court'), -0.25554607625831594)
    (('the', 'reversed'), -0.25566402305042857)
    (('are', 'be'), -0.25683638583442914)
    ((',', 'in'), -0.2570699618756045)
    (('taken', 'a'), -0.25823206116714204)
    (('is', ';'), -0.25904157565442176)
    (('made', '.'), -0.26037558183141485)
    ((',', 'any'), -0.26074328032319016)
    (('page', ','), -0.26074328032319016)
    ((',', 'indictment'), -0.2607432803231937)
    (('exceptions', ','), -0.2607432803231937)
    (('is', 'such'), -0.2612594009042404)
    (('can', 'that'), -0.26283172089577533)
    (('of', 'only'), -0.2629111170059062)
    (('act', 'not'), -0.2633460056245305)
    (('that', 'their'), -0.26540566455960146)
    (('one', 'be'), -0.2662350838366798)
    (('had', 'on'), -0.2663148105665982)
    (('all', 'to'), -0.2668329269452663)
    (('or', 'a'), -0.2675699257464821)
    (('the', 'correct'), -0.2678387369965307)
    (('the', 'mandamus'), -0.2678387369965307)
    (('the', 'may'), -0.2678387369965307)
    (('the', 'stated'), -0.2678387369965307)
    (('it', 'they'), -0.26838616058629583)
    (('and', 'should'), -0.2688985598875888)
    (('should', 'and'), -0.2688985598875888)
    (('see', ','), -0.26891721176889405)
    (('to', 'or'), -0.2690507521950849)
    (('.', 'other'), -0.26942072143442175)
    (('on', '.'), -0.26942072143442175)
    (('to', 'first'), -0.27000020602395125)
    (('cause', 'is'), -0.2700967641634193)
    (('it', 'on'), -0.2702369562742355)
    (('have', 'case'), -0.27066121595701276)
    (('s', 'to'), -0.2712651732605238)
    (('court', 'a'), -0.2723109587377195)
    (('made', 'court'), -0.2724961451882493)
    (('under', ','), -0.2726173806114822)
    (('only', ','), -0.2737994331486391)
    (('be', ','), -0.274106422199889)
    (('law', 'a'), -0.27420301807892855)
    (('circuit', 'the'), -0.2749281617828494)
    (('the', 'their'), -0.2749281617828494)
    (('any', 'by'), -0.27497871690865594)
    (('to', 'law'), -0.27568384452753136)
    (('of', 'interest'), -0.27585017271340107)
    (('subject', 'of'), -0.27585017271340107)
    (('was', 'his'), -0.2758501727134046)
    (('a', 'county'), -0.2768477393344888)
    (('in', '.'), -0.2777200446529058)
    (('appellant', ','), -0.2793589584905405)
    (('is', '.'), -0.27987507907158715)
    (('is', '('), -0.2810678819844199)
    (('is', 'he'), -0.2810678819844199)
    (('his', 'is'), -0.28106788198442345)
    (('the', 'however'), -0.28151367389360615)
    (('of', 'it'), -0.28239301858017285)
    (('statute', 'it'), -0.28239301858017285)
    (('a', 'could'), -0.2829999663160869)
    (('person', 'a'), -0.2829999663160869)
    ((';', 'be'), -0.2848507620040266)
    (('be', 'it'), -0.2848507620040266)
    (('(', 'in'), -0.28545569343251387)
    (('in', 'costs'), -0.2869978582409196)
    (('of', ':'), -0.2878228143794743)
    (('assigned', 'to'), -0.2888592332752644)
    (('a', 'as'), -0.28991464253561716)
    ((',', 'decided'), -0.2931647580155712)
    ((',', 'duties'), -0.2931647580155712)
    (('entitled', ','), -0.2931647580155712)
    (('these', ','), -0.2931647580155712)
    (('under', '.'), -0.2932674633887906)
    (('that', 'on'), -0.2934200407291989)
    (('from', 'it'), -0.2966965566096711)
    (('bound', 'the'), -0.29698508265604673)
    (('the', 'bound'), -0.29698508265604673)
    (('the', 'required'), -0.29698508265604673)
    (('that', 'opinion'), -0.29737805429788366)
    (('ought', 'a'), -0.29826672296939805)
    (('case', 'and'), -0.2984150435319499)
    (('and', 'where'), -0.29899062504402707)
    (('of', 'in'), -0.29902277241341935)
    (('to', 'and'), -0.29982340288635)
    ((')', 'to'), -0.3003854866562925)
    (('this', 'judgment'), -0.30054091964085927)
    (('to', 'under'), -0.30073333356355647)
    (('plaintiff', 'not'), -0.30156532785501966)
    (('legal', 'the'), -0.3017860689198706)
    (('’', 'is'), -0.30276295308373946)
    (('or', 'the'), -0.303171810108811)
    (('the', 'or'), -0.303171810108811)
    (('other', 'to'), -0.3033588029703793)
    (('to', 'who'), -0.3033588029703793)
    (('lockwood', ','), -0.30381200221507854)
    (('rendered', ','), -0.30381200221507854)
    (('verdict', ','), -0.30381200221507854)
    (('in', 'it'), -0.30407137159986064)
    (('he', 'on'), -0.30517074638362374)
    (('power', ','), -0.3056557021987665)
    (('necessary', ','), -0.3068396949126502)
    (('may', 'that'), -0.30722584025422606)
    (('he', 'or'), -0.30738857163344235)
    (('became', 'the'), -0.3084807214938756)
    (('far', 'the'), -0.3084807214938756)
    (('objections', 'the'), -0.3084807214938756)
    (('the', 'vested'), -0.3084807214938756)
    (('must', 'the'), -0.3101495418544964)
    (('one', 'not'), -0.31096402585727034)
    (('of', 'bill'), -0.31147408244412134)
    (('of', 'will'), -0.31147408244412134)
    ((',', 'decisions'), -0.31227358096327507)
    ((',', 'go'), -0.31227358096327507)
    ((',', 'mere'), -0.31227358096327507)
    ((',', 'particular'), -0.31227358096327507)
    ((',', 'witnesses'), -0.31227358096327507)
    (('a', 'can'), -0.3133736153596054)
    (('trial', 'is'), -0.3134893596768009)
    (('person', 'the'), -0.3146329484754702)
    (('shall', 'the'), -0.3146329484754702)
    (('an', 'the'), -0.3160939056037506)
    (('such', '.'), -0.31672643621277885)
    ((',', 'brought'), -0.3170114999699365)
    (('“', 'court'), -0.31716519452198)
    ((',', 'to'), -0.31798312086706915)
    (('dollars', 'the'), -0.31846481006649796)
    (('the', 'persons'), -0.31846481006649796)
    (('the', 'three'), -0.31846481006649796)
    ((',', 'bond'), -0.319159966548515)
    (('of', 'cases'), -0.32024429207185534)
    (('statute', 'the'), -0.32108075827351)
    (('and', 'declaration'), -0.32305432993449656)
    (('.', 'as'), -0.3236411124323091)
    (('state', 'this'), -0.32455780075939344)
    ((',', 'term'), -0.32487361774290946)
    (('provisions', ','), -0.32487361774290946)
    ((')', 'which'), -0.32520849962017095)
    (('is', 'there'), -0.32716429657387636)
    (('of', 'v.'), -0.32719705271362187)
    (('of', 'shall'), -0.32793868386185565)
    (('of', 'decision'), -0.32896150917296296)
    (('and', 'made'), -0.329116082721697)
    (('himself', ','), -0.3296906340406842)
    (('or', 'in'), -0.33018746454704484)
    (('to', 'verdict'), -0.3319279551671528)
    (('could', '.'), -0.3321564767823837)
    (('of', 'that'), -0.33311737338625136)
    ((',', 'another'), -0.33380674251291964)
    (('of', 'other'), -0.33474386176697024)
    (('cranch', 'the'), -0.33495293285506733)
    (('errors', 'the'), -0.33495293285506733)
    (('to', 'made'), -0.3349556478647209)
    (('before', 'this'), -0.3354051829963858)
    (('this', 'before'), -0.3354051829963858)
    (('for', 'is'), -0.3369932965824134)
    (('given', 'and'), -0.3381612223247039)
    (('was', 'that'), -0.33893469998156434)
    (('as', ';'), -0.3390711530019139)
    (('trial', 'be'), -0.3392985460264022)
    (('the', 'for'), -0.33953146745284)
    (('the', 'then'), -0.33962942006454355)
    (('states', '.'), -0.3398100493258198)
    (('court', 'which'), -0.34043497384482535)
    ((',', 'sheriff'), -0.3404704727939283)
    ((',', 'them'), -0.3404704727939283)
    (('defendant', 'that'), -0.34117317217756593)
    (('opinion', 'in'), -0.3434232917037363)
    (('him', 'the'), -0.3450065975189922)
    (('do', ','), -0.3456321779097067)
    (('on', 'in'), -0.34589154729448524)
    (('of', 'bond'), -0.34623950060479913)
    (('the', 'its'), -0.3463127950196139)
    (('.', 'ought'), -0.3474232334356948)
    (('ought', '.'), -0.3474232334356948)
    (('court', '’'), -0.34763047524903)
    (('were', 'the'), -0.34945250255018223)
    (('who', 'the'), -0.34945250255018223)
    ((',', 'good'), -0.34974828638193856)
    (('in', 's'), -0.3503237936097463)
    (('’', '.'), -0.35061680428240294)
    (('right', 'and'), -0.3509852626822898)
    (('court', 'may'), -0.35193061268265424)
    (('and', 'this'), -0.35243535330682363)
    (('general', 'to'), -0.35298957069498016)
    ((',', 'entered'), -0.3531072984718513)
    (('appellant', 'the'), -0.3535686110224141)
    (('an', 'to'), -0.35488910361046067)
    (('at', '.'), -0.3549964531404939)
    (('are', '.'), -0.35688356268476085)
    (('must', 'to'), -0.35716524666617033)
    (('court', 'and'), -0.35721955814038964)
    (('following', 'the'), -0.35860420091971434)
    (('process', 'the'), -0.35860420091971434)
    (('the', '“'), -0.35860420091971434)
    (('they', 'in'), -0.3591476439967529)
    (('will', 'in'), -0.3591476439967529)
    ((',', 'been'), -0.3593071148067821)
    (('an', 'a'), -0.3596672676335402)
    (('the', ','), -0.3599654040396665)
    (('to', 'upon'), -0.36080907515427896)
    (('said', 'a'), -0.3611267468079724)
    (('as', 'and'), -0.3620079642790692)
    ((',', 'see'), -0.36202661616037446)
    (('act', 'a'), -0.36261713437945176)
    (('s', 'is'), -0.3630094955205969)
    (('defendant', 'the'), -0.3631866135840127)
    (('having', 'of'), -0.36331301396374016)
    (('no', 'of'), -0.36331301396374016)
    (('of', 'plaintiff'), -0.36331301396374016)
    (('verdict', 'of'), -0.36331301396374016)
    (('cause', 'that'), -0.3633689182917621)
    (('by', 'justice'), -0.3639837229674008)
    (('it', 'an'), -0.3652125468792242)
    (('to', 'had'), -0.36548051487817546)
    (('decision', 'a'), -0.3664849518191673)
    ((',', 'opinion'), -0.3666453556581182)
    (('cases', 'in'), -0.3679178536244869)
    (('the', 'one'), -0.36971835101574513)
    (('.', 'reversed'), -0.3707040572716025)
    (('the', 'the'), -0.3715091848493408)
    (('affirmed', ','), -0.3717745927119367)
    (('is', 'will'), -0.37183334590760353)
    (('which', 'a'), -0.37226730441317457)
    (('the', 'might'), -0.3730209736885861)
    (('and', 'bill'), -0.37378513205542774)
    (('but', 'on'), -0.37411810010111424)
    (('a', 'such'), -0.3744851296629932)
    (('not', '”'), -0.3750943632769861)
    ((',', 'cause'), -0.37530304458604746)
    (('also', 'of'), -0.3753858462643187)
    (('is', 'shall'), -0.37622511502476286)
    (('of', 'from'), -0.3763103544926132)
    (('execution', 'in'), -0.37663507072559455)
    (('unless', 'the'), -0.37802165474695215)
    ((',', 'at'), -0.37874048972164687)
    (('to', '“'), -0.37962469719844805)
    (('motion', 'of'), -0.38018683252813545)
    (('now', 'of'), -0.38018683252813545)
    (('to', 'court'), -0.38070698910283696)
    (('and', 'cases'), -0.3825553416831582)
    (('but', 'of'), -0.3836534622479135)
    (('?', 'to'), -0.3840164663156038)
    (('with', 'of'), -0.3843746294915711)
    (('.', 'all'), -0.3848979388543583)
    ((',', 'certificate'), -0.3862741624070516)
    ((',', 'done'), -0.3862741624070516)
    ((',', 'say'), -0.3862741624070516)
    ((',', 'want'), -0.3862741624070516)
    (('application', ','), -0.3862741624070516)
    (('both', ','), -0.3862741624070516)
    (('of', 'declaration'), -0.3862741624070516)
    (('since', ','), -0.3862741624070516)
    ((',', 'admitted'), -0.38627416240705514)
    ((',', 'obtained'), -0.38627416240705514)
    ((',', 'proved'), -0.38627416240705514)
    ((',', 'subject'), -0.38627416240705514)
    (('provided', ','), -0.38627416240705514)
    (('whom', ','), -0.38627416240705514)
    (('certain', 'the'), -0.3864832334951487)
    (('decisions', 'the'), -0.3864832334951487)
    (('the', 'to'), -0.38726047785671014)
    (('to', 'state'), -0.3874230677588528)
    ((')', 'as'), -0.38969722607188473)
    (('out', 'to'), -0.39014256911244516)
    (('to', 'reversed'), -0.39014256911244516)
    (('what', 'to'), -0.39014256911244516)
    (('trial', 'a'), -0.3901892124104336)
    (('but', 'for'), -0.39066927636496374)
    (('and', 'decision'), -0.3912725587842658)
    (('party', 'and'), -0.3912725587842658)
    (('of', 'costs'), -0.3913273901333376)
    (('for', ','), -0.39177015016190353)
    (('the', 'equity'), -0.3933696190803886)
    (('the', 'pay'), -0.3933696190803886)
    (('of', 'at'), -0.3938473821118542)
    (('consideration', 'in'), -0.3939130621574307)
    (('in', 'may'), -0.3939130621574307)
    (('in', 'states'), -0.3939130621574307)
    (('the', 'at'), -0.3940564531999513)
    (('the', 'this'), -0.39503075345031036)
    (('a', 'court'), -0.39516770652325306)
    (('court', 'the'), -0.3975063818885971)
    (('to', 'before'), -0.39827044999584515)
    (('is', 'at'), -0.39906509138287305)
    (('estate', 'the'), -0.3990832702747831)
    (('here', 'the'), -0.3990832702747831)
    (('most', 'the'), -0.3990832702747831)
    (('premises', 'the'), -0.3990832702747831)
    (('senate', 'the'), -0.3990832702747831)
    (('the', 'character'), -0.3990832702747831)
    (('the', 'writing'), -0.3990832702747831)
    (('right', ','), -0.3990982027646375)
    (('a', ','), -0.3992966432006426)
    (('as', 'or'), -0.4001826673053017)
    (('below', 'the'), -0.4005412744826451)
    (('a', 'no'), -0.40083645660994094)
    (('a', 'plaintiff'), -0.4008364566099445)
    (('their', 'a'), -0.4008364566099445)
    (('of', 'is'), -0.4009076519904262)
    (('an', 'as'), -0.4011321211341681)
    (('not', 'will'), -0.4017294897804504)
    (('and', 'been'), -0.40311666416547354)
    (('a', '.'), -0.4032250251040459)
    (('as', 'court'), -0.4049237002965427)
    (('case', 'to'), -0.406352796293703)
    (('same', '.'), -0.40692424518435644)
    (('another', 'the'), -0.40801639504479326)
    (('the', 'another'), -0.40801639504479326)
    (('for', 'in'), -0.4082464131714616)
    (('and', 'may'), -0.408550550216102)
    (('justice', ','), -0.40864197543550773)
    (('of', 'being'), -0.4097059194481929)
    (('below', 'not'), -0.4104996994081844)
    (('not', 'his'), -0.4104996994081844)
    (('this', 'the'), -0.4106276085013292)
    (('in', 'are'), -0.41098657551637174)
    (('its', 'in'), -0.41098657551637174)
    (('have', 'this'), -0.4125542817572203)
    (('this', 'have'), -0.4125542817572203)
    (('is', 'be'), -0.41275099104987234)
    (('right', 'that'), -0.41315928500329235)
    (('debt', 'of'), -0.41335369646333575)
    ((',', 'than'), -0.4137548988291613)
    (('question', 'and'), -0.41497681937553565)
    ((',', 'between'), -0.41602150580110475)
    (('a', 'not'), -0.41699624643376154)
    ((',', 'person'), -0.41730105802767525)
    (('a', 'office'), -0.4177102751743398)
    (('office', 'a'), -0.4177102751743398)
    ((',', 'judgment'), -0.417752393880221)
    (('the', 'not'), -0.4182555795496299)
    ((',', 'two'), -0.41869564009942906)
    (('.', 'under'), -0.4187983454726485)
    (('issue', 'of'), -0.4188081265554473)
    (('hubbard', 'the'), -0.4198418304415803)
    (('is', 'law'), -0.41989558668173643)
    (('.', 'so'), -0.4199803980098018)
    (('one', 'in'), -0.4203852735186224)
    (('there', 'that'), -0.42043645070221913)
    (('sheriff', 'of'), -0.4226915610426758)
    (('for', 'and'), -0.42288390123013286)
    (('county', 'the'), -0.42395793891381217)
    (('good', 'the'), -0.42395793891381217)
    (('court', 'state'), -0.4244992386333024)
    (('then', 'in'), -0.4250617607280951)
    (('of', 'against'), -0.4255972922180824)
    ((',', 'amount'), -0.42580252659368867)
    (('a', 'must'), -0.4260737481089656)
    (('must', 'a'), -0.4260737481089656)
    (('1', 'to'), -0.4263627570251991)
    (('is', 'opinion'), -0.426498321505985)
    (('not', 'or'), -0.42721709435311794)
    (('day', 'the'), -0.42731695100372846)
    (('to', 'when'), -0.42768693797258095)
    (('of', 'p.'), -0.42785326615845065)
    (('office', 'in'), -0.42786039408076704)
    (('in', 'has'), -0.4290581484383935)
    (('is', 'and'), -0.4295355753454402)
    (('of', 'of'), -0.4301783191046944)
    (('is', 'under'), -0.43044550602264664)
    ((',', 'proceedings'), -0.43066828176550587)
    ((',', 'supreme'), -0.43066828176550587)
    ((',', 'sustained'), -0.43066828176550587)
    (('in', 'with'), -0.43204819104420267)
    (('may', '.'), -0.43291945371730023)
    (('1', 'in'), -0.43503204948302354)
    (('of', 'he'), -0.4357215094917919)
    (('the', 'out'), -0.4362362686922481)
    ((',', 'filed'), -0.4369002354770224)
    ((',', 'persons'), -0.4369002354770224)
    (('persons', ','), -0.4369002354770224)
    (('at', 'this'), -0.4371005800657386)
    (('.', ','), -0.44099734874788865)
    (('in', 'or'), -0.4412187769357878)
    (('have', '.'), -0.4414814672207221)
    (('parties', 'the'), -0.44233646822866746)
    (('note', 'be'), -0.4423920389905085)
    (('defendant', 'and'), -0.44249788213944186)
    (('a', 'a'), -0.44311970245487586)
    (('said', 'court'), -0.4447716336595633)
    ((',', 'authorities'), -0.44516785146062077)
    ((',', 'bound'), -0.44516785146062077)
    ((',', 'granted'), -0.44516785146062077)
    ((',', 'justices'), -0.44516785146062077)
    ((',', 'shown'), -0.44516785146062077)
    (('a', 'if'), -0.4454020090433879)
    (('right', '.'), -0.44574349407488256)
    (('make', 'of'), -0.4457751741557132)
    (('these', 'of'), -0.4457751741557132)
    (('it', 'this'), -0.44594045025176854)
    (('by', 'in'), -0.4462836276296649)
    (('that', 'upon'), -0.44628987799178077)
    (('if', 'on'), -0.4472528047313311)
    (('the', 'on'), -0.4494286264299312)
    (('no', 'that'), -0.4498302356970285)
    (('whether', 'that'), -0.45096131771567727)
    (('the', 'made'), -0.4514386753359183)
    (('be', 'opinion'), -0.4523075078555898)
    (('of', 'an'), -0.4533883582655882)
    ((';', ','), -0.4544456650486346)
    (('if', 'in'), -0.45555212794981514)
    (('and', 'at'), -0.45615843172315707)
    (('s', 'that'), -0.45628164964894324)
    (('a', 'there'), -0.4563315692016481)
    ((',', 'correct'), -0.45666349029844966)
    (('because', ','), -0.45666349029844966)
    (('object', ','), -0.45666349029844966)
    (('practice', ','), -0.45666349029844966)
    (('.', 'contract'), -0.4584545458244378)
    (('of', 'were'), -0.4602747438508281)
    (('might', 'the'), -0.4604838149389252)
    (('never', 'the'), -0.4604838149389252)
    (('other', 'the'), -0.4604838149389252)
    (('rule', 'the'), -0.4604838149389252)
    (('the', 'sold'), -0.4604838149389252)
    (('the', 'those'), -0.4604838149389252)
    (('void', 'the'), -0.4604838149389252)
    (('whose', 'the'), -0.4604838149389252)
    (('gave', 'the'), -0.46048381493892876)
    (('mandamus', 'the'), -0.46048381493892876)
    (('present', 'the'), -0.46048381493892876)
    (('the', '2.'), -0.46048381493892876)
    (('the', 'our'), -0.46048381493892876)
    (('the', 'proved'), -0.46048381493892876)
    (('the', 'too'), -0.46048381493892876)
    (('the', 'up'), -0.46048381493892876)
    (('too', 'the'), -0.46048381493892876)
    (('judgment', 'by'), -0.46055698266245315)
    (('of', 'they'), -0.4634771758891709)
    (('of', '“'), -0.4634771758891709)
    (('.', 'after'), -0.4640681522879646)
    (('that', 'in'), -0.464448863985794)
    (('any', ','), -0.4671941577906189)
    (('a', 'and'), -0.46790730941997793)
    (('might', 'of'), -0.4684952506557991)
    ((',', 'rule'), -0.46873632259902465)
    (('”', ','), -0.46926352707808405)
    (('(', 'on'), -0.46980144815642433)
    (('been', ','), -0.4703384271955251)
    (('that', 'which'), -0.4713017659998542)
    (('justice', 'in'), -0.47142242937834666)
    (('sufficient', 'of'), -0.4722473855169049)
    ((',', 'liable'), -0.4737370036573907)
    ((',', 'public'), -0.4737370036573907)
    (('for', 'been'), -0.47473354115343724)
    (('before', 'be'), -0.4751819741081782)
    (('same', 'and'), -0.4756647460746386)
    (('in', 'before'), -0.47732907034506766)
    (('facts', 'of'), -0.4774840338830515)
    (('no', 'the'), -0.4775573282978698)
    (('by', 'that'), -0.47824090222508175)
    (('that', 'by'), -0.47824090222508175)
    (('or', 'was'), -0.4784341129696692)
    (('must', 'not'), -0.4788057127990939)
    (('that', 'against'), -0.4796930362589933)
    (('an', 'it'), -0.48068976429916077)
    (('person', ','), -0.481431395447391)
    (('that', 'to'), -0.4818530831378638)
    (('in', 'but'), -0.4819530968705159)
    (('should', 'the'), -0.4843305568932905)
    (('in', '”'), -0.48451561093833817)
    (('order', ','), -0.4858098359579692)
    (('and', 'be'), -0.48675771432650805)
    (('at', 'it'), -0.48826298400395984)
    (('is', 'court'), -0.48839285523192544)
    ((',', 'et'), -0.4893676553711579)
    ((',', 'made'), -0.4897037520624572)
    (('he', 'that'), -0.4898172535327028)
    (('no', 'it'), -0.49015009354822325)
    ((',', 'legal'), -0.490610822221786)
    (('the', 'these'), -0.49290529263130267)
    ((':', 'of'), -0.49427369184690306)
    (('it', 'in'), -0.49671644954225513)
    (('.', 'not'), -0.49717965252068197)
    (('court', 'from'), -0.4974786839904084)
    (('his', 'was'), -0.49824259404985227)
    (('the', '.'), -0.5004099993516853)
    ((',', 'court'), -0.5004897312630767)
    (('law', 'the'), -0.5007476851526533)
    (('they', 'a'), -0.5010006185353753)
    ((',', ';'), -0.5017513798269881)
    ((',', 'answer'), -0.5017513798269881)
    ((',', 'appellant'), -0.5017513798269881)
    ((',', 'prisoner'), -0.5017513798269881)
    ((',', 'received'), -0.5017513798269881)
    (('prove', ','), -0.5017513798269881)
    (('section', ','), -0.5017513798269881)
    (('circuit', 'to'), -0.5018529566094614)
    (('.', 'could'), -0.5020814782246958)
    (('on', 'and'), -0.5025480402255873)
    (('the', 'that'), -0.5025912084407693)
    (('of', 'then'), -0.5029190812593214)
    (('be', '1'), -0.5032742811375286)
    (('.', 'plaintiff'), -0.5044407510986204)
    (('the', 'his'), -0.5048779342973795)
    (('a', '’'), -0.5049936879012371)
    (('the', ';'), -0.5055717044674672)
    (('.', 'would'), -0.5064599187352705)
    (('of', 'if'), -0.5074142399481012)
    (('see', '.'), -0.5082075810215372)
    (('what', '.'), -0.5082075810215372)
    (('have', 'that'), -0.5084329317000496)
    (('the', 'before'), -0.5099524912032258)
    (('the', 'with'), -0.510114582663526)
    (('in', '“'), -0.5111507374418025)
    (('case', 'which'), -0.5136379694495545)
    (('any', 'the'), -0.5139230739003864)
    (('.', 'party'), -0.5151771358363817)
    ((',', 'clerk'), -0.5155571793520188)
    ((';', 'by'), -0.5159868164124539)
    (('by', 'on'), -0.5159868164124539)
    (('on', 'by'), -0.5159868164124539)
    (('to', ','), -0.5162858336713896)
    (('.', 'before'), -0.5163354619049372)
    (('before', '.'), -0.5163354619049372)
    (('note', 'and'), -0.5180703123396384)
    (('against', 'of'), -0.5187066966095628)
    (('damages', 'the'), -0.5193775039924944)
    (('the', 'granted'), -0.5193775039924944)
    (('being', ','), -0.5201299091418434)
    (('and', 'when'), -0.5213830463804747)
    (('such', 'the'), -0.521595329242313)
    ((',', 'note'), -0.5217891330635318)
    (('s', 'it'), -0.5225967160330818)
    (('should', 'that'), -0.523717660150993)
    ((',', 'oath'), -0.5237776861569863)
    (('during', ','), -0.5237776861569863)
    (('&', 'the'), -0.524614152358641)
    (('and', 'will'), -0.5257882255004773)
    (('will', 'and'), -0.5257882255004773)
    (('been', 'that'), -0.5273516546187302)
    (('this', 'as'), -0.5273746837590814)
    (('and', 'opinion'), -0.5279857812047197)
    (('not', 'case'), -0.527993262426623)
    (('that', 'of'), -0.5286681825040596)
    (('not', 'said'), -0.5293359289180337)
    (('bond', 'the'), -0.5308731428303233)
    (('correct', 'the'), -0.5308731428303233)
    (('the', 'john'), -0.5308731428303233)
    (('trespass', 'the'), -0.5308731428303233)
    (('decided', '.'), -0.5324551272682143)
    ((',', 'whole'), -0.5331155507363263)
    (('this', 'he'), -0.5341162616381361)
    (('see', 'the'), -0.5357719422431657)
    (('court', 'at'), -0.537042017939644)
    ((',', 'common'), -0.5382772558521012)
    ((',', 'manner'), -0.5382772558521012)
    (('money', 'the'), -0.5384863269401983)
    (('proper', 'the'), -0.5384863269401983)
    (('him', 'of'), -0.5388845785471936)
    (('of', 'plaintiffs'), -0.5388845785471936)
    (('this', 'of'), -0.540102556703868)
    (('decree', ','), -0.5415523878849626)
    (('upon', 'is'), -0.5456628018058325)
    (('a', 'be'), -0.5460343721498795)
    (('had', 'it'), -0.5464227297593354)
    (('but', 'to'), -0.547284385856468)
    (('to', 'but'), -0.547284385856468)
    (('liable', 'the'), -0.5479466561892643)
    (('public', 'the'), -0.5479466561892643)
    (('the', 'either'), -0.5479466561892643)
    (('of', 'did'), -0.5488686671198195)
    (('the', 'be'), -0.5490982021124964)
    (('act', 'that'), -0.549114437216474)
    (('case', 'a'), -0.5492618791802748)
    (('is', 'state'), -0.5495567179103205)
    (('this', 'by'), -0.5522872537245682)
    (('action', 'the'), -0.5529300637035206)
    (('in', 'as'), -0.5538213536878267)
    (('deed', 'the'), -0.5556410479792646)
    (('officer', 'the'), -0.5556410479792646)
    (('in', 'could'), -0.5561844910563067)
    (('in', 'person'), -0.5561844910563067)
    (('some', ','), -0.5561991638493637)
    ((',', 'testimony'), -0.5561991638493673)
    ((',', 'title'), -0.5561991638493673)
    ((',', 'words'), -0.5561991638493673)
    (('(', 'this'), -0.5585693053626741)
    (('commission', 'the'), -0.5600194884898428)
    (('every', 'the'), -0.5600194884898428)
    (('a', 'evidence'), -0.5613011288031906)
    (('in', 'and'), -0.5614734365140421)
    (('cause', 'and'), -0.5642293018045521)
    (('now', 'the'), -0.5648204747536631)
    (('the', 'legal'), -0.5648204747536631)
    (('may', 'it'), -0.5661859845807662)
    (('which', 'on'), -0.5667631780435123)
    ((',', 'verdict'), -0.5668464080488711)
    (('is', '”'), -0.5675906407405868)
    ((',', 'out'), -0.5684774936278032)
    (('an', 'be'), -0.5693041514727284)
    (('the', 'they'), -0.5701083061134256)
    (('1', 'is'), -0.5705744991794042)
    (('who', ','), -0.5706987335444786)
    (('person', 'and'), -0.5708219791149816)
    (('.', 'upon'), -0.5719834914548514)
    (('only', '.'), -0.5719834914548514)
    (('point', 'of'), -0.5735307213540892)
    (('to', '’'), -0.5735887102083765)
    (('’', 'to'), -0.5735887102083765)
    (('fact', ','), -0.5739011655828214)
    (('following', ','), -0.5739011655828214)
    (('be', 'he'), -0.5743573791990109)
    (('which', 'in'), -0.5750625012619963)
    (('bank', 'the'), -0.5759610323588618)
    (('he', 'the'), -0.5759610323588618)
    (('the', '1825'), -0.5759610323588618)
    (('or', 'and'), -0.5761504987121704)
    (('a', 'him'), -0.576408021193398)
    (('in', 'v.'), -0.5765044754359003)
    (('not', ','), -0.5768800950758504)
    (('this', 'a'), -0.5776259993500688)
    (('from', '.'), -0.5784675250250508)
    (('in', 'is'), -0.5792849051876487)
    (('one', 'a'), -0.5801601560545038)
    (('plea', 'the'), -0.5807780486566365)
    (('an', ','), -0.5810903393910749)
    ((';', 'not'), -0.5811184710205808)
    (('.', 'act'), -0.5816986462880642)
    (('act', '.'), -0.5816986462880642)
    (('new', 'the'), -0.5824743393175353)
    (('which', 'for'), -0.5833143543073582)
    (('.', 'he'), -0.5838020072684351)
    ((',', 'costs'), -0.5842135400189612)
    (('the', 'him'), -0.5860146970227866)
    (('an', 'and'), -0.5860887357682891)
    (('evidence', 'and'), -0.5860887357682891)
    (('of', 'judgment'), -0.5874363233793041)
    ((',', 'exercise'), -0.587908023576702)
    ((',', 'facts'), -0.587908023576702)
    ((',', 'reverse'), -0.587908023576702)
    ((',', 'writing'), -0.587908023576702)
    (('most', ','), -0.587908023576702)
    (('act', 'to'), -0.5891645164628301)
    (('it', 'the'), -0.5901748915785632)
    (('his', 'v.'), -0.5902314585474144)
    (('in', 'on'), -0.5910040451310188)
    (('.', 'were'), -0.5913488163217835)
    (('decision', ','), -0.5913885923116631)
    (('has', 'the'), -0.5920134376442299)
    (('if', ','), -0.5923031375346142)
    (('in', 'trial'), -0.5929844092592624)
    (('court', 'case'), -0.5954320606132022)
    (('been', 'v.'), -0.595685888639526)
    (('note', 'the'), -0.5959987855954054)
    (('have', 'not'), -0.5970600148896033)
    ((')', 'was'), -0.5972316886812159)
    (('not', 'if'), -0.5976696472844303)
    (('as', 'that'), -0.5977182415947695)
    (('of', 'jurisdiction'), -0.5977782676007664)
    (('courts', 'the'), -0.5979873386888599)
    (('be', 'not'), -0.5987663368400042)
    ((',', 'their'), -0.5992678857412521)
    (('the', 'upon'), -0.5995478526764764)
    (('.', 'that'), -0.5999180950469558)
    (('had', ','), -0.6003989677599009)
    ((',', ','), -0.6011298542449879)
    (('to', 'they'), -0.6020171185348957)
    (('.', 'of'), -0.6025593657325636)
    (('any', 'is'), -0.6029959768717816)
    (('is', 'any'), -0.6029959768717816)
    (('necessary', 'the'), -0.6034417687809714)
    (('one', 'and'), -0.6049477630196058)
    (('1', 'it'), -0.607304971906359)
    (('and', 'question'), -0.6076218973179337)
    (('said', '.'), -0.6082226348861788)
    (('with', 'that'), -0.6083953749747906)
    (('must', 'it'), -0.6084967894387283)
    ((',', 'remedy'), -0.6086665837435028)
    (('their', 'the'), -0.60934720085341)
    (('had', 'of'), -0.6102692117839617)
    (('only', 'the'), -0.6110434915143088)
    ((',', '’'), -0.6110610988838836)
    (('would', '.'), -0.6133751226517816)
    (('to', 'we'), -0.613698923582529)
    (('an', 'not'), -0.6140330934933154)
    (('statute', 'a'), -0.6145431500801699)
    (('by', 'to'), -0.6151525207240311)
    (('to', 'general'), -0.6160239765287727)
    (('to', 'governor'), -0.6160239765287727)
    (('as', ','), -0.6165717818288456)
    (('in', 'no'), -0.6174374529838005)
    (('shall', ','), -0.6189349191973257)
    (('it', ','), -0.6195878701208457)
    (('be', '“'), -0.620034953593656)
    (('if', 'to'), -0.6204190904866849)
    ((',', 'given'), -0.6207394160440742)
    (('a', 'cases'), -0.6208021405518522)
    (('.', 'such'), -0.6215810177411996)
    ((',', 'doubt'), -0.6233133597079039)
    (('of', 'can'), -0.6237734761337101)
    (('the', 'all'), -0.6248706328398086)
    (('in', 'to'), -0.6248782891908746)
    (('such', 'in'), -0.6256433480732149)
    (('the', 'it'), -0.6263345548739814)
    (('into', ','), -0.627282261910846)
    (('court', 'as'), -0.6273161216329903)
    (('.', 'does'), -0.6293166665208041)
    (('.', 'one'), -0.6293166665208041)
    (('.', 'taken'), -0.6293166665208041)
    (('before', 'in'), -0.6293321637901172)
    (('as', '.'), -0.6303024506663597)
    (('great', 'the'), -0.6304088163812409)
    (('said', 'of'), -0.6317255995241027)
    (('after', '.'), -0.6339931537302768)
    ((',', 'matter'), -0.6342016758506368)
    (('a', 'any'), -0.6353017102469671)
    (('had', '.'), -0.6355486202323171)
    (('in', 'there'), -0.6364066895503875)
    (('consideration', 'to'), -0.6367825366955735)
    (('to', 'consideration'), -0.6367825366955735)
    (('to', 'states'), -0.6367825366955735)
    (('case', 'in'), -0.6374145100879751)
    ((',', 'plea'), -0.6378129294030153)
    (('it', 'it'), -0.6398430263060746)
    (('such', 'and'), -0.6402808361318861)
    (('of', 'after'), -0.6404226050092561)
    (('of', 'for'), -0.6423438200342098)
    ((',', 'defendants'), -0.6426139156668356)
    (('the', 'what'), -0.6426871461596768)
    (('be', 'which'), -0.6433047329165049)
    (('the', 'when'), -0.643705638994696)
    (('them', 'of'), -0.6450839823791235)
    (('in', 'plea'), -0.6454518291533944)
    (('in', 'were'), -0.6454518291533944)
    (('in', 'who'), -0.6454518291533944)
    (('the', 'without'), -0.6463503602502598)
    (('.', 'error'), -0.6479323446881509)
    (('he', '.'), -0.6479323446881509)
    (('i', 'the'), -0.6481108181146986)
    (('the', 'i'), -0.6481108181146986)
    (('up', ','), -0.6493085682408477)
    (('party', 'of'), -0.6508896040603247)
    (('was', ','), -0.651101850063899)
    (('cases', 'to'), -0.6514293126599746)
    (('suit', 'to'), -0.6514293126599746)
    (('justice', 'not'), -0.6515077989119789)
    (('of', 'as'), -0.6519986585986501)
    (('the', '1819'), -0.6531288928813233)
    (('the', 'very'), -0.6531288928813233)
    (('than', ','), -0.6547629983329557)
    (('be', 'may'), -0.6548003717543338)
    (('a', 'the'), -0.6565723968628951)
    (('his', 'a'), -0.6573280165769653)
    (('for', 'by'), -0.6580688747601577)
    (('court', '('), -0.6585107432365795)
    (('authority', ','), -0.65929265681347)
    (('other', 'and'), -0.6600893172120657)
    (('and', ';'), -0.6600893172120692)
    (('.', 'record'), -0.6602106744665868)
    (('a', 's'), -0.6621017695906808)
    (('general', 'the'), -0.6621176761085756)
    (('several', 'the'), -0.6621176761085756)
    (('court', 'but'), -0.6623630687321906)
    (('circuit', 'a'), -0.663870862443737)
    ((')', 'of'), -0.6643458845397525)
    (('or', '.'), -0.6646497396330844)
    (('.', 'being'), -0.6663108740030061)
    (('either', ','), -0.6663820815997887)
    (('not', 'of'), -0.6672751147702947)
    (('his', 'in'), -0.6674781354833961)
    (('this', 'in'), -0.6679464669404815)
    (('.', 'note'), -0.6679700979246945)
    (('(', 'is'), -0.6680910050936681)
    (('was', 'was'), -0.6692881353860791)
    (('the', 'although'), -0.6699371805678744)
    (('the', 'defense'), -0.6699371805678744)
    (('is', 'must'), -0.6713019902626947)
    (('upon', '.'), -0.671519165005769)
    (('same', 'the'), -0.6719879201326364)
    (('it', 'such'), -0.6723853884126925)
    (('should', ','), -0.673155310195213)
    ((',', 'decision'), -0.6738507525036361)
    (('circuit', 'in'), -0.6740209813501643)
    (('of', 'had'), -0.6743995492036774)
    ((',', 'courts'), -0.6757807796020359)
    ((',', 'form'), -0.6757807796020359)
    (('for', 'as'), -0.6775504241531216)
    (('however', 'of'), -0.6818425323892399)
    (('it', 'no'), -0.6827951714906213)
    (('the', 'because'), -0.6828762362753729)
    (('our', 'the'), -0.6828762362753764)
    (('pleaded', 'the'), -0.6828762362753764)
    (('to', 'to'), -0.6841745912830959)
    (('given', '.'), -0.6844582207132675)
    (('upon', 'in'), -0.6849801933400315)
    (('and', 'circuit'), -0.6886584694088391)
    (('a', 'was'), -0.6900941830873464)
    (('it', 'any'), -0.692193869492872)
    (('united', 'the'), -0.6931445717291993)
    (('person', '.'), -0.6947265561670939)
    (('any', 'that'), -0.696268131000128)
    (('be', 'must'), -0.6971111766122959)
    (('his', 'the'), -0.6975230122397775)
    (('to', 'if'), -0.6984216024879579)
    ((',', 'act'), -0.6985520872606941)
    (('was', 'of'), -0.6998764552194991)
    (('was', 'this'), -0.7051618029743665)
    (('was', 'court'), -0.7055675672973543)
    ((',', 'intended'), -0.7082022572944133)
    ((',', 'part'), -0.7082022572944133)
    ((',', 'time'), -0.7082022572944133)
    (('shown', ','), -0.7082022572944133)
    (('sufficient', ','), -0.7082022572944133)
    ((',', 'commission'), -0.7082022572944169)
    ((',', 'more'), -0.7082022572944169)
    (('reynolds', ','), -0.7082022572944169)
    (('secretary', 'the'), -0.7084113283825104)
    (('the', 'think'), -0.7084113283825104)
    (('think', 'the'), -0.7084113283825104)
    (('he', ','), -0.7132554850207029)
    (('that', 'from'), -0.7158083173957728)
    (('for', 'be'), -0.7164394375467182)
    (('the', 'now'), -0.7168235681987127)
    (('is', 'of'), -0.7171669967910965)
    (('.', 'jury'), -0.7200821304439877)
    (('of', 'when'), -0.7221064026029644)
    (('where', 'of'), -0.7221064026029644)
    (('the', 'obtained'), -0.7235182207727213)
    (('the', 'provided'), -0.7235182207727213)
    ((',', 'office'), -0.7250760758588086)
    (('a', 'act'), -0.725187213764162)
    (('have', 'it'), -0.7267510915292377)
    (('state', 'be'), -0.7273689977049784)
    (('the', 'such'), -0.73018195105373)
    (('may', 'and'), -0.7304786451034637)
    (('the', 'whether'), -0.7335023093453437)
    (('three', 'the'), -0.7335023093453437)
    (('general', '.'), -0.7340889884378647)
    (('assigned', 'of'), -0.7352817913506975)
    (('suit', 'of'), -0.7352817913506975)
    (('for', 'court'), -0.735497773362404)
    (('party', '.'), -0.7375695571728293)
    (('a', 'defendant'), -0.7396383700617015)
    (('of', 'by'), -0.7396469839120989)
    (('having', 'the'), -0.7405917341316623)
    (('lockwood', 'the'), -0.7405917341316623)
    (('the', 'according'), -0.7405917341316623)
    (('the', 'liable'), -0.7405917341316623)
    (('the', 'well'), -0.7405917341316623)
    (('court', 'his'), -0.7409729034285562)
    (('and', 'state'), -0.7441535820005427)
    (('this', 'on'), -0.7445360413085069)
    (('at', 'and'), -0.7456650489181413)
    ((',', 'taken'), -0.7461701074934339)
    (('the', '”'), -0.7470065736950886)
    (('the', 'as'), -0.7473649627270866)
    (('are', 'and'), -0.7475521584624083)
    (('be', 'said'), -0.7476413927312393)
    (('of', 'so'), -0.7483379441761464)
    (('this', 'this'), -0.7484150009282473)
    (('other', 'of'), -0.7497813610458124)
    (('who', 'of'), -0.7497813610458124)
    (('said', 'in'), -0.7497884889681323)
    (('justice', 'a'), -0.7507789276669037)
    (('after', ','), -0.7508465947029066)
    (('no', 'by'), -0.751006846076649)
    (('of', 'before'), -0.7515836036797978)
    (('nor', 'the'), -0.7532655641667709)
    (('.', 'demurrer'), -0.754847548604662)
    (('case', 'by'), -0.7550423593118083)
    ((',', 'sale'), -0.755507972072774)
    (('we', ','), -0.755507972072774)
    (('was', 'if'), -0.7558018698174926)
    ((',', 'payment'), -0.7647857856607807)
    ((',', 'prove'), -0.7647857856607807)
    (('commenced', ','), -0.7647857856607807)
    (('received', ','), -0.7647857856607807)
    (('all', ','), -0.7647857856607843)
    (('may', 'the'), -0.7653383964673459)
    (('have', 'of'), -0.7698390133870667)
    (('we', 'of'), -0.7706148644629813)
    (('the', 'we'), -0.7708239355510784)
    (('could', ','), -0.7709380126423753)
    ((')', 'that'), -0.772889412603039)
    (('had', 'that'), -0.772889412603039)
    (('note', 'that'), -0.772889412603039)
    (('of', 'was'), -0.7738770366632792)
    (('one', 'to'), -0.7742860604455046)
    (('must', ','), -0.7765082706853228)
    (('this', '1'), -0.7785349893045819)
    (('justice', 'it'), -0.7811988755516168)
    (('of', 'but'), -0.7822028387381899)
    (('the', 'make'), -0.7824119098262869)
    (('the', 'sufficient'), -0.7824119098262869)
    (('would', 'the'), -0.7824119098262869)
    (('the', 'coles'), -0.7824119098262905)
    (('the', 'more'), -0.7824119098262905)
    (('and', ','), -0.7829216106741193)
    (('in', 'he'), -0.7829553529033291)
    (('without', 'of'), -0.7836448129120974)
    (('a', 'so'), -0.7858613868223472)
    (('within', ','), -0.7883726059783989)
    (('.', 'said'), -0.7887948805279983)
    (('office', '.'), -0.7887948805280018)
    (('his', 'that'), -0.7893775353916084)
    (('when', 'to'), -0.7902570173572911)
    (('said', 'not'), -0.7923703347518263)
    (('then', 'of'), -0.7924256984543057)
    (('to', 'county'), -0.7929017386128514)
    (('the', 'so'), -0.7954680626517359)
    (('by', 'if'), -0.7955723985100924)
    (('in', 'only'), -0.796011505728778)
    (('in', 'so'), -0.796011505728778)
    (('the', 'by'), -0.7977482202086925)
    (('of', 'with'), -0.7994121287704132)
    (('all', '.'), -0.7999354381332004)
    (('was', 'or'), -0.8003622078570309)
    (('.', 'power'), -0.8054736216746328)
    (('before', 'that'), -0.8056793477207123)
    (('been', 'it'), -0.808679611973723)
    (('what', ','), -0.8094855931315976)
    (('only', 'and'), -0.8106489937874493)
    (('of', 'no'), -0.8107719909349633)
    ((',', 'action'), -0.8112957502585196)
    (('whether', 'of'), -0.8119030729536121)
    (('property', 'the'), -0.8141207695536252)
    (('to', 'ought'), -0.8143207222477606)
    (('as', 'be'), -0.8147086632847298)
    (('or', 'as'), -0.8152201665841474)
    (('said', 'the'), -0.8163592417496268)
    (('to', 'only'), -0.8164885589304696)
    (('a', 'been'), -0.8189016485863583)
    (('”', '.'), -0.8189778860243777)
    ((',', 'jurors'), -0.8192335696831599)
    (('’', 'and'), -0.8192879120613199)
    (('at', 'to'), -0.8218939419525597)
    ((',', 'of'), -0.8223732772137247)
    (('the', 'executed'), -0.8230538943236354)
    ((',', 'affidavit'), -0.8268467537930348)
    ((',', 'appointment'), -0.8268467537930348)
    ((',', 'settled'), -0.8268467537930348)
    (('notice', ','), -0.8268467537930348)
    (('take', ','), -0.8268467537930348)
    (('has', 'that'), -0.8277977537054291)
    (('be', 'judgment'), -0.8288829986703021)
    (('and', 'trial'), -0.8300143186543814)
    (('could', 'of'), -0.8304390243910369)
    (('person', 'of'), -0.8304390243910369)
    (('judgment', 'to'), -0.83289146994154)
    (('also', ','), -0.8337331393782748)
    (('party', 'the'), -0.8355232462858488)
    (('and', 'right'), -0.83641208985253)
    (('of', 'point'), -0.8365651271878818)
    (('of', 'see'), -0.8365651271878818)
    (('will', ','), -0.836935571416614)
    (('(', 'the'), -0.8374532733842486)
    ((',', '.'), -0.8379701322647435)
    (('the', 'found'), -0.8389954381926543)
    (('the', 'he'), -0.8389954381926579)
    (('and', 'and'), -0.839145330406506)
    (('has', ','), -0.839731879999718)
    (('is', 'justice'), -0.8440040763755796)
    (('on', 'court'), -0.8444774791824159)
    (('that', 'as'), -0.8456457550383547)
    ((',', 'statutes'), -0.8457057810443516)
    (('until', ','), -0.8457057810443516)
    (('.', 'be'), -0.8467296496121435)
    (('in', 'was'), -0.8470856903230484)
    (('an', 'court'), -0.8513968931222031)
    (('judgment', 'or'), -0.8514208082997143)
    ((';', 'to'), -0.8517954276664206)
    (('was', 'have'), -0.8547279109735832)
    (('s', 'a'), -0.8547468475330788)
    (('the', 'if'), -0.8555466144565038)
    (('to', 's'), -0.8562276739816816)
    (('case', 'this'), -0.8580607021240034)
    (('has', '.'), -0.8603819627770193)
    (('due', 'of'), -0.8608126734345589)
    (('and', 'statute'), -0.8617231783817196)
    (('the', 'within'), -0.8625822585102725)
    (('the', 'been'), -0.8664761746147605)
    (('of', 'has'), -0.8668114140560021)
    (('a', 'shall'), -0.8679624670372448)
    (('shall', 'a'), -0.8679624670372448)
    (('.', 'state'), -0.868058159052655)
    (('or', 'that'), -0.8691047278623429)
    (('.', 'without'), -0.8703247660245985)
    ((',', 'this'), -0.871295225254304)
    (('state', 'a'), -0.8713690684804902)
    ((',', 'consideration'), -0.8717009895772918)
    (('ever', ','), -0.8717009895772954)
    (('remedy', ','), -0.8717009895772954)
    (('(', 'it'), -0.8727661151346915)
    (('no', 'and'), -0.8730830405462662)
    (('first', '.'), -0.8734920451032835)
    (('which', 'of'), -0.8753122431296703)
    (('1819', 'the'), -0.8755213142177709)
    (('those', 'the'), -0.8755213142177709)
    ((',', 'record'), -0.8765997889901342)
    (('.', 'should'), -0.8782299641099449)
    (('to', 'et'), -0.8800536877079352)
    (('have', ','), -0.8802630030807173)
    (('state', 'in'), -0.8815191873869175)
    (('or', 'is'), -0.8827477776505113)
    (('.', '.'), -0.8833104076190779)
    ((',', 'right'), -0.8845250299348777)
    (('of', 'should'), -0.8846594153889242)
    (('scam.', '.'), -0.8860920818829179)
    (('been', 'of'), -0.888293409856665)
    (('in', 'against'), -0.8883083530495597)
    (('on', 'was'), -0.8912537869986927)
    (('case', 'on'), -0.8921495927032836)
    (('to', 'error'), -0.892437412163769)
    (('jurors', 'the'), -0.8934432222150335)
    (('and', 'below'), -0.897128514512918)
    (('would', 'and'), -0.897128514512918)
    (('this', 'if'), -0.8985907413222911)
    (('and', 'record'), -0.8988761767991846)
    (('only', 'of'), -0.900341037621196)
    (('judgment', ','), -0.9003531769716986)
    (('in', 'said'), -0.9017915824131819)
    (('to', 'he'), -0.9018361101660162)
    (('”', 'of'), -0.9063273326869279)
    (('note', 'a'), -0.9083202046533856)
    (('the', 'against'), -0.9097912163025157)
    ((',', 'estate'), -0.9098361184640638)
    (('1', 'a'), -0.9103087577468365)
    (('a', ':'), -0.9103087577468365)
    (('plaintiff', 'the'), -0.9105167355739745)
    (('and', 'court'), -0.9118084098180255)
    (('was', 'be'), -0.9153279219775499)
    ((',', 'case'), -0.9158936292844331)
    (('and', '.'), -0.9195482688220338)
    (('of', 'plea'), -0.9197063624881245)
    (('was', 'it'), -0.9198229391954627)
    (('suit', 'the'), -0.9199154335762216)
    (('the', 'statutes'), -0.9199154335762252)
    (('the', 'a'), -0.9213050901687154)
    ((',', 'dollars'), -0.9223270626472626)
    ((',', 'three'), -0.9223270626472626)
    (('a', 'it'), -0.9313511733087232)
    (('have', 'and'), -0.9316857365492872)
    (('the', 'brought'), -0.9317895338645137)
    (('an', 'is'), -0.932060253040774)
    (('.', 'evidence'), -0.9323857341568527)
    (('an', '.'), -0.9323857341568527)
    ((',', 'first'), -0.933761957709546)
    (('the', 'but'), -0.9344150032713401)
    (('and', ':'), -0.9350963647119386)
    (('and', 'law'), -0.9364205456593169)
    ((',', 'jury'), -0.9364712449675316)
    (('.', 'been'), -0.9384474869440531)
    (('in', 'judgment'), -0.9399644664603564)
    (('that', 's'), -0.9417084768191835)
    (('in', 'defendant'), -0.9424335669105268)
    (('on', 'on'), -0.9452748012972414)
    (('the', '5'), -0.945910642109169)
    (('the', 'ever'), -0.945910642109169)
    (('of', 'to'), -0.9461725782211055)
    (('under', 'and'), -0.9469704650002271)
    (('.', 'which'), -0.94749262654706)
    (('their', 'of'), -0.948275514684898)
    (('and', 'any'), -0.9495959344070499)
    (('s', 'the'), -0.9512309740424314)
    (('the', 'give'), -0.9523369112685991)
    (('two', 'the'), -0.9523369112685991)
    (('of', 'might'), -0.9539220778260393)
    (('any', 'it'), -0.9552282753266645)
    (('his', 'it'), -0.9552282753266645)
    (('it', 'his'), -0.9552282753266645)
    (('sheriff', 'the'), -0.9552485066885055)
    (('not', 'has'), -0.9570668213923277)
    (('as', 'which'), -0.9605595297481848)
    (('real', 'the'), -0.96298415546811)
    (('be', 'this'), -0.9706539522320092)
    ((',', 'appeal'), -0.9712366631282094)
    ((',', 'equity'), -0.9712366631282094)
    (('executed', ','), -0.9712366631282094)
    (('held', ','), -0.9712366631282094)
    (('and', 'act'), -0.9723672420657117)
    (('there', 'to'), -0.9723855684800107)
    (('common', 'the'), -0.975056987768685)
    (('is', 'before'), -0.9754415994261585)
    (('that', 'or'), -0.976019931778854)
    (('the', 'after'), -0.9770593406798334)
    (('to', 'justice'), -0.977326309750282)
    (('for', 'for'), -0.9783771538249368)
    (('that', '’'), -0.9809976079332401)
    (('’', 'that'), -0.9809976079332401)
    ((')', 'for'), -0.9833057049715386)
    (('is', 'case'), -0.9835239457240164)
    (('no', ','), -0.9883101764871505)
    (('reversed', 'of'), -0.9885682206329314)
    (('they', ','), -0.9889386648616671)
    ((',', 'parties'), -0.9896151924430647)
    (('be', 'state'), -0.990403403538771)
    (('writ', 'the'), -0.9909985316377039)
    (('.', 'suit'), -0.9918867459055143)
    (('the', 'no'), -0.9954056331604875)
    (('2', 'the'), -0.9965367151791327)
    (('the', 'dollars'), -0.9965367151791362)
    (('would', ','), -0.9977088744894012)
    (('.', 'execution'), -1.000603963006622)
    (('execution', '.'), -1.000603963006622)
    (('defendant', 'of'), -1.0016752092744063)
    (('of', 'there'), -1.0037706272766016)
    (('there', 'of'), -1.0037706272766016)
    (('the', 'clearly'), -1.0048043311627382)
    (('the', 'do'), -1.0048043311627382)
    (('such', 'to'), -1.0060163463612923)
    (('so', 'the'), -1.006972167845447)
    (('action', 'to'), -1.0093367046529025)
    (('to', 'note'), -1.0093367046529025)
    (('court', 'been'), -1.0094617393544567)
    (('bill', 'the'), -1.0106808974994088)
    ((',', 'new'), -1.0107650273148465)
    ((',', 'principle'), -1.0107650273148465)
    (('of', 'given'), -1.0128157668796085)
    (('a', 'he'), -1.0138133335006962)
    (('and', 'case'), -1.0146220775313601)
    (('ill.', 'and'), -1.0162331274373422)
    (('“', 'to'), -1.0170546178137378)
    (('by', 'is'), -1.0223986687120288)
    (('an', 'that'), -1.0253324071691168)
    (('to', 'below'), -1.0258248274414683)
    (('.', 'have'), -1.02644396794188)
    (('any', '.'), -1.0284126219306273)
    (('of', 'trial'), -1.030737674876871)
    (('on', 'is'), -1.036649154317974)
    ((',', 'debt'), -1.038350858986746)
    (('of', 'verdict'), -1.0413849190763784)
    ((',', 'bill'), -1.0433864488840427)
    (('of', 'where'), -1.0440344974903262)
    (('court', 'an'), -1.0440419710646012)
    (('the', 'say'), -1.045446315660083)
    ((',', 'evidence'), -1.049239175129479)
    ((',', 'award'), -1.0492391751294825)
    ((',', 'contract'), -1.0492391751294825)
    (('in', 'by'), -1.0503549512985266)
    ((',', 'below'), -1.0521566585117768)
    (('.', ':'), -1.059000941764051)
    ((',', 'him'), -1.0643460675196899)
    (('remanded', ','), -1.0643460675196899)
    (('.', 'at'), -1.0654898359455096)
    (('does', ','), -1.0680982023807992)
    (('be', 'court'), -1.0705953901059182)
    (('give', ','), -1.0707723366791235)
    (('defendants', 'the'), -1.079393647583423)
    (('office', 'the'), -1.079393647583423)
    (('has', 'and'), -1.0811255571123546)
    (('states', 'of'), -1.0832050947710066)
    ((',', 'error'), -1.0867138805481424)
    ((',', 'section'), -1.0867138805481424)
    (('has', 'it'), -1.0867578980319657)
    (('s', 'in'), -1.0872893877759537)
    (('such', 'that'), -1.0914971491987906)
    (('of', 'have'), -1.091767108274432)
    (('sale', 'the'), -1.0927520304384402)
    (('affirmed', 'of'), -1.0978518707354077)
    (('trial', 'to'), -1.0988882896311942)
    ((',', 'counsel'), -1.098992210326582)
    ((',', 'property'), -1.1024811964064618)
    ((',', 'statute'), -1.1024811964064618)
    (('v.', ')'), -1.1042580524576273)
    ((',', 'suit'), -1.1087401868781406)
    ((',', 'decide'), -1.1087401868781441)
    (('in', 'statute'), -1.110120096156841)
    (('this', 'and'), -1.1114272538030292)
    (('debt', 'the'), -1.1125605115186197)
    (('court', 'he'), -1.1194845266822853)
    (('award', 'the'), -1.1234488276613561)
    (('is', 'or'), -1.1237558771543057)
    (('in', 'from'), -1.1244236341863356)
    (('to', 'record'), -1.1271081632786526)
    ((';', 'of'), -1.128292984299545)
    (('of', 'made'), -1.1293015093604595)
    (('the', 'from'), -1.1321006979142645)
    (('not', 'court'), -1.1323978454854497)
    ((',', 'deed'), -1.1335080920270855)
    (('to', 'affirmed'), -1.1368561398302148)
    (('are', 'a'), -1.1378020507761484)
    (('that', 'for'), -1.1464724847101664)
    (('brought', ','), -1.1470864985276243)
    (('is', 'has'), -1.149563098855932)
    (('to', 'other'), -1.1513557095253297)
    (('having', ','), -1.151808908770029)
    (('of', 'not'), -1.1527019419405384)
    (('by', 'have'), -1.1575328454999756)
    ((',', 'following'), -1.1588636663039793)
    (('be', 'v.'), -1.1593198799201687)
    (('to', 'was'), -1.1603444927525821)
    (('the', 'received'), -1.160923533080016)
    (('1', 'the'), -1.1698936865845724)
    (('at', 'that'), -1.170409150623854)
    (('that', 'at'), -1.170409150623854)
    (('which', ','), -1.1721493570542023)
    (('will', 'of'), -1.1739705586941866)
    (('.', 'person'), -1.180153383337334)
    (('and', '’'), -1.18185799144603)
    (('of', 'question'), -1.1827407683219207)
    (('affirmed', 'the'), -1.1829498394100142)
    (('the', 'decide'), -1.1829498394100177)
    (('on', 'for'), -1.1842183988975385)
    (('the', 'necessary'), -1.1884042695021257)
    (('court', 'by'), -1.1927970729611772)
    (('clearly', ','), -1.193629084464657)
    (('more', ','), -1.193629084464657)
    (('the', 'interest'), -1.1974494091051326)
    (('a', 'that'), -1.1984598406497753)
    (('judgment', 'it'), -1.199700230134031)
    (('and', 'an'), -1.20757711251456)
    ((',', 'constitution'), -1.2163491609647394)
    ((':', ','), -1.2176745584313053)
    (('that', ','), -1.2182490512376596)
    (('of', 'be'), -1.2216297817155173)
    (('post', ','), -1.2227754301241731)
    (('can', 'the'), -1.2234446176380764)
    (('either', 'the'), -1.2260185613019026)
    (('if', 'of'), -1.227306320755364)
    (('of', 'we'), -1.2300464831002778)
    (('such', 'of'), -1.2300464831002778)
    (('they', 'the'), -1.2330733188358565)
    ((',', 'third'), -1.234271068962002)
    (('for', 'not'), -1.23509956789972)
    (('there', ','), -1.23972549905411)
    ((',', 'power'), -1.2442551575346243)
    (('laws', 'the'), -1.2447551238834897)
    (('the', 'them'), -1.2447551238834897)
    (('be', 'of'), -1.2454765236698826)
    ((')', 'it'), -1.2468624479004298)
    (('said', 'is'), -1.2479010180492232)
    (('.', 'to'), -1.248847181471902)
    (('the', 'there'), -1.2498048141662679)
    (('a', 'said'), -1.2542115428914613)
    (('it', 'or'), -1.2573478891338468)
    (('in', 'plaintiff'), -1.2589834820713222)
    (('of', 'evidence'), -1.2607432803231937)
    (('judgment', 'not'), -1.262654231436791)
    (('the', 'v.'), -1.262803147099234)
    (('justice', 'to'), -1.2668329269452663)
    (('on', 'to'), -1.2668329269452663)
    ((',', 'point'), -1.268917211768894)
    (('who', '.'), -1.2694207214344218)
    (('.', 'by'), -1.2737859139958232)
    (('the', 'an'), -1.2754519211064057)
    (('where', 'to'), -1.2756838445275314)
    (('of', 'him'), -1.275850172713401)
    (('by', 'not'), -1.2773613682196512)
    (('by', 'it'), -1.2815215627754313)
    (('its', 'the'), -1.2849122503554717)
    (('that', 'with'), -1.2864672800874288)
    (('of', 'right'), -1.288674213070987)
    ((',', 'these'), -1.2931647580155712)
    (('.', 'circuit'), -1.2979898736311917)
    (('judgment', 'this'), -1.3005409196408593)
    (('by', '('), -1.3035478691054259)
    (('his', 'be'), -1.3068770683340247)
    (('the', 'some'), -1.3084807214938756)
    (('the', 'yet'), -1.3084807214938756)
    (('that', 'that'), -1.314567815595396)
    ((')', 'by'), -1.315074122486454)
    (('of', 'would'), -1.3202442920718553)
    ((',', 'here'), -1.3248736177429095)
    (('of', 'whether'), -1.3264762457833719)
    (('is', 'been'), -1.3271642965738764)
    (('and', 'v.'), -1.3281075576607826)
    ((')', ','), -1.3299850315605077)
    (('can', ','), -1.3356480893370843)
    (('the', 'see'), -1.3431268643007677)
    (('in', 'justice'), -1.3458915472944888)
    ((',', 'law'), -1.3471035651263747)
    (('a', 'they'), -1.3489975250903257)
    (('to', 'statute'), -1.3529895706949802)
    (('?', ','), -1.3559005133635331)
    (('a', 'his'), -1.3577677347180561)
    (('in', 'will'), -1.359147643996753)
    (('and', 'justice'), -1.36052903535316)
    (('was', 'which'), -1.360739070299914)
    (('.', 'ill.'), -1.3625301258259022)
    (('to', 'no'), -1.3643494328595267)
    (('if', 'that'), -1.368864906046614)
    (('ill.', 'the'), -1.373020973688586)
    ((',', 'question'), -1.3756269182075442)
    (('be', 'is'), -1.3762251150247593)
    (('can', 'of'), -1.3827653766299157)
    (('the', 'any'), -1.3838625333360106)
    (('the', 'c.'), -1.3864832334951487)
    (('the', 'settled'), -1.3864832334951487)
    (('which', 'that'), -1.3888396058078811)
    (('a', 'have'), -1.3923249567544254)
    (('it', 'case'), -1.3946499332324684)
    (('but', ','), -1.3978621366822637)
    (('should', 'of'), -1.399232588218684)
    (('and', 'or'), -1.3992727366280917)
    (('not', 'and'), -1.4114102043553451)
    (('is', 'is'), -1.4118164733393002)
    (('is', 'have'), -1.4151607775717068)
    (('in', 'in'), -1.4155914151771185)
    (('of', 'been'), -1.4188081265554437)
    (('the', 'where'), -1.4213132176582484)
    ((',', 'fact'), -1.4218980721377719)
    (('i', ','), -1.4218980721377719)
    (('an', 'this'), -1.429527360360936)
    (('with', 'in'), -1.4320481910442027)
    (('the', 'gilm.'), -1.4344886064059814)
    (('in', 'must'), -1.4362238670153964)
    (('did', ','), -1.4369002354770224)
    (('to', 'should'), -1.4382368573134912)
    (('to', 'not'), -1.4432449780313057)
    (('.', 'right'), -1.4457434940748826)
    (('of', 'therefore'), -1.4457751741557132)
    (('to', 'is'), -1.4513166843574794)
    (('there', '.'), -1.453020659773813)
    (('to', 'must'), -1.456700920217088)
    (('court', 'or'), -1.4581300165145805)
    (('3', 'the'), -1.4604838149389252)
    (('by', 'judgment'), -1.4605569826624532)
    (('.', 'for'), -1.46284965978235)
    (('is', 'which'), -1.465492453121847)
    (('a', 'v.'), -1.4732449521379927)
    (('be', 'case'), -1.483264320406036)
    ((',', 'due'), -1.4858098359579692)
    ((',', 's'), -1.4870748031945453)
    (('has', 'of'), -1.488299790802273)
    (('the', 'between'), -1.4902311583329784)
    (('for', 'was'), -1.4927674639837)
    ((',', 'declaration'), -1.4966981521007021)
    (('he', 'and'), -1.4980325591030947)
    (('may', 'of'), -1.4982425940498487)
    (('that', 'and'), -1.5007814231437813)
    (('to', 'are'), -1.5018529566094614)
    (('?', 'of'), -1.5085109295036752)
    (('.', ';'), -1.5104288209382162)
    ((',', 'jurisdiction'), -1.5155571793520188)
    (('which', 'and'), -1.522585793462131)
    (('declaration', 'of'), -1.5237776861569863)
    (('not', 'was'), -1.5270974567220357)
    (('in', 'upon'), -1.5329770998949819)
    (('plaintiff', 'of'), -1.5332380154060523)
    (('must', 'of'), -1.5405533989078144)
    (('people', 'the'), -1.5479466561892643)
    (('been', 'is'), -1.549556717910324)
    (('in', 'have'), -1.5544781691059057)
    (('that', 'but'), -1.5587646072501933)
    (('ought', 'the'), -1.5709078046325757)
    (('at', 'in'), -1.5790244674144205)
    (('set', ','), -1.5826713752105555)
    (('a', 'had'), -1.5863921097660203)
    ((',', 'state'), -1.59063266091324)
    ((',', 'execution'), -1.591388592311663)
    (('shall', 'of'), -1.6080466030545928)
    ((',', 'land'), -1.6086665837435028)
    ((',', 'second'), -1.6086665837435028)
    (('of', '1'), -1.6097509092668396)
    (('the', '3'), -1.6124869083839748)
    (('or', 'this'), -1.6212229844744677)
    (('this', 'or'), -1.6212229844744677)
    (('s', 'of'), -1.62457832694448)
    (('the', 'smith'), -1.6304088163812374)
    (('ought', ','), -1.6342016758506368)
    (('’', 'a'), -1.6424972116511718)
    (('note', 'of'), -1.6484043406707336)
    (('’', 'in'), -1.652647330557599)
    (('“', ','), -1.6563633257747945)
    (('gilm.', 'the'), -1.656881027742429)
    (('the', '&'), -1.6621176761085756)
    (('the', 'scam.'), -1.6621176761085756)
    (('.', 'his'), -1.669958651018149)
    ((',', 'county'), -1.6716763812693003)
    (('by', 'as'), -1.6873907467806752)
    (('opinion', 'to'), -1.697324078630622)
    (('is', 'judgment'), -1.7181849147341843)
    (('(', ','), -1.7237152566657379)
    (('not', 'judgment'), -1.7480810586070312)
    (('’', 'of'), -1.756976862450017)
    (('rep.', 'the'), -1.7581643635796134)
    (('be', 'be'), -1.7604882722868425)
    (('post', 'the'), -1.782411909826287)
    (('by', 'be'), -1.785173449227841)
    (('a', 'this'), -1.787079364979018)
    (('(', 'to'), -1.7913595738044492)
    (('.', 'and'), -1.7964000380491711)
    (('on', 'not'), -1.8035108923570284)
    ((')', 'be'), -1.8209036622442376)
    (('from', 'to'), -1.822972592500669)
    (('error', 'of'), -1.8242867974094423)
    (('when', ','), -1.832530392296615)
    (('.', 'with'), -1.8336246619887824)
    (('would', 'of'), -1.8348174649016151)
    (('”', 'to'), -1.8384163978652204)
    ((',', 'assigned'), -1.845705781044348)
    (('court', 'v.'), -1.8564501208484927)
    (('’', 'the'), -1.8636079926742681)
    (('to', 'would'), -1.8738217339964187)
    (('it', 'judgment'), -1.8777721352466692)
    ((',', 'trial'), -1.878127258736729)
    (('which', '.'), -1.8909090981806926)
    (('it', 'court'), -1.8995188427403775)
    (('the', '2'), -1.9034273107876523)
    (('that', '.'), -1.9065794332810064)
    (('page', 'the'), -1.9199154335762216)
    (('to', 'has'), -1.9349620288103253)
    (('it', 'and'), -1.9561387802738253)
    (('and', 'but'), -1.982017412099431)
    (('but', 'and'), -1.982017412099431)
    (('before', 'to'), -1.983232950717003)
    (('4', 'the'), -2.021198769413406)
    (('the', '4'), -2.021198769413406)
    (('v.', 'the'), -2.0353926509961617)
    (('they', 'of'), -2.0484396766103288)
    (('to', 'can'), -2.0518200359744156)
    (('in', '('), -2.063063272096066)
    (('whether', ','), -2.074330156092312)
    (('to', 'v.'), -2.0824083558078357)
    (('the', '('), -2.100487679218041)
    (('but', '.'), -2.1059219891515433)
    (('it', 'of'), -2.109556421717958)
    (('he', 'of'), -2.11379341460443)
    (('of', '('), -2.1223049144599457)
    (('.', 'any'), -2.143889839350564)
    (('is', 'was'), -2.152553141461741)
    ((';', '.'), -2.1625055175179106)
    ((',', 'states'), -2.193629084464657)
    (('p.', 'the'), -2.1974494091051326)
    (('not', 'v.'), -2.2040488219407557)
    (('to', '('), -2.2063970730832914)
    (('in', 'not'), -2.2812954988767373)
    (('the', 'ill.'), -2.290558813496613)
    (('(', 'and'), -2.300093181491185)
    (('in', 'been'), -2.3144785946630257)
    (('it', 'v.'), -2.3337398985803937)
    (('where', ','), -2.3471035651263747)
    (('the', '1'), -2.3793470522135216)
    (('is', 'v.'), -2.3965450994043564)
    (('scam.', 'the'), -2.399083270274783)
    (('if', '.'), -2.401449115118208)
    (('to', '1'), -2.426362757025199)
    ((',', 'circuit'), -2.4737370036573942)
    (('his', 'of'), -2.5128893700142534)
    (('and', '('), -2.5631275873249777)
    (('v.', 'and'), -2.690677637045493)
    ((',', 'affirmed'), -2.6937026875992984)
    (('not', 'not'), -2.702388967914164)
    (('there', 'the'), -2.77336677022328)
    (('(', 'of'), -2.8227446326010366)
    (('1', 'of'), -2.872785315100632)
    (('(', '.'), -2.9094245857135412)
    (('v.', 'of'), -2.9762898908544955)
    (('in', 'be'), -3.030115679388718)
    (('of', ')'), -3.2493083852609104)




Let's clean the text a bit - remove numbers for example, perhaps this will reveal more patterns? 

__We'll get more into "preprocessing raw text" in our next lab__ - think of this as a simple introduction.


```python
!pip install gensim
```

    Collecting gensim
      Downloading gensim-4.4.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
    Requirement already satisfied: numpy>=1.18.5 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from gensim) (2.2.6)
    Requirement already satisfied: scipy>=1.7.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from gensim) (1.15.3)
    Collecting smart_open>=1.8.1 (from gensim)
      Downloading smart_open-7.5.0-py3-none-any.whl.metadata (24 kB)
    Requirement already satisfied: wrapt in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from smart_open>=1.8.1->gensim) (1.17.2)
    Downloading gensim-4.4.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (27.9 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.9/27.9 MB[0m [31m7.8 MB/s[0m  [33m0:00:03[0m[0m eta [36m0:00:01[0m0:01[0m:01[0m
    [?25hDownloading smart_open-7.5.0-py3-none-any.whl (63 kB)
    Installing collected packages: smart_open, gensim
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2/2[0m [gensim]━━━━[0m [32m1/2[0m [gensim]
    [1A[2KSuccessfully installed gensim-4.4.0 smart_open-7.5.0



```python
import scipy
from gensim.parsing.preprocessing import strip_numeric

case_corpus_tokenized = word_tokenize(strip_numeric(case_corpus)) ## we used the strip_numeric function from gensim
text = Text(case_corpus_tokenized)
```

**When you get import errors, look up the problem and see what is going on.**

For example, there could be a problem with `scipy` so try - conda install "scipy<1.13" which should solve this. 


```python
#case_corpus_tokenized
```


```python
bigram_finder = BigramCollocationFinder.from_words(case_corpus_tokenized, 
                                                   window_size = 10)

bigram_finder.apply_freq_filter(5) 
bigram_finder.nbest(bigram_measures.pmi, 30)
```




    [('nul', 'tiel'),
     ('crouch', 'hall'),
     ('dane', 'dig.'),
     ('laid', 'down'),
     ('brothers', 'sisters'),
     ('taylor', 'sprinkle'),
     ('george', 'lunceford'),
     ('ill.app.d', 'n.e.d'),
     ('jonathan', 'mayo'),
     ('forcible', 'detainer'),
     ('expire', 'end'),
     ('scire', 'facias'),
     ('goods', 'chattels'),
     ('st.', 'clair'),
     ('nil', 'debet'),
     ('thompson', 'armstrong'),
     ('ex', 'facto'),
     ('bryan', 'morrison'),
     ('heirs', 'assigns'),
     ('six', 'cent'),
     ('mason', 'wash'),
     ('affix', 'seal'),
     ('total', 'failure'),
     ('per', 'cent'),
     ('six', 'per'),
     ('scates', 'comp.'),
     ('st', 'october'),
     ('witnesses', 'devisee'),
     ('become', 'vacant'),
     ('t.', 'r.')]



Let's try it with **Trigrams**


```python
# Trigram
trigram_finder = TrigramCollocationFinder.from_words(case_corpus_tokenized,
                                                       window_size = 5)
```


```python
trigram_finder.apply_freq_filter(5) 
trigram_finder.nbest(trigram_measures.pmi, 30)
```




    [('six', 'per', 'cent'),
     ('six', 'cent', 'damages'),
     ('six', 'per', 'damages'),
     ('happen', 'during', 'recess'),
     ('nul', 'tiel', 'record'),
     ('ex', 'post', 'facto'),
     ('per', 'cent', 'damages'),
     ('dane', 's', 'dig.'),
     ('dane', '’', 'dig.'),
     ('r.', 'r.', 'co.'),
     ('st.', 'clair', 'county'),
     ('replevy', 'three', 'years'),
     ('per', 'cent', 'interest'),
     ('expire', 'at', 'end'),
     ('crouch', 'v.', 'hall'),
     ('plea', 'nil', 'debet'),
     ('taylor', 'v.', 'sprinkle'),
     ('scates', 'comp.', 'p.'),
     ('ex', 'facto', 'law'),
     ('statutes', 'p.', 'sec'),
     ('section', 'third', 'article'),
     ('scates', '’', 'comp.'),
     ('sec', 'scates', '’'),
     ('chief', 'wilson', '*'),
     ('purple', 's', 'statutes'),
     ('purple', '’', 'statutes'),
     ('thompson', 'v.', 'armstrong'),
     ('an', 'ex', 'facto'),
     ('with', 'per', 'cent'),
     ('register', 'land', 'office')]



Try doing this on **fourgrams** on your own


```python
# FourGram

```

# `Spacy`, `PhraseMachine`: other methods to better handle multi-word phrases

In my view, one of the better ways of addressing multi-word phrases are  the `Phrasemachine` and `Spacy` Libraries. 

The basic idea is that multi-word phrases tend to follow certain patterns based on their **part of speech tag** - for example, phrases like "old man", "semantic tree" have a pattern of **"adj-noun"**, etc. Thus, it's not enough to just look at words, but also, it's important to detect what part of speech they are. 

Thus, we must understand the firstly the idea of an [NLP task](https://nlpprogress.com/), and secondly a task of "Part of Speech tagging"  

### Part of Speech tagging.

Let's first start by exploring POS tagging


```python
sentence = "The quick brown fox jumps over the lazy dog."
```


```python
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger_eng')
```

    [nltk_data] Downloading package averaged_perceptron_tagger_eng to
    [nltk_data]     /home/leondgarse/nltk_data...
    [nltk_data]   Unzipping taggers/averaged_perceptron_tagger_eng.zip.





    True




```python
# Tokenize sentence into words (more on this in the following labs)
tokens = word_tokenize(sentence)

# Apply POS tagging
tags = pos_tag(tokens)
tags
```




    [('The', 'DT'),
     ('quick', 'JJ'),
     ('brown', 'NN'),
     ('fox', 'NN'),
     ('jumps', 'VBZ'),
     ('over', 'IN'),
     ('the', 'DT'),
     ('lazy', 'JJ'),
     ('dog', 'NN'),
     ('.', '.')]



If we want to see what the tags actually mean we can explore the original Penn Treebank POS tagset.


```python
# Download tagset descriptions 
nltk.download('tagsets_json')

# See all Penn Treebank tags
nltk.help.upenn_tagset()
```

    [nltk_data] Downloading package tagsets_json to
    [nltk_data]     /home/leondgarse/nltk_data...


    $: dollar
        $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
    '': closing quotation mark
        ' ''
    (: opening parenthesis
        ( [ {
    ): closing parenthesis
        ) ] }
    ,: comma
        ,
    --: dash
        --
    .: sentence terminator
        . ! ?
    :: colon or ellipsis
        : ; ...
    CC: conjunction, coordinating
        & 'n and both but either et for less minus neither nor or plus so
        therefore times v. versus vs. whether yet
    CD: numeral, cardinal
        mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
        seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
        fifteen 271,124 dozen quintillion DM2,000 ...
    DT: determiner
        all an another any both del each either every half la many much nary
        neither no some such that the them these this those
    EX: existential there
        there
    FW: foreign word
        gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
        lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
        terram fiche oui corporis ...
    IN: preposition or conjunction, subordinating
        astride among uppon whether out inside pro despite on by throughout
        below within for towards near behind atop around if like until below
        next into if beside ...
    JJ: adjective or numeral, ordinal
        third ill-mannered pre-war regrettable oiled calamitous first separable
        ectoplasmic battery-powered participatory fourth still-to-be-named
        multilingual multi-disciplinary ...
    JJR: adjective, comparative
        bleaker braver breezier briefer brighter brisker broader bumper busier
        calmer cheaper choosier cleaner clearer closer colder commoner costlier
        cozier creamier crunchier cuter ...
    JJS: adjective, superlative
        calmest cheapest choicest classiest cleanest clearest closest commonest
        corniest costliest crassest creepiest crudest cutest darkest deadliest
        dearest deepest densest dinkiest ...
    LS: list item marker
        A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
        SP-44007 Second Third Three Two * a b c d first five four one six three
        two
    MD: modal auxiliary
        can cannot could couldn't dare may might must need ought shall should
        shouldn't will would
    NN: noun, common, singular or mass
        common-carrier cabbage knuckle-duster Casino afghan shed thermostat
        investment slide humour falloff slick wind hyena override subhumanity
        machinist ...
    NNP: noun, proper, singular
        Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
        Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
        Shannon A.K.C. Meltex Liverpool ...
    NNPS: noun, proper, plural
        Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
        Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
        Apache Apaches Apocrypha ...
    NNS: noun, common, plural
        undergraduates scotches bric-a-brac products bodyguards facets coasts
        divestitures storehouses designs clubs fragrances averages
        subjectivists apprehensions muses factory-jobs ...
    PDT: pre-determiner
        all both half many quite such sure this
    POS: genitive marker
        ' 's
    PRP: pronoun, personal
        hers herself him himself hisself it itself me myself one oneself ours
        ourselves ownself self she thee theirs them themselves they thou thy us
    PRP$: pronoun, possessive
        her his mine my our ours their thy your
    RB: adverb
        occasionally unabatingly maddeningly adventurously professedly
        stirringly prominently technologically magisterially predominately
        swiftly fiscally pitilessly ...
    RBR: adverb, comparative
        further gloomier grander graver greater grimmer harder harsher
        healthier heavier higher however larger later leaner lengthier less-
        perfectly lesser lonelier longer louder lower more ...
    RBS: adverb, superlative
        best biggest bluntest earliest farthest first furthest hardest
        heartiest highest largest least less most nearest second tightest worst
    RP: particle
        aboard about across along apart around aside at away back before behind
        by crop down ever fast for forth from go high i.e. in into just later
        low more off on open out over per pie raising start teeth that through
        under unto up up-pp upon whole with you
    SYM: symbol
        % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
    TO: "to" as preposition or infinitive marker
        to
    UH: interjection
        Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
        huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
        man baby diddle hush sonuvabitch ...
    VB: verb, base form
        ask assemble assess assign assume atone attention avoid bake balkanize
        bank begin behold believe bend benefit bevel beware bless boil bomb
        boost brace break bring broil brush build ...
    VBD: verb, past tense
        dipped pleaded swiped regummed soaked tidied convened halted registered
        cushioned exacted snubbed strode aimed adopted belied figgered
        speculated wore appreciated contemplated ...
    VBG: verb, present participle or gerund
        telegraphing stirring focusing angering judging stalling lactating
        hankerin' alleging veering capping approaching traveling besieging
        encrypting interrupting erasing wincing ...
    VBN: verb, past participle
        multihulled dilapidated aerosolized chaired languished panelized used
        experimented flourished imitated reunifed factored condensed sheared
        unsettled primed dubbed desired ...
    VBP: verb, present tense, not 3rd person singular
        predominate wrap resort sue twist spill cure lengthen brush terminate
        appear tend stray glisten obtain comprise detest tease attract
        emphasize mold postpone sever return wag ...
    VBZ: verb, present tense, 3rd person singular
        bases reconstructs marks mixes displeases seals carps weaves snatches
        slumps stretches authorizes smolders pictures emerges stockpiles
        seduces fizzes uses bolsters slaps speaks pleads ...
    WDT: WH-determiner
        that what whatever which whichever
    WP: WH-pronoun
        that what whatever whatsoever which who whom whosoever
    WP$: WH-pronoun, possessive
        whose
    WRB: Wh-adverb
        how however whence whenever where whereby whereever wherein whereof why
    ``: opening quotation mark
        ` ``


    [nltk_data]   Unzipping help/tagsets_json.zip.


One of the problems with part of speech tagging is that it can somtimes be incorrect. 


```python
sentence = "Time flies like an arrow. Fruit flies like a banana."
```


```python
# Tokenize sentence into words
tokens = word_tokenize(sentence)

# Apply POS tagging
tags = pos_tag(tokens)
tags
```




    [('Time', 'NNP'),
     ('flies', 'NNS'),
     ('like', 'IN'),
     ('an', 'DT'),
     ('arrow', 'NN'),
     ('.', '.'),
     ('Fruit', 'NNP'),
     ('flies', 'VBZ'),
     ('like', 'IN'),
     ('a', 'DT'),
     ('banana', 'NN'),
     ('.', '.')]




```python
# let's put this in a dataframe, so you can see how to apply this in pandas
example = pd.DataFrame({
    "text": [
        "The quick brown fox jumps over the lazy dog.",
        "Time flies like an arrow. Fruit flies like a banana."
    ]
})

example
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The quick brown fox jumps over the lazy dog.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time flies like an arrow. Fruit flies like a b...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define a function to POS-tag each sentence
def pos_tag_text(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)

# Apply to DataFrame
example['pos_tags'] = example['text'].apply(pos_tag_text)
example
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
      <th>pos_tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The quick brown fox jumps over the lazy dog.</td>
      <td>[(The, DT), (quick, JJ), (brown, NN), (fox, NN...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time flies like an arrow. Fruit flies like a b...</td>
      <td>[(Time, NNP), (flies, NNS), (like, IN), (an, D...</td>
    </tr>
  </tbody>
</table>
</div>




```python
example['pos_tags'][1]
```




    [('Time', 'NNP'),
     ('flies', 'NNS'),
     ('like', 'IN'),
     ('an', 'DT'),
     ('arrow', 'NN'),
     ('.', '.'),
     ('Fruit', 'NNP'),
     ('flies', 'VBZ'),
     ('like', 'IN'),
     ('a', 'DT'),
     ('banana', 'NN'),
     ('.', '.')]



If we want to see what the tags actually mean we can explore the original Penn Treebank POS tagset.

### Better POS tagging with the `spacy` library

spaCy is a more modern, fast, and production-ready Natural Language Processing (NLP) library in Python.
It can do things like:

- Tokenization – splitting text into words or symbols

- POS Tagging – identifying grammatical roles (noun, verb, adjective…)

- Named Entity Recognition (NER) – finding names, locations, dates

- Dependency Parsing – understanding the grammatical structure

- Phrase Extraction – identifying meaningful chunks like “data analysis”

It is widely used because it is fast, accurate, and works well on large datasets.


```python
!pip install spacy
```

    Collecting spacy
      Downloading spacy-3.8.11-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (27 kB)
    Collecting spacy-legacy<3.1.0,>=3.0.11 (from spacy)
      Downloading spacy_legacy-3.0.12-py2.py3-none-any.whl.metadata (2.8 kB)
    Collecting spacy-loggers<2.0.0,>=1.0.0 (from spacy)
      Downloading spacy_loggers-1.0.5-py3-none-any.whl.metadata (23 kB)
    Collecting murmurhash<1.1.0,>=0.28.0 (from spacy)
      Downloading murmurhash-1.0.15-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (2.3 kB)
    Collecting cymem<2.1.0,>=2.0.2 (from spacy)
      Downloading cymem-2.0.13-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (9.7 kB)
    Collecting preshed<3.1.0,>=3.0.2 (from spacy)
      Downloading preshed-3.0.12-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (2.5 kB)
    Collecting thinc<8.4.0,>=8.3.4 (from spacy)
      Downloading thinc-8.3.10-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (15 kB)
    Collecting wasabi<1.2.0,>=0.9.1 (from spacy)
      Downloading wasabi-1.1.3-py3-none-any.whl.metadata (28 kB)
    Collecting srsly<3.0.0,>=2.4.3 (from spacy)
      Downloading srsly-2.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
    Collecting catalogue<2.1.0,>=2.0.6 (from spacy)
      Downloading catalogue-2.0.10-py3-none-any.whl.metadata (14 kB)
    Collecting weasel<0.5.0,>=0.4.2 (from spacy)
      Downloading weasel-0.4.3-py3-none-any.whl.metadata (4.6 kB)
    Collecting typer-slim<1.0.0,>=0.3.0 (from spacy)
      Downloading typer_slim-0.21.1-py3-none-any.whl.metadata (16 kB)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from spacy) (4.67.1)
    Requirement already satisfied: numpy>=1.19.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from spacy) (2.2.6)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from spacy) (2.32.5)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from spacy) (2.12.4)
    Requirement already satisfied: jinja2 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from spacy) (3.1.6)
    Requirement already satisfied: setuptools in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from spacy) (80.9.0)
    Requirement already satisfied: packaging>=20.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from spacy) (25.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (0.7.0)
    Requirement already satisfied: pydantic-core==2.41.5 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (2.41.5)
    Requirement already satisfied: typing-extensions>=4.14.1 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (4.15.0)
    Requirement already satisfied: typing-inspection>=0.4.2 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (0.4.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from requests<3.0.0,>=2.13.0->spacy) (2025.10.5)
    Collecting blis<1.4.0,>=1.3.0 (from thinc<8.4.0,>=8.3.4->spacy)
      Downloading blis-1.3.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.5 kB)
    Collecting confection<1.0.0,>=0.0.1 (from thinc<8.4.0,>=8.3.4->spacy)
      Downloading confection-0.1.5-py3-none-any.whl.metadata (19 kB)
    Requirement already satisfied: click>=8.0.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from typer-slim<1.0.0,>=0.3.0->spacy) (8.3.0)
    Collecting cloudpathlib<1.0.0,>=0.7.0 (from weasel<0.5.0,>=0.4.2->spacy)
      Downloading cloudpathlib-0.23.0-py3-none-any.whl.metadata (16 kB)
    Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from weasel<0.5.0,>=0.4.2->spacy) (7.5.0)
    Requirement already satisfied: wrapt in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.4.2->spacy) (1.17.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from jinja2->spacy) (3.0.3)
    Downloading spacy-3.8.11-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (33.2 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m33.2/33.2 MB[0m [31m7.8 MB/s[0m  [33m0:00:04[0m[0m eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hDownloading catalogue-2.0.10-py3-none-any.whl (17 kB)
    Downloading cymem-2.0.13-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (260 kB)
    Downloading murmurhash-1.0.15-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (134 kB)
    Downloading preshed-3.0.12-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (874 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m875.0/875.0 kB[0m [31m5.9 MB/s[0m  [33m0:00:00[0m
    [?25hDownloading spacy_legacy-3.0.12-py2.py3-none-any.whl (29 kB)
    Downloading spacy_loggers-1.0.5-py3-none-any.whl (22 kB)
    Downloading srsly-2.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m6.9 MB/s[0m  [33m0:00:00[0m
    [?25hDownloading thinc-8.3.10-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.9 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.9/3.9 MB[0m [31m7.5 MB/s[0m  [33m0:00:00[0mm [31m8.5 MB/s[0m eta [36m0:00:01[0m
    [?25hDownloading blis-1.3.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (11.4 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.4/11.4 MB[0m [31m7.8 MB/s[0m  [33m0:00:01[0m[0m eta [36m0:00:01[0m0:01[0m01[0m
    [?25hDownloading confection-0.1.5-py3-none-any.whl (35 kB)
    Downloading typer_slim-0.21.1-py3-none-any.whl (47 kB)
    Downloading wasabi-1.1.3-py3-none-any.whl (27 kB)
    Downloading weasel-0.4.3-py3-none-any.whl (50 kB)
    Downloading cloudpathlib-0.23.0-py3-none-any.whl (62 kB)
    Installing collected packages: wasabi, typer-slim, spacy-loggers, spacy-legacy, murmurhash, cymem, cloudpathlib, catalogue, blis, srsly, preshed, confection, weasel, thinc, spacy
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15/15[0m [spacy]m━━[0m [32m14/15[0m [spacy][thinc]d]
    [1A[2KSuccessfully installed blis-1.3.3 catalogue-2.0.10 cloudpathlib-0.23.0 confection-0.1.5 cymem-2.0.13 murmurhash-1.0.15 preshed-3.0.12 spacy-3.8.11 spacy-legacy-3.0.12 spacy-loggers-1.0.5 srsly-2.5.2 thinc-8.3.10 typer-slim-0.21.1 wasabi-1.1.3 weasel-0.4.3



```python
!python -m spacy download en_core_web_sm
```

    Collecting en-core-web-sm==3.8.0
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl (12.8 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.8/12.8 MB[0m [31m7.4 MB/s[0m  [33m0:00:01[0m7.8 MB/s[0m eta [36m0:00:01[0m:01[0m
    [?25hInstalling collected packages: en-core-web-sm
    Successfully installed en-core-web-sm-3.8.0
    [38;5;2m✔ Download and installation successful[0m
    You can now load the package via spacy.load('en_core_web_sm')



```python
import spacy

# Load small English language model
nlp = spacy.load("en_core_web_sm") 
```


```python
# simple example
doc = nlp("The quick brown fox jumps over the lazy dog.") # use the nlp() function 
for token in doc:
    print(token.text, token.pos_)
```

    The DET
    quick ADJ
    brown ADJ
    fox NOUN
    jumps VERB
    over ADP
    the DET
    lazy ADJ
    dog NOUN
    . PUNCT



```python
def pos_tag_text(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]  # coarse-grained POS
```


```python
# Function to POS tag using spaCy
def pos_tag_text(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]  # coarse-grained POS
```


```python
example['pos_tags'] = example['text'].apply(pos_tag_text)
example
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
      <th>pos_tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The quick brown fox jumps over the lazy dog.</td>
      <td>[(The, DET), (quick, ADJ), (brown, ADJ), (fox,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time flies like an arrow. Fruit flies like a b...</td>
      <td>[(Time, NOUN), (flies, VERB), (like, ADP), (an...</td>
    </tr>
  </tbody>
</table>
</div>




```python
example['pos_tags'][1] # even spacy gets it wrong! 
```




    [('Time', 'NOUN'),
     ('flies', 'VERB'),
     ('like', 'ADP'),
     ('an', 'DET'),
     ('arrow', 'NOUN'),
     ('.', 'PUNCT'),
     ('Fruit', 'NOUN'),
     ('flies', 'VERB'),
     ('like', 'ADP'),
     ('a', 'DET'),
     ('banana', 'NOUN'),
     ('.', 'PUNCT')]



For more details see the amazing [Spacy website](https://spacy.io/usage/linguistic-features) which contains countless examples and code.

Understanding POS tagging is essential because phrase extraction tools like PhraseMachine rely on POS patterns (like “adjective + noun” or “noun + noun”) to identify meaningful phrases.


```python
def extract_phrases(text):
    doc = nlp(text)
    phrases = []
    temp = []

    for token in doc:
        if token.pos_ in ["ADJ", "NOUN", "PROPN"]: # If the token is an adjective (ADJ), common noun (NOUN), or proper noun (PROPN) it's likely a phrase
            temp.append(token.text)                # add to "temp"
        else:                                      # if NOT adj/noun/proper noun, then join the strings - ie  
            if len(temp) > 1:                      # if length of temp is > 1 - which removes single word phrases 
                phrases.append(" ".join(temp))     # add to "phrases" list 
            temp = []                              # reset temp for next phrase

    if len(temp) > 1:                              # if temp is non-empty, add final phrase to "phrases" list and conclude the loop  
        phrases.append(" ".join(temp))

    return phrases
```


```python
df = pd.DataFrame({
    "text": [
        "The quick brown fox jumps over the lazy dog.",
        "Time flies like an arrow. Fruit flies like a banana.",
        "He is a working at the National University of Singapore."
    ]
})
```


```python
df['phrases'] = df['text'].apply(extract_phrases)
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
      <th>phrases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The quick brown fox jumps over the lazy dog.</td>
      <td>[quick brown fox, lazy dog]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time flies like an arrow. Fruit flies like a b...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>He is a working at the National University of ...</td>
      <td>[National University]</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the tagging pattern is imperfect because the underlying POS tag didn't catch "fruit flies". But depending on the pattern that you define (adj-adj-noun), it could be beneficial.



### Phrasemachine

Phrasemachine is based on a paper titled ["Technical terminology: some linguistic properties and an algorithm for identification in text" by Juteson and Katz (1995)](https://brenocon.com/JustesonKatz1995.pdf) 

This paper proposes an algorithmic solution to finding multi-word phrases using lexical "Part of Speech" patterns. 

Part of speech tagging is an **NLP task** where we seek to find information concerning, for example, whether a word is a noun or an adjective or an adverb, etc.

### Installing phrasemachine - and libraries that are not on Anaconda Navigator

Anaconda Navigator shows only packages that are available in Anaconda’s own repositories (channels like defaults or conda-forge). These packages are pre-built, tested, and guaranteed to work with the rest of the Anaconda ecosystem. For example, libraries like `pandas`, `numpy`, `scikit-learn`

Libraries like `phrasemachine` (or very new/less popular ones) are not included in Anaconda’s default channels, and thus have to be install  manually using `pip` or `conda` via the terminal. If something is not in `conda`


```python
!pip install phrasemachine
```

    Collecting phrasemachine
      Downloading phrasemachine-1.0.7.tar.gz (2.7 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.7/2.7 MB[0m [31m4.6 MB/s[0m  [33m0:00:00[0mm [31m4.8 MB/s[0m eta [36m0:00:01[0m
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: nltk in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from phrasemachine) (3.9.1)
    Requirement already satisfied: click in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from nltk->phrasemachine) (8.3.0)
    Requirement already satisfied: joblib in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from nltk->phrasemachine) (1.3.2)
    Requirement already satisfied: regex>=2021.8.3 in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from nltk->phrasemachine) (2024.11.6)
    Requirement already satisfied: tqdm in /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages (from nltk->phrasemachine) (4.67.1)
    Building wheels for collected packages: phrasemachine
      Building wheel for phrasemachine (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for phrasemachine: filename=phrasemachine-1.0.7-py3-none-any.whl size=2694894 sha256=9aa576d2b421e2b45ecbf0d67d1c2fd985f3b44520831392279a638939d04b6c
      Stored in directory: /home/leondgarse/.cache/pip/wheels/8c/d7/03/06fc0f1ecaeda7a4ea7f80fb17c1cdf35c88b1c987f1b03d02
    Successfully built phrasemachine
    Installing collected packages: phrasemachine
    Successfully installed phrasemachine-1.0.7



```python
import phrasemachine
```

    /home/leondgarse/virtualenvs/workon312/lib/python3.12/site-packages/phrasemachine/phrasemachine.py:6: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      from pkg_resources import resource_filename


Let's try an example on a string


```python
text = """
Data science requires a combination of statistical learning, 
machine learning algorithms, and domain expertise. High-performance 
computing power is often necessary for deep neural networks.
"""

# Process the text
doc = nlp(text)
tokens = [token.text for token in doc]
pos_tags = [token.tag_ for token in doc]

# Extract phrases
phrases = phrasemachine.get_phrases(tokens=tokens, 
                                    postags=pos_tags)

# Print the counts of extracted phrases
phrases['counts']
```




    Counter({'data science': 1,
             'combination of statistical learning': 1,
             'statistical learning': 1,
             'machine learning': 1,
             'domain expertise': 1,
             'computing power': 1,
             'deep neural networks': 1,
             'neural networks': 1})



Let's try an example on a pandas dataframe


```python
def extract_phrases(text):
    doc = nlp(text)
    tokens = [t.text for t in doc]
    postags = [t.tag_ for t in doc]
    
    # get_phrases returns a dict; we just want the list of phrases
    results = phrasemachine.get_phrases(tokens=tokens, postags=postags)
    return list(results['counts'].keys())
```


```python
# Apply the function to the 'text' column
df['phrases'] = df['text'].apply(extract_phrases)
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
      <th>text</th>
      <th>phrases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The quick brown fox jumps over the lazy dog.</td>
      <td>[quick brown fox, brown fox, lazy dog]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time flies like an arrow. Fruit flies like a b...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>He is a working at the National University of ...</td>
      <td>[working at the national, working at the natio...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['phrases'][2]
```




    ['working at the national',
     'working at the national university',
     'working at the national university of singapore',
     'national university',
     'national university of singapore',
     'university of singapore']



Because these methods rely on POS tags, the quality of the phrases is directly tied to the accuracy of your tagger (e.g., SpaCy POS tagger). If the tagger misidentifies a verb as a noun, PhraseMachine will likely pick it up.

Let's go back to our cases dataset


```python
df = df_cleaned[:100]
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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Apply the function to the 'text' column
df['phrases'] = df['text'].apply(extract_phrases)
```

    /tmp/ipykernel_29263/4267135625.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['phrases'] = df['text'].apply(extract_phrases)



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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
      <th>phrases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
      <td>[mr. presiding, mr. presiding justice, mr. pre...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
      <td>[opinion of the court, criminal prosecution, c...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>[opinion of the court, action of debt, action ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>[opinion of the court, action of covenant, fif...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
      <td>[opinion of the court, record in this cause, m...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['phrases'][0]
```




    ['mr. presiding',
     'mr. presiding justice',
     'mr. presiding justice eberspacher',
     'presiding justice',
     'presiding justice eberspacher',
     'justice eberspacher',
     'opinion of the court',
     'defendant tobin',
     'jury of the crime',
     'jury of the crime of burglary',
     'crime of burglary',
     'judgment upon the verdict',
     'five year',
     'five year term',
     'five year term in the illinois',
     'five year term in the illinois state',
     'five year term in the illinois state penitentiary',
     'year term',
     'year term in the illinois',
     'year term in the illinois state',
     'year term in the illinois state penitentiary',
     'term in the illinois',
     'term in the illinois state',
     'term in the illinois state penitentiary',
     'illinois state',
     'illinois state penitentiary',
     'state penitentiary',
     'judgment of the court',
     'release by federal authorities',
     'federal authorities',
     'following issues',
     'lack of authority',
     'arrest of sherri',
     'arrest of sherri tobin',
     'sherri tobin',
     'possession of a firearm',
     'rise to this case',
     'night of february',
     '11:00 p.m.',
     'company of sherri',
     'company of sherri tobin',
     'daniel stout',
     'michael hume',
     'eddie dunn',
     'defendant in the vicinity',
     'defendant in the vicinity of the oliver',
     'defendant in the vicinity of the oliver c.',
     'vicinity of the oliver',
     'vicinity of the oliver c.',
     'vicinity of the oliver c. joseph',
     'vicinity of the oliver c. joseph automobile',
     'vicinity of the oliver c. joseph automobile agency',
     'oliver c.',
     'oliver c. joseph',
     'oliver c. joseph automobile',
     'oliver c. joseph automobile agency',
     'oliver c. joseph automobile agency in belleville',
     'c. joseph',
     'c. joseph automobile',
     'c. joseph automobile agency',
     'c. joseph automobile agency in belleville',
     'joseph automobile',
     'joseph automobile agency',
     'joseph automobile agency in belleville',
     'automobile agency',
     'automobile agency in belleville',
     'agency in belleville',
     'james muir',
     'agency door',
     'car in the meantime',
     'four men',
     'street to the agency',
     'agency building',
     'agency building by the same door',
     'building by the same door',
     'same door',
     'officers rettle',
     'two other policemen',
     'other policemen',
     'suspects at the back door',
     'back door',
     'one suspect',
     'other suspects',
     'air compressor',
     'police with mr.',
     'police with mr. muir',
     'mr. muir',
     'few blocks',
     '.38 caliber',
     '.38 caliber snub',
     'caliber snub',
     'nosed revolver',
     'serial numbers',
     'waist band',
     'gun into the building',
     'door jamb',
     'door jamb of the door',
     'jamb of the door',
     'mr. oliver',
     'mr. oliver p.',
     'mr. oliver p. joseph',
     'oliver p.',
     'oliver p. joseph',
     'p. joseph',
     'possession of oliver',
     'possession of oliver c.',
     'possession of oliver c. joseph',
     'selling of automobiles',
     'presence in a public building',
     'presence in a public building for a purpose',
     'public building',
     'public building for a purpose',
     'public building for a purpose inconsistent',
     'building for a purpose',
     'building for a purpose inconsistent',
     'building for a purpose inconsistent with the purposes',
     'purpose inconsistent',
     'purpose inconsistent with the purposes',
     'inconsistent with the purposes',
     'people v. urban',
     'ill. app.2d',
     '266 n.e.2d',
     'people v. weaver',
     '41 ill.2d',
     '243 n.e.2d',
     '243 n.e.2d 245 cert',
     'n.e.2d 245 cert',
     '245 cert',
     '395 u.s.',
     '89 s.ct',
     '89 s.ct .',
     's.ct .',
     '23 l.ed.2d',
     'basement upon arrival',
     'basement upon arrival of the police',
     'arrival of the police',
     'sufficient evidence',
     'sufficient evidence for the jury',
     'evidence for the jury',
     'regard to the question',
     'regard to the question of intent',
     'question of intent',
     'sufficient circumstantial evidence',
     'sufficient circumstantial evidence for the jury',
     'circumstantial evidence',
     'circumstantial evidence for the jury',
     'theft in the building',
     'factual environment',
     'people v. johnson',
     '28 ill.2d',
     '192 n.e.2d',
     'court in johnson',
     'absence of inconsistent circumstances',
     'inconsistent circumstances',
     'proof of unlawful breaking',
     'unlawful breaking',
     'entry into a building',
     'personal property',
     'subject of larceny',
     'rise to an inference',
     'conviction of burglary',
     'other inferences',
     'human experience',
     'assumption that the unlawful entry',
     'unlawful entry',
     'absence of other proof',
     'other proof',
     'likely purpose',
     'circumstances of the entry',
     'sufficient evidence of intent',
     'evidence of intent',
     'third aueged error',
     'aueged error',
     'question of firearms',
     'evidence of sherri',
     'evidence of the .38 cafibre',
     'evidence of the .38 cafibre gun',
     '.38 cafibre',
     '.38 cafibre gun',
     'cafibre gun',
     'regard to tobin',
     'only evidence',
     'conviction of tobin',
     'finder of fact',
     'finder of fact of the total circumstances',
     'fact of the total circumstances',
     'total circumstances',
     'harmless error',
     'careful search',
     'careful search of the record',
     'search of the record',
     'such evidence',
     'element of the crime',
     'people v. landgham',
     '122 ill.',
     '122 ill. app.2d',
     '275 n.e.2d',
     'people v. jones',
     '125 ill.',
     '125 ill. app.2d',
     '259 n.e.2d',
     'heavier sentence',
     'co -',
     'co - defendants',
     '- defendants',
     'same offense',
     'probation for a period',
     'probation for a period of five years',
     'period of five years',
     'five years',
     'ten years',
     'constitutional right',
     'constitutional right to a trial',
     'constitutional right to a trial by jury',
     'right to a trial',
     'right to a trial by jury',
     'trial by jury',
     'basic principles',
     '118 ill.',
     '118 ill. app.2d',
     '254 n.e.2d',
     'offense in a like category',
     'like category',
     'identical punishment',
     'proper variation',
     'proper variation in sentences',
     'variation in sentences',
     'different offenders',
     'circumstances of the individual case',
     'individual case',
     'general rule',
     'punishment for the offense',
     'statutory regulation',
     'appeuate court',
     'appeuate court wiu',
     'appeuate court wiu interfere',
     'court wiu',
     'court wiu interfere',
     'wiu interfere',
     'reasonable view',
     'people v. hobbs',
     '58 ill.',
     '58 ill. app.2d',
     '205 n.e.2d',
     'disparity of sentences',
     'disparity of sentences between defendants',
     'sentences between defendants',
     'use of the power',
     'trial court',
     'people v. thompson',
     '36 ill.2d',
     '224 n.e.2d',
     'prior criminal record',
     'criminal record',
     'records of the other defendants',
     'other defendants',
     'record on this appeal',
     'prior burglary',
     'prior burglary conviction',
     'prior burglary conviction as a juvenile',
     'burglary conviction',
     'burglary conviction as a juvenile',
     'conviction as a juvenile',
     'pretrial bail',
     'presence at trial',
     'virtue of a writ',
     'virtue of a writ of habeas',
     'virtue of a writ of habeas corpus',
     'virtue of a writ of habeas corpus ad',
     'writ of habeas',
     'writ of habeas corpus',
     'writ of habeas corpus ad',
     'writ of habeas corpus ad prosequendum',
     'habeas corpus',
     'habeas corpus ad',
     'habeas corpus ad prosequendum',
     'corpus ad',
     'corpus ad prosequendum',
     'ad prosequendum',
     'apparent ringleader',
     'apparent ringleader of the burglary',
     'ringleader of the burglary',
     'factual situation',
     'possibility of rehabilitation',
     '25 years',
     'practical matter',
     'room for rehabilitation',
     'such sentence',
     'exercise of the discretion',
     'exercise of the discretion of parole',
     'exercise of the discretion of parole authorities',
     'discretion of parole',
     'discretion of parole authorities',
     'discretion of parole authorities at a time',
     'parole authorities',
     'parole authorities at a time',
     'authorities at a time',
     'such discretion',
     'minimum of seven years',
     'seven years',
     'maximum of 20 years',
     '20 years',
     'possible future federal sentence',
     'future federal sentence',
     'federal sentence',
     'ill. rev.',
     'ill. rev. stat',
     'rev. stat',
     'ch .',
     'attention to the fact',
     'attention to the fact that ch',
     'attention to the fact that ch .',
     'fact that ch',
     'fact that ch .',
     'par .',
     'concurrent sentence',
     'unexpired sentence',
     'federal district',
     'federal district court',
     'district court',
     'oral argument',
     'oral argument in june',
     'argument in june',
     'involvement with federal authorities',
     'united states',
     'united states v. tobin',
     'states v. tobin',
     'new trial',
     'new trial in may',
     'trial in may',
     'construction by the court',
     'ministerial officer',
     'people v. walton',
     'judgment of conviction',
     'minimum sentence',
     'minimum sentence of seven years',
     'sentence of seven years',
     'maximum sentence',
     'maximum sentence of twenty years',
     'sentence of twenty years',
     'twenty years',
     'judgment with sentence']



We seem to catch a lot of generic phrases. Let's try  Frequency Filtering - ie, we don't really care about phrases that appear only once. 


```python
# Flatten the list of phrases into a single series
all_phrases = df['phrases'].explode()

# Calculate counts for every unique phrase
phrase_counts = all_phrases.value_counts()

phrase_counts.head(20)
```




    phrases
    opinion of the court      95
    et al                     43
    et al .                   43
    al .                      43
    circuit court             43
    judgment of the court     37
    justice lockwood          28
    al . v.                   28
    . v.                      28
    et al . v.                28
    chief justice             27
    1 scam                    25
    common law                19
    justice reynolds          19
    3 scam                    19
    justice smith             18
    supreme court             18
    sec .                     17
    chief justice reynolds    17
    4 scam                    15
    Name: count, dtype: int64




```python
# Define our minimum phrase frequency threshold
min_freq = 2 # at least two times 

# Get the list of phrases that meet the criteria
frequent_phrases = phrase_counts[phrase_counts >= min_freq].index.tolist()
frequent_phrase_set = set(frequent_phrases) # Using a set for faster lookups

# Filter the phrases column in the original DataFrame
df['filtered_phrases'] = df['phrases'].apply(
    lambda x: [p for p in x if p in frequent_phrase_set]
)
```

    /tmp/ipykernel_29263/3257970016.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['filtered_phrases'] = df['phrases'].apply(



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
      <th>decision_date</th>
      <th>name_abbreviation</th>
      <th>text</th>
      <th>year</th>
      <th>phrases</th>
      <th>filtered_phrases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1771-10-12</td>
      <td>People v. Tobin</td>
      <td>Mr. PRESIDING JUSTICE EBERSPACHER\ndelivered t...</td>
      <td>1771</td>
      <td>[mr. presiding, mr. presiding justice, mr. pre...</td>
      <td>[opinion of the court, judgment of the court, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1819-12-01</td>
      <td>Whitesides v. People</td>
      <td>Opinion of the Court. This was a criminal pros...</td>
      <td>1819</td>
      <td>[opinion of the court, criminal prosecution, c...</td>
      <td>[opinion of the court, plaintiffs in error, cr...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1819-12-01</td>
      <td>Chipps v. Yancey</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>[opinion of the court, action of debt, action ...</td>
      <td>[opinion of the court, action of debt, state o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1819-12-01</td>
      <td>Taylor v. Sprinkle</td>
      <td>Opinion of the Court.\n*\nThis was an action o...</td>
      <td>1819</td>
      <td>[opinion of the court, action of covenant, fif...</td>
      <td>[opinion of the court, action of covenant, fif...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1819-12-01</td>
      <td>Coleen v. Figgins</td>
      <td>Opinion of the Court.\n†\nIt appears from the ...</td>
      <td>1819</td>
      <td>[opinion of the court, record in this cause, m...</td>
      <td>[opinion of the court, madison circuit, madiso...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['filtered_phrases'][0]
```




    ['opinion of the court',
     'judgment of the court',
     'sufficient evidence',
     'personal property',
     'co -',
     'five years',
     'constitutional right',
     'trial by jury',
     'general rule',
     'ch .',
     'united states',
     'new trial']




```python

```
