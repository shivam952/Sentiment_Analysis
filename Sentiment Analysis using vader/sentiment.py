# we take the help of VADER( Valence aware Dictionaty for sentiment Reasoning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('amazonreviews.tsv', delimiter='\t',quoting=3)
dataset.dropna(inplace=True) #remove null columns

blanks =[]
for i,lb,rv in dataset.itertuples():
    if(type(rv)==str):
        if(rv.isspace()):
            blanks.append(i)
            
dataset.drop(blanks,inplace=True) #remove blanks

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

dataset['scores'] = dataset['review'].apply(lambda review : sid.polarity_scores(review))
dataset['compound']  = dataset['scores'].apply(lambda score_dict: score_dict['compound'])

dataset['comp_score'] = dataset['compound'].apply(lambda c : 'pos' if c >=0 else 'neg')

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy = accuracy_score(dataset['label'], dataset['comp_score'])
              
            
            