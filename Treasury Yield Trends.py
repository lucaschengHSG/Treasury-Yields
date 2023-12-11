### Treasury Yield Trends ###

import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re

# Note 1st time installation of non-standard libraries
# pip install ntscraper
from ntscraper import Nitter

# pip install textblob

import textblob
from textblob import TextBlob
from bs4 import BeautifulSoup as soup
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


## Data Collection
scraper = Nitter()

def get_tweets(name, modes, no):
    tweets = scraper.get_tweets(name,mode = modes, number =no)
    final_tweets = []
    for tweet in tweets['tweets']:
        data = tweet['text']
        final_tweets.append(data)
    data = pd.DataFrame(final_tweets, columns =['text'])
    return data

data = get_tweets(['treasury' or 'yields'],'hashtag', 10000)

# Check directory before extracting csv
data.to_csv("TYields.csv")

# View scarpped data
tyields_data = pd.read_csv("TYields.csv")

tyields_data = tyields_data.iloc[:,1:]

## Data Processing
tyields_data.head()

# Check for missing data
tyields_data.isna().sum()
tyields_data.describe()
tyields_data.info()

## Exploratory Analysis
# Removing hyperlinks, special characters from raw tweets
def clean_tweet(tweet): 
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split())

# Categorize sentiment based on polarity
# using the textblob method
def analyse_tweet(tweet):
    cleaned_tweets = clean_tweet(tweet) 
    tweet_sentiment = TextBlob(tweet).sentiment.polarity
    return tweet_sentiment

def get_tweet_sentiment(tweet): 
    tweet_polarity = analyse_tweet(tweet) 
    if tweet_polarity >0:
        return 'positive'
    elif tweet_polarity == 0:
        return 'neutral'
    else:
        return 'negative'

tyields_data['Sentiment'] = tyields_data['text'].apply(lambda x: get_tweet_sentiment(x))
tyields_data['Sentiment'].value_counts().plot(kind='bar')

## Word cloud visualization
def wordcloud(string):
    wc = WordCloud(background_color=color, width=1200,height=600,mask=None,random_state=1,
                   max_font_size=200,stopwords=stop_words,collocations=False).generate(string)
    fig=plt.figure(figsize=(20,8))
    plt.axis('off')
    plt.title('--- WordCloud for {} --- '.format(title),weight='bold', size=30)
    plt.imshow(wc)

tyields_data['Text'] = tyields_data['text'].apply(lambda x: clean_tweet(x))

stop_words=set(STOPWORDS)
tyields_data_string = " ".join(tyields_data['Text'].astype('str'))

# create the wordcloud
tweet_string  = " ".join(tweet for tweet in tyields_data["Text"])
from wordcloud import WordCloud
tweet_wordcloud = WordCloud(background_color="white", max_words=50).generate(tweet_string)

# Word cloud in Plots
import matplotlib.pyplot as plt   # for wordclouds & charts
plt.imshow(tweet_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()




