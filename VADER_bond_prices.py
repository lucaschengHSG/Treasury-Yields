
### Treasury Yield and Bond Trends ###

# Import relevant libraries

import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates

# Note 1st time installation of non-standard libraries
# pip install ntscraper
from ntscraper import Nitter

# pip install textblob
from datetime import datetime
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

import os
#pip install nltk
import nltk



## Data Collection
scraper = Nitter()

def get_tweets(name, modes, no, language):
    tweets = scraper.get_tweets(name,mode = modes, number =no, since=since_date, until=end_date)
    final_tweets = []
    for tweet in tweets['tweets']:
        data = {'text': tweet['text'], 'created_at': tweet['date']}
        final_tweets.append(data)
    data = pd.DataFrame(final_tweets, columns =['text', 'created_at'])
    return data

since_date = '2023-09-01'
end_date = '2023-12-12'

data = get_tweets(['bond prices'],'term', 10000, language = 'en')

# Remove irrelevant and often present bitcoin data
data = data[~data['text'].str.contains('bitcoin|btc', case=False, na=False)]


# To avoid potential scraping errors, filter the DataFrame to only include 
# data for dates from September 1st onwards
data = data[data['created_at'] >= '2023-09-01']



# Check directory before extracting csv
data.to_csv("Bond_Prices.csv")

# View scraped data
tyields_data = pd.read_csv("Bond_Prices.csv")

tyields_data = tyields_data.iloc[:,1:]

tyields_data = tyields_data[tyields_data['created_at'] >= '2023-09-01']

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



# Import and initialize the VADER sentiment intensity analyzer
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# Optional: Neutralize specific words biasing the analysis
neutralized_words = ['yes","beneficial","useful","extraordinary","exciting',
                     'happy','very','inverse','wealth','fascinating','agree',
                     'love','like', 'great','awesome','positive','high']

for word in neutralized_words:
        sia.lexicon[word] = 0.
        
# Add or adapt specific lexicon values if needed:

sia.lexicon["success"] = 1.5
sia.lexicon["nuking"] = -5
sia.lexicon['inflation'] = -3
sia.lexicon['dip'] = -2
sia.lexicon['decline'] = -1
sia.lexicon['squeeze'] = -2
sia.lexicon['spike'] = -3
sia.lexicon['dump'] = -4
sia.lexicon['rise'] = -0.5
sia.lexicon["bear"] = -4
sia.lexicon["bearish"] = -4
sia.lexicon['tank'] = -3
sia.lexicon['selloff']= -4
sia.lexicon['tanking'] = -3
sia.lexicon['bubble'] = -3


# Use VADER for sentiment analysis

def analyse_tweet_with_vader(tweet):
    # Clean the tweet
    cleaned_tweet = clean_tweet(tweet)
    
    # Obtain polarity scores for the cleaned tweet
    sentiment_scores = sia.polarity_scores(cleaned_tweet)
    
    return sentiment_scores['compound']

# Extract the sentiment compound score 

def get_tweet_sentiment(tweet): 
    # Using VADER for sentiment analysis instead of TextBlob
    tweet_compound_score = analyse_tweet_with_vader(tweet) 
    return tweet_compound_score


# Apply the function to calculate tweet sentiment
tyields_data['Sentiment'] = tyields_data['text'].apply(lambda x: get_tweet_sentiment(x))

##################################

## Optional: Word cloud 
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial'
def wordcloud(string):
    wc = WordCloud(background_color=color, width=1200,height=600,mask=None,random_state=1,
                   max_font_size=200,stopwords=stop_words,collocations=False).generate(string)
    fig=plt.figure(figsize=(20,8))
    plt.axis('off')
    plt.title('--- WordCloud for {} --- '.format(title),weight='bold', size=30)
    plt.imshow(wc)
    plt.savefig("Wordcloud.png", bbox_inches='tight')
tyields_data['Text'] = tyields_data['text'].apply(lambda x: clean_tweet(x))

stop_words=set(STOPWORDS)
tyields_data_string = " ".join(tyields_data['Text'].astype('str'))

# Create the wordcloud
tweet_string  = " ".join(tweet for tweet in tyields_data["Text"])
from wordcloud import WordCloud
tweet_wordcloud = WordCloud(background_color="white", max_words=50).generate(tweet_string)

# Word cloud in Plots
plt.imshow(tweet_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


###########################

# Creating a scatter plot of sentiment observations

date_format = "%b %d, %Y · %I:%M %p UTC"
tyields_data['created_at'] = tyields_data['created_at'].apply(lambda x: datetime.strptime(x, date_format))

tyields_data = tyields_data[tyields_data['created_at'] >= '2023-09-01']
# Sort the DataFrame by date in ascending order
tyields_data = tyields_data.sort_values(by='created_at')

# The scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(tyields_data['created_at'], tyields_data['Sentiment'], alpha=0.6, color='blue')

# Format the dates on the x-axis
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

# Add labels and a title to the plot
plt.xlabel('Date')
plt.ylabel('Sentiment')
plt.title('Sentiment Analysis Over Time')

# Show the plot
plt.show()


##########################

# Perform K-means clustering to obtain sentiment clusters

from sklearn.cluster import KMeans

# Reshape sentiment values for k-means which expects a 2D array
X = tyields_data['Sentiment'].values.reshape(-1, 1)

# Define the k-means clustering algorithm with 3 clusters for positive, neutral, and negative sentiments
initial_centroids = np.array([[0.75], [0], [-0.5]])
kmeans = KMeans(n_clusters=3, init=initial_centroids, n_init=1, random_state=0)

# Fit the model to the data
kmeans.fit(X)

# Predict the cluster index for each sentiment value
tyields_data['Cluster'] = kmeans.predict(X)


# Create a scatterplot
plt.figure(figsize=(10, 6))

# Define cluster names based on the centroids
# Sort the clusters to ensure 0: Negative, 1: Neutral, 2: Positive
sorted_idx = np.argsort(kmeans.cluster_centers_.reshape(-1))
cluster_names = ['Negative', 'Neutral', 'Positive']
colors = ['red', 'grey', 'green']  # Colors corresponding to Negative, Neutral, Positive

for idx, cluster in enumerate(sorted_idx):
    clustered_data = tyields_data[tyields_data['Cluster'] == cluster]
    plt.scatter(clustered_data['created_at'], clustered_data['Sentiment'], alpha=0.6, color=colors[idx], label=cluster_names[idx])

# Format the dates on the x-axis
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  
# Add labels, a legend, and a title to the plot
plt.xlabel('Date')
plt.ylabel('Sentiment')
plt.title('Sentiment Analysis Over Time with K-Means Clustering')
plt.legend(title='Cluster')

# Show the plot
plt.show()


# Optional: compute the number of occurences of each sentiment 
value_counts = tyields_data['Cluster'].value_counts()

# Printing the count for each value
print("Count for 1 - Positive:", value_counts.get(0, 0))
print("Count for 0 - Neutral:", value_counts.get(2, 0))
print("Count for -1 - Negative:", value_counts.get(1, 0))


######################

# Plotting the daily sentiment count

# Convert the 'created_at'column to just the date, without the time
tyields_data['date'] = tyields_data['created_at'].dt.date

# Group by date and count occurrences of each sentiment cluster
daily_sentiment_counts = tyields_data.groupby(['date', 'Cluster']).size().unstack(fill_value=0)

# Define a function to calculate a daily sentiment score
def calculate_daily_score(row):
    positive_weight = 1
    neutral_weight = 0
    negative_weight = -1
    score = (row.get(0, 0) * positive_weight) + (row.get(1, 0) * neutral_weight) + (row.get(2, 0) * negative_weight)
    return score

# Apply the function to calculate the daily sentiment score
daily_sentiment_counts['daily_score'] = daily_sentiment_counts.apply(calculate_daily_score, axis=1)

# Print the daily sentiment scores
print(daily_sentiment_counts[['daily_score']])



# Plotting the daily sentiment counts
plt.figure(figsize=(12, 6))

# Plot each sentiment cluster count
plt.plot(daily_sentiment_counts.index, daily_sentiment_counts[0], label='Positive', marker='o', linestyle='-', color='green')
plt.plot(daily_sentiment_counts.index, daily_sentiment_counts[1], label='Neutral', marker='o', linestyle='-', color='gray')
plt.plot(daily_sentiment_counts.index, daily_sentiment_counts[2], label='Negative', marker='o', linestyle='-', color='red')

# Format the plot
plt.title('Daily Sentiment Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.grid(True)

# Format the dates on the x-axis for better readability
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

# Show the plot
plt.show()


######################



# Plotting the daily sentiment counts as a stacked bar chart

daily_sentiment_counts.reset_index(inplace=True)

plt.figure(figsize=(12, 6))

# Create the stacked bar chart
plt.bar(daily_sentiment_counts['date'], daily_sentiment_counts[0], label='Positive', color='green')
plt.bar(daily_sentiment_counts['date'], daily_sentiment_counts[1], bottom=daily_sentiment_counts[0], label='Neutral', color='gray')
plt.bar(daily_sentiment_counts['date'], daily_sentiment_counts[2], bottom=daily_sentiment_counts[0] + daily_sentiment_counts[1], label='Negative', color='red')

# Format the plot
plt.title('Prevalence of Sentiment Polarity per Day')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.grid(True)

# Format the dates on the x-axis for better readability
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

# Show the plot
plt.show()



########################

# Create a heatmap for dominant daily sentiment 

# Prepare data

# Group by date and count occurrences of each sentiment cluster
tyields_data['date'] = tyields_data['created_at'].dt.date.astype(str)

daily_sentiment_counts = tyields_data.groupby(['date', 'Cluster']).size().unstack(fill_value=0)

# Define a function to determine the most common sentiment
def most_common_sentiment(row):
    # Identify the cluster with the highest count
    most_common = row.idxmax()
    # Map the cluster number to a sentiment label 
    sentiments = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
    return sentiments.get(most_common, 'Unknown')

# Apply the function to determine the most common sentiment for each day
daily_sentiment_counts['most_common_sentiment'] = daily_sentiment_counts.apply(most_common_sentiment, axis=1)


# Define a mapping from sentiment labels to numerical values
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

# Create a new column 'sentiment_value' by replacing string labels with corresponding numerical values
daily_sentiment_counts['sentiment_value'] = daily_sentiment_counts['most_common_sentiment'].replace(sentiment_mapping)

daily_sentiment_counts.index = pd.to_datetime(daily_sentiment_counts.index)



# Prepare the pivot table for the heatmap
pivot_table = daily_sentiment_counts.pivot_table(
    index=daily_sentiment_counts.index.month, 
    columns=daily_sentiment_counts.index.day, 
    values='sentiment_value'
)

# Define the colormap
cmap = ListedColormap(['red', 'grey', 'green'])  # Negative, Neutral, Positive


# Create the heatmap
plt.figure(figsize=(15, 9))
ax = sns.heatmap(pivot_table, cmap=cmap, cbar=False, linewidths=.5, annot=True)

# Define month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Get the unique months present in the index and set the y-tick labels accordingly
ax.set_yticklabels([month_labels[month - 1] for month in pivot_table.index], rotation=0)

# Create a legend
colors = ['red', 'grey', 'green']  # Negative, Neutral, Positive
sentiments = ['Negative', 'Neutral', 'Positive']
patches = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
legend = ax.legend(patches, sentiments, loc='lower right', title='Sentiment')

# Set title and labels
plt.title('Calendar Heatmap of Daily Sentiment Prevalence')
plt.xlabel('Day of the Month')
plt.ylabel('Month')

# Optional: Adjust layout to make room for the legend
#plt.tight_layout()

# Display the plot
plt.show()



##############################

# Construct and backtest trading strategy

import yfinance as yf 

# Download financial data
ticker = ("GOVT")
start = '2023-09-02'
end = '2023-12-12'
bonds = yf.download(ticker, start, end, interval = '1d')

close = bonds['Adj Close'].to_frame()
close.index = pd.to_datetime(close.index)
daily_sentiment_counts.index = pd.to_datetime(daily_sentiment_counts.index)

ticker2 = 'SPY'  
spy_data = yf.download(ticker2, start, end, interval = '1d')  

close['sentiment'] = daily_sentiment_counts['sentiment_value']

close['Position_long'] = np.where(close['sentiment'] == 1 , 1, np.nan) # sentiment is positive, enter a long position, denoted as 1, write nan if not 
close['Position_short'] = np.where(close['sentiment'] == -1, -1, np.nan) # if sentiment is negative enter a short position, denoted as -1, write nan if not 
close['Position_long_short'] = np.where(close['sentiment'] ==1, 1, np.where(close['sentiment'] == -1, -1, np.nan))
# combines the two preceding conditions into one column

# Extends the positions (first signal stays long until a short happens and vice versa) using .ffill()
close['Position_long_short'].ffill(inplace=True) 


# Obtain the signals
# Done by taking the difference >> if no signal happened in the current day, the difference will be zero, as the position remained the same 
close['Signal_long_short'] = close['Position_long_short'].diff()
close['Signal_long_short'][0] = 2* close['Position_long_short'][0]

# Calculates buy and sell price

close['buy_price'] = np.where((close['Signal_long_short']>0), close['Adj Close'],np.nan) 
# long position occured if signal > 0 (position went from -1 to 1 therefore difference = 2)
close['sell_price'] = np.where((close['Signal_long_short']<0), close['Adj Close'],np.nan)
# long position occured if signal < 0 (position went from 1 to -1 therefore difference = -2)


# Graph the adjusted close price along with buy and sell signals
plt.figure(figsize = (12,10))
plt.subplot(211)
plt.plot(close[['Adj Close', 'buy_price', 'sell_price']], lw=2)
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Sentiment Signals and Adjusted Close Price of GOVT')

# plots the markers for buy(green) and sell(red) signals (at the prices where a long or short position is undertaken)
plt.plot(close.index, close.buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
plt.plot(close.index, close.sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')

# CALCULATE RETURNS

#GOVT
govt_log_returns = np.log(close['Adj Close'] / close['Adj Close'].shift(1))
close['log_returns'] = np.log(close['Adj Close'] / close['Adj Close'].shift(1))

#SPY
benchmark_log_returns = np.log(spy_data['Adj Close'] / spy_data['Adj Close'].shift(1))
close['benchmark_returns'] = np.log(spy_data['Adj Close'] / spy_data['Adj Close'].shift(1))


# CALCULATE STRATEGY RETURNS

close['Strategy_returns'] = close['Position_long_short'].shift(1) * close['log_returns'] 

# BUY AND HOLD
close['buy_hold_cumsum'] = np.exp((close['log_returns']).cumsum()) 
print("stock cum sum: ", round((close['buy_hold_cumsum'][-1]),6)) 

# Sentiment STRATEGY
close['strategy_cumsum'] = np.exp((close['Strategy_returns']).cumsum()) 
print("strat cum sum: ", round((close['strategy_cumsum'][-1]),6))

# BENCHMARK
close['bench_cumsum'] = np.exp((close['benchmark_returns']).cumsum()) 
print("benchmark cum sum: ", round((close['bench_cumsum'][-1]),6))

print("Cumulative Returns:\n", np.exp(close[['log_returns', 'Strategy_returns', 'benchmark_returns']].sum()))

# Graph the results

ax = close[['Strategy_returns', 'log_returns', 'benchmark_returns']].cumsum().apply(np.exp).plot(
    figsize=(10, 6), 
    title="Strategy vs Buy and Hold vs Benchmark", 
    legend=True)

# Rename legend labels
ax.legend(["Sentiment Strategy", "Buy and Hold", "Benchmark"])



