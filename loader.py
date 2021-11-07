import csv
import re
import tweepy as tweepy
import pandas as pd


# Twitter dev keys
#CONSUMER_KEY = os.getenv("TW_CONSUMER_KEY")
#CONSUMER_SECRET = os.getenv("TW_CONSUMER_SECRET")
#OAUTH_TOKEN = os.getenv("TW_OAUTH_TOKEN")
#OAUTH_TOKEN_SECRET = os.getenv("OAUTH_TOKEN_SECRET")

def load_csv_file(filename):
    """"
    Loads given csv file with tweet_id, category into dictionary
    """
    col_names = ["id", "classification"]
    df = pd.read_csv(filename, names=col_names)
    tweet_ids = df.id.to_list()
    classifications = df.classification.to_list()
    
    return tweet_ids, classifications
    
def load_tweet_file(filename):
    """
    Loads pre-loaded tweets from file, where tweets are separated by \n.
    """
    col_names = ["tweet"]
    df = pd.read_csv(filename, names=col_names)
    tweets = df.tweet.to_list()
    
    return tweets

def load_tweets(tweet_ids):
    """
    Loads tweets from list of IDs into new list of values
    """
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)
    
    tweets = []
    for id in tweet_ids:
        tweets.append(api.get_status(id))
    
    return tweets
    
def save_tweets(tweets, filedest):
    """
    Saves list of processed tweets into file, separated by \n. 
    Existing \n are removed from tweets.
    """
    with open(filedest, "w") as file:
        for tweet in tweets:
            processed_tweet = re.sub(r'\s', ' ', tweet)
            file.write(processed_tweet + "\n")

        