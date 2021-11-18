import os
import csv
import re
import tweepy as tweepy
import pickle
import pandas as pd
import random
import time


TWEET_LIMIT = 1000

# Twitter dev keys
CONSUMER_KEY = os.getenv("TW_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("TW_CONSUMER_SECRET")
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
    col_names = ["tweet", "classification"]
    df = pd.read_csv(filename, names=col_names, encoding='latin1')
    df.tweet.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
    tweets = df.tweet.to_list()
    classifications = df.classification.to_list()

    return tweets, classifications

def load_tweets(tweet_ids, orig_targets):
    """
    Loads tweets from list of IDs into new list of values
    """
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)

    orig_dict = {}
    for idx in range(len(tweet_ids)):
        orig_dict[tweet_ids[idx]] = orig_targets[idx]
    keys = list(orig_dict.keys())
    random.shuffle(keys)

    tweets = []
    targets = []
    for key in keys:
        if len(tweets) > TWEET_LIMIT:
            break
        elif len(tweets) % 10 == 0:
            print(len(tweets), "tweets found")
        tweet_id = key
        target = orig_dict[key]
        try:
            tweets.append(api.get_status(tweet_id).text)
            targets.append(target)
        except Exception as e:
            if "Too Many Requests" in repr(e):
                print(e)
                break
        time.sleep(3)

    return tweets, targets
    
def save_tweets(tweets, targets, filedest):
    """
    Saves list of processed tweets into file, separated by \n. 
    Existing \n are removed from tweets.
    """
    with open(filedest, "w", newline='') as file:
        writer = csv.writer(file)
        for t in range(len(tweets)):
            try:
                writer.writerow([tweets[t], targets[t]])
            except Exception as e:
                print(e)

    with open("tweets.txt", "w") as file:
        for t in range(len(tweets)):
            try:
                file.write(tweets[t])
                file.write(",")
                file.write(targets[t])
                file.write('\n')
            except Exception as e:
                print(e)
                
def save_model(classifier_filename, classifier):
    """
    Uses pickle library to save classifier
    """
    with open(classifier_filename, 'wb') as pfile:
        pickle.dump(classifier, pfile)
        print("Classifier saved at", classifier_filename)
        
def load_model(classifier_filename):
    """
    Uses pickle library to load classifier
    """
    with open(classifier_filename, 'rb') as fmodel:
        classifier = pickle.load(fmodel)
        print("Classifier loaded from", classifier_filename)
        return classifier
