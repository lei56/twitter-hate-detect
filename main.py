import sys
import loader as load
import preprocess as prep
import process as proc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DATA_FILE = "NAACL_SRW_2016.csv"
TWEETS_FILE = "tweets.csv"

def main():
    # Invalid command line arguments provided
    if len(sys.argv) != 2:
        print("Requires 1 command line argument.")
        print("gen: Queries twitter.com for tweets.")
        print("load: Loads tweets from TWEETS_FILE.")
        return
        
    # Ran with gen argument, pull tweets from Twitter, and saves to TWEETS_FILE
    if sys.argv[1] == "gen":
        # Load data from csv file. CSV file is from https://github.com/ZeerakW/hatespeech
        tweet_ids, orig_targets = load.load_csv_file(DATA_FILE)
        print("CSV load complete. Found", len(tweet_ids), "tweets.")
        # Load tweets from the given IDs
        tweets, targets = load.load_tweets(tweet_ids, orig_targets)
        print("Tweet Loading Complete. Found", len(tweets), "tweets.")
        # Save loaded tweets to TWEETS_FILE
        load.save_tweets(tweets, targets, TWEETS_FILE)
        print("Tweet Saving Complete. Saved", len(tweets), "tweets.")
        
    # Ran with load argument, load preprocessed data from data file and train model
    if sys.argv[1] == "load":
        # Load tweets from TWEETS_FILE
        tweets, targets = load.load_tweet_file(TWEETS_FILE)
        print("CSV load complete. Found", len(tweets), "tweets.")
        
    # Convert targets into none: 0, racism: 1, sexism: 2
    vec_targets = []
    for target in targets:
        if target == "racism":
            vec_targets.append(1)
        elif target == "sexism":
            vec_targets.append(2)
        else:
            vec_targets.append(0)
        
    # Preprocess tweet contents
    tweets = prep.preprocess(tweets)
    
    # Turn tweets into numerical features to calcuate TFIDF values
    vector = proc.vectorize(tweets)
    tfidfs = proc.calc_tfidfs(vector)
    
    # Split data into training and test data split
    x_train, x_test, y_train, y_test = train_test_split(tfidfs, vec_targets, test_size=0.2, random_state=0)
    
    # Create classifier and train on data
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    


if __name__ == '__main__':
    main()