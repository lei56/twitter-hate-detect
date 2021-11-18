import sys
import loader as load
import preprocess as prep
import process as proc
import evaluate as eva
import sklearn.ensemble
from sklearn.model_selection import train_test_split


DATA_FILE = "NAACL_SRW_2016.csv"
TWEETS_FILE = "tweets.csv"

def main():
    # Invalid command line arguments provided
    if len(sys.argv) != 2:
        print("Requires 1 command line argument.")
        print("gen: Queries twitter.com for tweets.")
        print("load: Loads tweets from TWEETS_FILE and trains models.")
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
        return
        
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
        
        # Create classifiers and train on data
        # Bagging Classifier
        bc = sklearn.ensemble.BaggingClassifier(n_estimators=1000, random_state=0)
        bc.fit(x_train, y_train)
        y_p_bc = bc.predict(x_test)
        # Evaluation
        print("Bagging Classifier Scores:")
        eva.evaluate(y_test, y_p_bc)
        # Save model
        load.save_model("bc_classifier", bc)
        
        # AdaBoost Classifier
        abc = sklearn.ensemble.AdaBoostClassifier(n_estimators=1000, random_state=0)
        abc.fit(x_train, y_train)
        y_p_abc = abc.predict(x_test)
        # Evaluation
        print("AdaBoost Classifier Scores:")
        eva.evaluate(y_test, y_p_abc)
        # Save model
        load.save_model("abc_classifier", abc)
        
        # Random Forest Classifier
        rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=0)
        rfc.fit(x_train, y_train)
        y_p_rfc = rfc.predict(x_test)
        # Evaluation
        print("Random Forest Classifier Scores:")
        eva.evaluate(y_test, y_p_rfc)
        # Save model
        load.save_model("rfc_classifier", rfc)
        
        # Extra Trees Classifier
        etc = sklearn.ensemble.ExtraTreesClassifier(n_estimators=1000, random_state=0)
        etc.fit(x_train, y_train)
        y_p_etc = etc.predict(x_test)
        # Evaluation
        print("Extra Trees Classifier Scores:")
        eva.evaluate(y_test, y_p_etc)
        # Save model
        load.save_model("etc_classifier", etc)
        
        # Gradient Boosting Classifier
        gbc = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
        gbc.fit(x_train, y_train)
        y_p_gbc = gbc.predict(x_test)
        # Evaluation
        print("Gradient Boosting Classifier Scores:")
        eva.evaluate(y_test, y_p_gbc)
        # Save model
        load.save_model("gbc_classifier", gbc)
        
        # Histogram Gradient Boosting Classifier
        hgbc = sklearn.ensemble.HistGradientBoostingClassifier(random_state=0)
        hgbc.fit(x_train, y_train)
        y_p_hgbc = hgbc.predict(x_test)
        # Evaulation
        print("Histogram Gradient Boosting Classifier Scores:")
        eva.evaluate(y_test, y_p_hgbc)
        # Save model
        load.save_model("hgb_classifier", hgbc)
    
        return
    
    
if __name__ == '__main__':
    main()