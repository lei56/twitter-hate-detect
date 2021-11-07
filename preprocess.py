import re
import nltk

def preprocess(tweets):
    """
    This applies text preprocessing to clean up a given list of tweets. 
    Largely based on https://stackabuse.com/text-classification-with-python-and-scikit-learn/
    """
    stemmer = nltk.stem.WordNetLemmatizer()
    processed = []
    
    for tweet in tweets:
        # Remove special characters
        proc = re.sub(r'\W', ' ', tweet)
        # Remove all single characters
        proc = re.sub(r'\s+[a-zA-z]\s+', ' ', proc)
        # Remove single characters from start
        proc = re.sub(r'\^[a-zA-z]\s+', ' ', proc)
        # Substitute multiple spaces with a single space
        proc = re.sub(r'\s+', ' ', proc, flags=re.I)
        # Remove prefixed 'b'
        proc = re.sub(r'^b\s+', '', proc)
        # Convert to all lowercase
        proc = proc.lower()
        # Lemmatizationn
        proc = proc.split()
        proc = [stemmer.lemmatize(word) for word in proc]
        proc = ' '.join(proc)
        # Processing finished, append to processed list
        processed.append(proc)
    
    return processed