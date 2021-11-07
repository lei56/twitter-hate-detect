import sklearn.feature_extraction.text as txt
from nltk.corpus import stopwords


def vectorize(tweets, features=2000):
    """
    This method vectorizes given list of tweets based on frequency of words
    """
    vectorizer = txt.CountVectorizer(max_features=features, min_df=3, max_df=0.5, stop_words=stopwords.words('english'))
    vec = vectorizer.fit_transform(tweets).toarray()
    
    return vec
    
def calc_tfidfs(vector):
    """
    This method calculates TFIDF values from vectorized tweets
    """
    tfidf_converter = txt.TfidfTransformer()
    tfidfs = tfidf_voncerter.fit_transform(vector).toarray()
    
    return tfidfs