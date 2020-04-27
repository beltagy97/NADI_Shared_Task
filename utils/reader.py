import re
import pyarabic.araby as arb
import pandas as pd
import numpy as np




labels_dictionary = {'Iraq':0,
'Egypt': 1,
'Algeria':2,
'Yemen':3,
'Saudi_Arabia':4,
'Syria':5,
'United_Arab_Emirates':6,
'Oman':7,
'Jordan':8,
'Tunisia':9,
'Kuwait':10,
'Morocco':11,
'Libya':12,
'Qatar':13,
'Lebanon':14,
'Sudan':15,
'Mauritania':16,
'Palestine':17,
'Somalia':18,
'Bahrain':19,
'Djibouti':20}


def clean_tweet(tweet):
    result = re.sub(r"http\S+", "", tweet)
    result=re.sub(r"pic\S+", "", result)
    result = re.sub(r"@\S+","",result)
    result = arb.strip_tashkeel(result)
    return result


def read_tweets(filename):
    """
    function to read labeled training set or dev set and output dataframe
    """
    
    df = pd.read_csv( filename +".tsv", sep="\t",dtype="string")
    df['#2 tweet_content'] = df['#2 tweet_content'].apply(lambda x : clean_tweet(x))
    df['label'] = df['#3 country_label'].apply(lambda x : labels_dictionary.get(x)) 
    return df



def read_tweets_unlabeled(path):
    """
    function to read unlabeled data and output a dataframe
    """
    filenames = glob.glob(os.path.join(path, "*.csv"))
    li = []

    for file in filenames:
        df = pd.read_csv(file)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    del li
    frame = frame.dropna()
    return frame
