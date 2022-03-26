import pandas as pd
import re

def cleaning_pic_url (text):
    text = re.sub(r'pic.twitter.com/[\w]*',"", text)
    return text
def cleaning_quotes(text):
    text = re.sub(r'--do[\w]*',"", text)
    return text

def cleaning_mentions(text):
    text = re.sub("@[A-Za-z0-9_]+","", text)
    return text

def cleaning_hashtag(text):
    text = re.sub("#[A-Za-z0-9_]+","", text)
    return text

def removing_cont(text):
    text = re.sub("(cont)","", text)
    return text

def get_data(file):
    df = pd.read_csv(file)
    ## Making it all lower case
    df.content = df.content.str.lower()
    df['contents_w/0_http'] = df['content'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
    df['contents_w/0_https'] = df['contents_w/0_http'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    df['contents_w/0_https'] =  df['contents_w/0_https'].apply(lambda x: cleaning_pic_url(x))
    df['contents_w/0_https'] =  df['contents_w/0_https'].apply(lambda x: cleaning_quotes(x))
    df['contents_w/0_https'] =  df['contents_w/0_https'].apply(lambda x: cleaning_mentions(x))
    df['contents_w/0_https'] =  df['contents_w/0_https'].apply(lambda x: cleaning_hashtag(x))
    df['contents_w/0_https'] =  df['contents_w/0_https'].apply(lambda x: removing_cont(x))


    tweets = df['contents_w/0_https']
    tweets.to_csv(r'twitter_data/preprocessed_donaldtrump.txt', header=None, index=None, sep='\t', mode='a')




get_data('twitter_data/realdonaldtrump.csv')