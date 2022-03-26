import pandas as pd
import re

def get_data(file):
    df = pd.read_csv(file)
    ## Making it all lower case
    df.content = df.content.str.lower()
    df['contents_new'] = df['content'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
    df['contents_newer'] = df['contents_new'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    
    ## Getting rid of links
    # df1['post_title'].str.replace('http.*.com', '',regex = True)
    # df.content = df.content.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
    # df.content.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))

    tweets = df[['contents_newer']].copy()

    tweets.to_csv(r'twitter_data/preprocessed_donaldtrump.csv', header=None, index=None, sep='\t', mode='a')




get_data('twitter_data/realdonaldtrump.csv')