import numpy as np
import pandas as pd
import datetime
import os
import sys

source_dir = '../data/'
dest_dir = '../data/'
raw_data_file = source_dir + 'raw_data_anon.csv'
raw_emotions_file = source_dir + 'emotions_anon.csv'
raw_metadata_file = source_dir + 'metadata_anon.txt'
tweets_file = dest_dir + 'tweets.csv'
emotions_file = dest_dir + 'emotions.csv'

try: os.chdir(os.path.dirname(sys.argv[0]))
except: pass

def timediff(t1, t2):
    """
    Function that calculates dime difference in seconds between two datetimes
    In: time1, time2
    Out: time difference in seconds 
    """
    if pd.isnull(t2):
        return 0
    else:
        return (t1 - t2).total_seconds()
    

df = pd.read_csv(raw_data_file, na_values='None')

# user verified
df['user_verified'] = df.user_verified.fillna(False).astype(int)

# veracity
df.loc[df.veracity == 'FALSE', 'veracity'] = False
df.loc[df.veracity == 'TRUE', 'veracity'] = True

df = df[df.veracity.isin([False, True])]

df['datetime'] = [datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in df.tweet_date]
# select tweets with children and rename then to parent
parents = df.loc[df.was_retweeted == 1, ['tid', 'datetime']]
parents.columns = ['parent_' + c for c in parents.columns]
# merce df with parent time with tweets df
df = pd.merge(df, parents, how='left', on=['parent_tid'])
df['retweet_delay'] = [timediff(time, parent_time) for time, parent_time in zip(df.datetime, df.parent_datetime)]
del parents


# hour; cyclic encoding
h = np.array([dt.hour for dt in df['datetime']])
df['hour_cos'] = np.cos(2 * np.pi * h / 23.0)
df['hour_sin'] = np.sin(2 * np.pi * h / 23.0)
del h

# weekday; cyclic encoding
weekday = np.array([dt.weekday() for dt in df['datetime']])
df['wd_cos'] = np.cos(2 * np.pi * weekday / 6.0)
df['wd_sin'] = np.sin(2 * np.pi * weekday / 6.0)
del weekday

# delay from root
roots = df.loc[df.parent_tid == -1, ['cascade_id', 'datetime']]
roots.rename(columns={'datetime': 'root_datetime'}, inplace=True)
df = pd.merge(df, roots, on='cascade_id')
df['root_delay'] = [timediff(time, root_time) for time, root_time in zip(df.datetime, df.root_datetime)]
del roots

# include metadata
META_VARS = ['virality']
meta = []
with open(raw_metadata_file, 'r') as fin:
    for line in fin.readlines():
        cascade_id, metadata = eval(line)
        metadata = {k:metadata[k] for k in META_VARS}
        metadata['cascade_id'] = cascade_id
        meta.append(metadata)

df_meta = pd.DataFrame(meta)
df_meta['virality'] = df_meta.virality.fillna(False).astype(float)
df = pd.merge(df, df_meta, on='cascade_id')
del df_meta
del meta

# root tweet exposure
grpd = df.groupby('cascade_id')
exposure = grpd.size().to_frame('cascade_size')
exposure['cascade_followers'] = grpd['user_followers'].sum()

df = pd.merge(df, exposure, on=['cascade_id'])
del grpd
del exposure


df.sort_values(['cascade_id', 'tid'], inplace=True)
df.to_csv(tweets_file, header=True, index=False)

# EMOTIONS

df_emo = pd.read_csv(raw_emotions_file)

ids = df[['tid', 'cascade_id']]
# drop "misc" column and add to emo df column with corresponding cascade id, rather than root tweet
df_emo = pd.merge(ids, df_emo.drop('misc', axis=1), left_on='tid', right_on='tweet_id')
df_emo = df_emo.drop(['tid', 'tweet_id'], axis=1)

df_emo.to_csv(emotions_file, header=True, index=False)
