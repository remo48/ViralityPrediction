import numpy as np
import pandas as pd
import datetime
import os
import sys
import argparse
import shutil
from random import shuffle
import torch as th
from tqdm import tqdm
import networkx as nx
from collections import Counter
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.append('../code/')
from dataset import Cascade

try: os.chdir(os.path.dirname(sys.argv[0]))
except: pass

def logp(x):
    return np.log(x+1)

class Preprocessing:

    def __init__(self, data_dir, raw_data_file='raw_data_anon.csv', raw_emotions_file='emotions_anon.csv', verbose=1):
        '''
        Preprocessing the raw data into experiment data

        args:
            data_dir: Directory, where data is stored
            raw_data_file: Name of raw data file
            raw_emotions_file: Name of raw emotions file
            verbose: Verbosity of output (0 = no output)
        '''

        self.data_dir = data_dir
        self.graphs_dir = os.path.join(self.data_dir, 'graphs/')
        self.grouped_dir = os.path.join(self.data_dir, 'grouped/')
        self.raw_data_file = os.path.join(self.data_dir, raw_data_file)
        self.raw_emotions_file = os.path.join(self.data_dir, raw_emotions_file)
        
        self.tweets_file = os.path.join(self.data_dir, 'tweets.csv')
        self.emo_file = os.path.join(self.data_dir, 'emotions.csv')
        self.tweets_train_file = os.path.join(self.data_dir, 'tweets_train.csv')
        self.tweets_test_file = os.path.join(self.data_dir, 'tweets_test.csv')
        self.emo_train_file = os.path.join(self.data_dir, 'emo_train.csv')
        self.emo_test_file = os.path.join(self.data_dir, 'emo_test.csv')
        self.tweets = None
        self.emo = None
        self.tweets_train = None
        self.tweets_test = None
        self.emo_train = None
        self.emo_test = None

        self.verbose = verbose

        self.to_encode = ['user_followers_log',
                    'user_followees_log',
                    'user_account_age_log',
                    'user_engagement_log',
                    'retweet_delay_log',
                    'user_verified',
                    'hour_cos',
                    'hour_sin',
                    'wd_cos',
                    'wd_sin',
                    'n_children_log',
                    'depth']

        self.to_encode_structureless = ['user_followers_log',
                                'user_followees_log',
                                'user_account_age_log',
                                'user_engagement_log',
                                'retweet_delay_log',
                                'user_verified',
                                'hour_cos',
                                'hour_sin',
                                'wd_cos',
                                'wd_sin']

        self.to_group = ['user_followers_log',
                    'user_followees_log',
                    'user_account_age_log',
                    'user_engagement_log',
                    'retweet_delay_log']

        self.to_standardize = ['user_followers_log', 
                        'user_followees_log',
                        'user_engagement_log', 
                        'user_account_age_log',
                        'retweet_delay_log',
                        'depth']


    def feature_engineer(self):
        '''
        This function processes the raw data files and loads it into memory
        '''
        dtypes = {
            'tid' : int,
            'veracity' : object,
            'cascade_id' : int,
            'rumor_id' : object,
            'rumor_category' : object,
            'parent_tid' : int,
            'tweet_date' : object,
            'user_account_age' : object,
            'user_verified' : object,
            'user_followers' : object,
            'user_followees' : object,
            'user_engagement' : float,
            'cascade_root_tid' : int,
            'was_retweeted' : int
        }

        print("Start preprocessing the raw data ")
        print("Load data... ", end='', flush=True)
        df = pd.read_csv(self.raw_data_file, dtype=dtypes, na_values='None')
        df = df.drop(['rumor_id', 'rumor_category', 'veracity'], axis=1)

        df['user_verified'] = df.user_verified.fillna(False).astype(bool).astype(int)

        df['datetime'] = [datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in df.tweet_date]
        print(df.shape[0], 'tweets loaded')

        print("Computing root and retweet delay... ", end='', flush=True)
        parents = df.loc[df.was_retweeted == 1, ['tid', 'datetime']]
        parents.columns = ['parent_' + c for c in parents.columns]
        df = pd.merge(df, parents, how='left', on=['parent_tid'])
        df['retweet_delay'] = [self.__timediff(time, parent_time) for time, parent_time in zip(df.datetime, df.parent_datetime)]
        del parents

        roots = df.loc[df.parent_tid == -1, ['cascade_id', 'datetime']]
        roots.rename(columns={'datetime': 'root_datetime'}, inplace=True)
        df = pd.merge(df, roots, on='cascade_id')
        df['root_delay'] = [self.__timediff(time, root_time) for time, root_time in zip(df.datetime, df.root_datetime)]
        del roots
        print('done')

        df = df.drop(['parent_datetime', 'root_datetime', 'was_retweeted', 'tweet_date'], axis=1)

        print("Encode tweet date... ", end='', flush=True)
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
        print('done')

        print("Add cascade size... ", end='', flush=True)
        grpd = df.groupby('cascade_id')
        exposure = grpd.size().to_frame('cascade_size')
        df = pd.merge(df, exposure, on=['cascade_id'])
        del grpd
        del exposure
        print("done")

        df.sort_values(['cascade_id', 'tid'], inplace=True)

        print("Load emotion data... ", end='', flush=True)
        df_emo = pd.read_csv(raw_emotions_file)

        ids = df[['tid', 'cascade_id']]
        df_emo = pd.merge(ids, df_emo.drop('misc', axis=1), left_on='tid', right_on='tweet_id')
        df_emo = df_emo.drop(['tid', 'tweet_id'], axis=1)
        print("done")

        print("Impute missing user data... ", end='', flush=True)
        si = SimpleImputer(strategy='median')
        df[['user_followers', 'user_followees', 'user_account_age']] = si.fit_transform(df[['user_followers', 'user_followees', 'user_account_age']].values)

        self.tweets = df
        self.emo = df_emo
        print("done")
        
        print("Saving dataframes to directory... ", end='', flush=True)
        self.tweets.to_csv(self.tweets_file, header=True, index=False)
        self.emo.to_csv(self.emo_file, header=True, index=False)
        print("done")


    def generate_experiment_data(self, split_ratio=0.85, lower_threshold=100, upper_threshold=10000):
        print("Start generating experiment data...")
        ss = StandardScaler()

        if self.tweets is None:
            self.tweets = pd.read_csv(self.tweets_file)

        if self.emo is None:
            self.emo = pd.read_csv(self.emo_file)

        tweets = self.tweets[(self.tweets.cascade_size >= lower_threshold) & (self.tweets.cascade_size <= upper_threshold)].reset_index(drop=True)
        emos = self.emo

        tweets = tweets.sort_values(['cascade_id', 'datetime']).reset_index(drop=True)
        tweets['new_tid'], tweets['new_parent_tid'] = self.__get_new_tid(tweets)
        tweets['depth'] = self.__get_depths(tweets)

        IDs = list(set(tweets.cascade_id).intersection(emos.cascade_id))

        shuffle(IDs)
        split = int(len(IDs) * split_ratio)
        train_ids, test_ids = pd.DataFrame({'cascade_id': IDs[:split]}), pd.DataFrame({'cascade_id': IDs[split:]})

        tweets_train = pd.merge(tweets, train_ids, how='inner').reset_index(drop=True)
        tweets_test = pd.merge(tweets, test_ids, how='inner').reset_index(drop=True)
        emo_train = pd.merge(emos, train_ids, how='inner').reset_index(drop=True)
        emo_test = pd.merge(emos, test_ids, how='inner').reset_index(drop=True)
        del tweets
        del emos

        for cname in ['user_followers', 'user_followees', 'user_engagement', 'user_account_age', 'retweet_delay', 'cascade_size']:
            tweets_train[cname + '_log'] = logp(tweets_train[cname].values)
            tweets_test[cname + '_log'] = logp(tweets_test[cname].values)
        tweets_train = tweets_train.drop(['user_followers', 'user_followees', 'user_engagement', 'user_account_age', 'retweet_delay', 'cascade_size'], axis=1)
        tweets_test = tweets_test.drop(['user_followers', 'user_followees', 'user_engagement', 'user_account_age', 'retweet_delay', 'cascade_size'], axis=1)
        
        tweets_train[self.to_standardize] = ss.fit_transform(tweets_train[self.to_standardize].values)
        tweets_test[self.to_standardize] = ss.transform(tweets_test[self.to_standardize].values)

        emo_train.iloc[:, 1:] = ss.fit_transform(emo_train.iloc[:, 1:].values)
        emo_test.iloc[:, 1:] = ss.transform(emo_test.iloc[:, 1:].values)

        print("Saving experiment data to directory... ", end='', flush=True)
        tweets_train.to_csv(self.tweets_test_file, header=True, index=False)
        tweets_test.to_csv(self.tweets_test_file, header=True, index=False)
        emo_train.to_csv(self.emo_train_file, header=True, index=False)
        emo_test.to_csv(self.emo_test_file, header=True, index=False)

        self.tweets_train = tweets_train
        self.tweets_test = tweets_test
        self.emo_train = emo_train
        self.emo_test = emo_test
        print("done")


    def generate_cascades(self, crops_dict):
        self.__cleanup()
        ss = StandardScaler()
        ss_grouped = StandardScaler()

        if self.tweets_train is None:
            self.tweets_train = pd.read_csv(self.tweets_train_file)

        if self.tweets_test is None:
            self.tweets_test = pd.read_csv(self.tweets_test_file)

        if self.emo_train is None:
            self.emo_train = pd.read_csv(self.emo_train_file)

        if self.emo_test is None:
            self.emo_test = pd.read_csv(self.emo_test_file)

        for (df_tweets, df_emo, post) in zip(*[(self.tweets_train.copy(), self.tweets_test.copy()), (self.emo_train, self.emo_test), ('', '_test')]):   
            not_cropped = self.crop(df_tweets)   
            if post == '':
                not_cropped['n_children_log'] = ss.fit_transform(not_cropped.n_children.values.reshape(-1, 1))
            else:
                not_cropped['n_children_log'] = ss.transform(not_cropped.n_children.values.reshape(-1, 1))
            
            grouped = self.to_grouped(not_cropped, self.to_group)
            
            if post == '':
                grouped.iloc[:, 2:] = ss_grouped.fit_transform(grouped.iloc[:, 2:])
            else:
                grouped.iloc[:, 2:] = ss_grouped.transform(grouped.iloc[:, 2:])

            pd.merge(grouped, df_emo, on='cascade_id').to_csv(self.grouped_dir + 'grouped' + post + '.csv', header=True, index=False)

            self.__save_cascades(not_cropped, df_emo, self.to_encode, post)
            self.__save_cascades(not_cropped, df_emo, self.to_encode_structureless, '_structureless' + post)


        for k, v in crops_dict.items():  
            # loop through train and test
            for (df_tweets, df_emo, post) in zip(*[(self.tweets_train.copy(), self.tweets_test.copy()), (self.emo_train, self.emo_test), ('', '_test')]):
                cropped = self.crop(df_tweets, v)
                if post == '':
                    cropped['n_children_log'] = ss.fit_transform(cropped.n_children.values.reshape(-1, 1))
                else:
                    cropped['n_children_log'] = ss.transform(cropped.n_children.values.reshape(-1, 1))

                grouped = self.to_grouped(cropped, self.to_group)            
                if post == '':
                    grouped.iloc[:, 2:] = ss_grouped.fit_transform(grouped.iloc[:, 2:])
                else:
                    grouped.iloc[:, 2:] = ss_grouped.transform(grouped.iloc[:, 2:])
                
                pd.merge(grouped, df_emo, on='cascade_id').to_csv(self.grouped_dir + 'grouped_' + k + post + '.csv', header=True, index=False)

                self.__save_cascades(cropped, df_emo, self.to_encode, '_' + k + post)


    def crop(self, tweets, threshold=False):
        """
        Returns df cropped at a certain threshold

        Args :
            tweets: df with tweets
            threshold: threshold of time
        """
        def get_parents(df):
            # returns df with number of children per tweet
            df = df.copy()
            parents = df.groupby(['cascade_id', 'new_parent_tid']).agg({'tid': 'count'}).reset_index()
            parents = parents.rename(columns={'tid': 'n_children', 'new_parent_tid': 'new_tid'})
            return parents

        cropped = tweets.copy()

        if threshold: 
            cropped = cropped[cropped.root_delay < threshold]

        cropped = pd.merge(cropped, get_parents(cropped), on=['cascade_id', 'new_tid'], how='left')
        cropped = cropped.fillna({'n_children': 0})
        cropped['n_children_log'] = logp(cropped.n_children)
        cropped['is_leaf'] = (cropped.n_children == 0).astype(int)
        return cropped


    def to_grouped(self, df, to_group):
        """
        Create df for standard feature classifiers with 1 cascade per row
        In: 
            - df : all tweets
            - to_group : variables to average per cascade
        Out: df with averaged node vars and aggregate statistics per cascade
        """
        
        def get_cascade_statistics(small):
            # get depth, breadth and depth to breadth ratio for a single cascade
            
            depths = small.depth
            cascade_depth = max(depths)
            cascade_breadth = max(Counter(depths).values())
            db_ratio = cascade_depth / cascade_breadth

            return cascade_depth, cascade_breadth, db_ratio

        aggs = {k: 'mean' for k in to_group}
        aggs['n_children_log'] = 'max' 
            
        grouped = df.groupby(['cascade_id', 'cascade_size_log']).agg(aggs).reset_index()

        sizes = []
        depths = []
        breadths = []
        db_ratios = []

        for cid in grouped.cascade_id:
            small = df[df.cascade_id == cid]
            depth, breadth, db_ratio = get_cascade_statistics(small)
            sizes.append(small.shape[0])
            depths.append(depth)
            breadths.append(breadth)
            db_ratios.append(db_ratio)

        grouped['size'], grouped['depth'], grouped['breadth'] = sizes, depths, breadths
        grouped['db_ratio'] = db_ratios

        return grouped


    def __timediff(self, t1, t2):
        """
        Function that calculates dime difference in seconds between two datetimes
        
        Args: time1, time2
        """
        if pd.isnull(t2):
            return 0
        else:
            return (t1 - t2).total_seconds()


    def __get_new_tid(self, df):
        """
        Assigns to tweets an id from 0 to N per cascade  in chronological order
        where N is the size of the cascade -1
        In : df with tweets
        Out: series with new tweets and new parent tweets
        """

        df = df.copy()
        df['new_tid'] = df.groupby('cascade_id').cumcount().astype(int)
        parents = df[['cascade_id', 'tid', 'new_tid']]
        parents = parents.rename({'tid': 'parent_tid', 'new_tid': 'new_parent_tid'}, axis=1)
        df = pd.merge(df, parents, on=['cascade_id', 'parent_tid'], how='left')
        df['new_parent_tid'] = df['new_parent_tid'].fillna(-1).astype('int64')
        return df['new_tid'], df['new_parent_tid']


    def __get_depths(self, df):
        """
        get node depth for ALL CASCADES
        In : df with all tweets 
        Out : list with depth of all nodes for every cascade
        """

        print('getting_depths')
        
        all_depths = []
        
        for cid in tqdm(df.cascade_id.unique()):
            small = df[df.cascade_id == cid].copy()
            all_depths.append(self.__get_nodes_depths(small))
        
        return list(chain(*all_depths))        
        
        
    def __get_nodes_depths(self, small):
        """
        get nodes depth for SINGLE CASCADE
        In : df with tweets for a single cascade
        Out : list with depth of all nodes
        """
        
        # create networkx graph
        g = nx.DiGraph()
        # add cascade nodes   
        g.add_nodes_from(small.new_tid.values)
        # add edges ; skip first because root node has no incoming parent
        g.add_edges_from([(u, v) for u, v in zip(small.new_parent_tid, small.new_tid)][1:])

        depths = nx.shortest_path_length(g, 0)
        
        return [depths[k] for k in sorted(depths.keys())]


    def __save_cascades(self, df, df_emo, to_encode, name_append=''):
        """
        Save all the cascades with custom "Cascade" data structure and pt ~"pytorch" format
        including nodes, edges,nodes covariates (X), label, root features
        In:
            - df: df with all the tweets with covariates
            - df_emo: df with affective response to root tweet
            - to_encode: column names to select as node covariates
            - name_append: refers to specific cascade variant. Nothin for standard, otherwise 
            crop type, structureless, test ...
        """
        print('saving cascades', name_append, flush=True)
        
        for cid in tqdm(df.cascade_id.unique()):	
            small = df[df.cascade_id == cid].copy().reset_index()
            emo = df_emo[df_emo.cascade_id == cid]

            if emo.empty:
                continue

            th_emo = th.Tensor(emo.iloc[:, 1:].values)

            th_cascade_size = th.tensor([[small.cascade_size_log[0]]], dtype=th.float32)
            
            X = th.Tensor(small[to_encode].values)
            is_leaf = th.Tensor(small.is_leaf.values)
            src, dest = small.new_tid[1:].values, small.new_parent_tid[1:].values

            c = Cascade(cid, X, th_cascade_size, th_emo, src, dest, is_leaf)

            th.save(c, self.graphs_dir + str(cid) + name_append + '.pt')

    
    def __cleanup(self):
        print('cleanup graph directory')
        for dir in [self.graphs_dir, self.grouped_dir]:
            for filename in os.listdir(dir):
                file_path = os.path.join(self.graphs_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


def cascade_generator():
    if lower_threshold != -1 and upper_threshold != -1:
        # get cascade ids of cascades whose size is between lower and upper thesholds<
        cascade_ids = [k for k, v in Counter(tweets.cascade_id).items() if v >= lower_threshold and v < upper_threshold]
        vprint('cascade ids retrieved')
        sieve = pd.DataFrame({'cascade_id': cascade_ids})
        vprint('started merging')
        tweets = pd.merge(sieve, tweets, how='left', on='cascade_id')
        vprint('finished merging')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest = 'data_dir', default = '../data/', type = str)
    parser.add_argument('--verbose', dest='verbose', default=1, type=int)
    parser.add_argument('--feature_engineer', action='store_true', default=False, dest='feature_engineer')

    args = parser.parse_args()

    crops = {
    '15_mins': 60 * 15,
    '30_mins': 60 * 30,
    '1_hour': 60. * 60,
    '3_hour': 60. * 60 * 3,
    '24_hour': 60. * 60 * 24}

    data_dir = args.data_dir
    raw_data_file = args.data_dir + 'raw_data_anon.csv'
    raw_emotions_file = args.data_dir + 'emotions_anon.csv'
    raw_metadata_file = args.data_dir + 'metadata_anon.txt'

    preprocessor = Preprocessing(data_dir)
    preprocessor.feature_engineer()
    preprocessor.generate_experiment_data()
    preprocessor.generate_cascades(crops)

    

    