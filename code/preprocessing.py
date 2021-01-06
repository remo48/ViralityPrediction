from pathlib import Path
import numpy as np
import pandas as pd
import os
import sys
import argparse
import shutil
from random import shuffle
import torch as th
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from dataset import Cascade
from utils import Logger

def logp(x):
    return np.log(x+1)

class Preprocessor:

    def __init__(self, data_dir, verbose=1):
        '''
        Preprocessing the raw data into experiment data

        args:
            data_dir: Directory, where data is stored
            raw_data_file: Name of raw data file
            raw_emotions_file: Name of raw emotions file
            verbose: Verbosity of output (0 = no output)
        '''

        self.data_dir = data_dir

        self.logger = Logger(verbose)

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

        self.to_group = {'user_followers_log' : 'mean',
                    'user_followees_log' : 'mean',
                    'user_account_age_log' : 'mean',
                    'user_engagement_log': 'mean',
                    'retweet_delay_log' : 'mean',
                    'n_children_log': 'max',
                    'depth' : 'max'}

        self.to_standardize = ['user_followers_log', 
                        'user_followees_log',
                        'user_engagement_log', 
                        'user_account_age_log',
                        'retweet_delay_log',
                        'depth',
                        'n_children_log']


    def __load_data(self, raw_data_file='raw_data_anon.csv', raw_emotions_file='emotions_anon.csv', lower_threshold=100, upper_threshold=10000):
        '''
        This function loads the raw data in the Preprocessor
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
        raw_data_file = os.path.join(self.data_dir, raw_data_file)
        raw_emotions_file = os.path.join(self.data_dir, raw_emotions_file)

        self.logger.step_start('Load data...')
        df = pd.read_csv(raw_data_file, dtype=dtypes, na_values='None')
        self.logger.step_end('{} tweets loaded'.format(df.shape[0]))

        df['datetime'] = pd.to_datetime(df['tweet_date'], format='%Y-%m-%d %H:%M:%S')
        df = df.drop(['rumor_id', 'rumor_category', 'veracity', 'tweet_date'], axis=1)

        self.logger.step_start('Load emotions data...')
        df_emo = pd.read_csv(raw_emotions_file)
        ids = df[['tid', 'cascade_id']]
        df_emo = pd.merge(ids, df_emo.drop('misc', axis=1), left_on='tid', right_on='tweet_id')
        df_emo = df_emo.drop(['tid', 'tweet_id'], axis=1)
        self.logger.step_end()

        self.logger.step_start('Drop cascades without emotion data...')
        df = df[df.cascade_id.isin(df_emo.cascade_id)]
        self.logger.step_end('{} tweets after dropping'.format(df.shape[0]))

        self.logger.step_start('Drop cascades not in threshold...')
        df_cascade_size = df['cascade_id'].value_counts().reset_index()
        df_cascade_size.columns = ['cascade_id', 'cascade_size']

        df = pd.merge(df, df_cascade_size, on='cascade_id')
        df = df[(df.cascade_size >= int(lower_threshold)) & (df.cascade_size <= int(upper_threshold))].reset_index(drop=True)
        self.logger.step_end('{} tweets after filtering'.format(df.shape[0]))

        return df, df_emo, df_cascade_size

    def __process_cascade_size(self, df):
        df['cascade_size_log'] = logp(df['cascade_size'].values)
        df['category'] = pd.cut(df['cascade_size'], bins=[0,1,100,1000,100000], labels=[0,1,2,3])
        return df

    def __impute_data(self, df):
        self.logger.step_start('Impute missing user data...')
        df['user_verified'] = (df.user_verified == 'True').astype(int)
        
        si = SimpleImputer(strategy='median')
        df[['user_followers', 'user_followees', 'user_account_age']] = si.fit_transform(df[['user_followers', 'user_followees', 'user_account_age']].values)
        self.logger.step_end()

        return df

    def __compute_delay(self, df):
        self.logger.step_start('Compute retweet delay...')
        parents = df.loc[df.was_retweeted == 1, ['tid', 'datetime']]
        parents.columns = ['parent_' + c for c in parents.columns]
        df = pd.merge(df, parents, how='left', on=['parent_tid'])
        df['retweet_delay'] = (df['datetime'] - df['parent_datetime']).dt.total_seconds().fillna(0).astype(int)
        del parents
        self.logger.step_end()

        self.logger.step_start('Compute root delay...')
        roots = df.loc[df.parent_tid == -1, ['cascade_id', 'datetime']]
        roots.rename(columns={'datetime': 'root_datetime'}, inplace=True)
        df = pd.merge(df, roots, on='cascade_id')
        df['root_delay'] = (df['datetime'] - df['root_datetime']).dt.total_seconds().fillna(0).astype(int)
        del roots
        self.logger.step_end()
        df = df.drop(['parent_datetime', 'root_datetime'], axis=1)

        return df

    def __encode_tweet_date(self, df):
        self.logger.step_start('Encode tweet hour...')
        h = np.array([dt.hour for dt in df['datetime']])
        df['hour_cos'] = np.cos(2 * np.pi * h / 23.0)
        df['hour_sin'] = np.sin(2 * np.pi * h / 23.0)
        del h
        self.logger.step_end()

        self.logger.step_start('Encode tweet weekday...')
        weekday = np.array([dt.weekday() for dt in df['datetime']])
        df['wd_cos'] = np.cos(2 * np.pi * weekday / 6.0)
        df['wd_sin'] = np.sin(2 * np.pi * weekday / 6.0)
        del weekday
        self.logger.step_end()

        return df

    def __compute_new_tid(self, df):
        self.logger.step_start('Compute new tid...')
        df = df.sort_values(by=['cascade_id', 'root_delay'])
        df['new_tid'] = df.groupby('cascade_id').cumcount()

        parents = df.loc[df.was_retweeted == 1, ['tid', 'new_tid']]
        parents.columns = ['parent_tid', 'new_parent_tid']
        df = pd.merge(df, parents, on=['parent_tid'], how='left')
        df['new_parent_tid'] = df['new_parent_tid'].fillna(-1).astype(int)
        self.logger.step_end()

        return df

    def __compute_depth(self, df):
        self.logger.step_start('Compute depth...')
        df = df.sort_values(by=['cascade_id', 'new_tid'])
        all_depths = []

        for cid in df.cascade_id.unique():
            cascade = df[df['cascade_id'] == cid]
            g = nx.DiGraph()
            g.add_nodes_from(cascade.new_tid.values)
            g.add_edges_from([(u, v) for u, v in zip(cascade.new_parent_tid, cascade.new_tid)][1:])
            depths = nx.single_source_shortest_path_length(g, 0)
            all_depths += [v for _,v in sorted(depths.items())]

        df['depth'] = all_depths
        self.logger.step_end()
        return df

    def feature_engineer(self, 
                        raw_data_file='raw_data_anon.csv', 
                        raw_emotions_file='emotions_anon.csv',
                        lower_threshold=100, 
                        upper_threshold=10000,
                        store=False,
                        data_file='tweets.csv',
                        emotions_file='emotions.csv',
                        cascade_size_file='cascade_size.csv'):
        '''
        This function processes the raw data files and loads it into memory

        args: 
            raw_data_file: Path to raw tweet data file
            raw_emotions_file: Path to raw emotions file
            lower_threshold: Minimum size of cascades to include
            upper_threshold: Maximum size of cascades to include
            store: Store preprocessed dataframes to filesystem
            data_file: (ignored if store=False) filename of processed dataframe
            data_file: (ignored if store=False) filename of processed emotions dataframe

        '''  
        self.logger.make_title('Feature engineer')
        df, df_emo, df_cascade_size = self.__load_data(raw_data_file, raw_emotions_file, lower_threshold, upper_threshold)
        df = self.__impute_data(df)
        df = self.__compute_delay(df)
        df = self.__encode_tweet_date(df)
        df = self.__compute_new_tid(df)
        df = self.__compute_depth(df)
        df_cascade_size = self.__process_cascade_size(df_cascade_size)

        if store:
            self.logger.step_start('Saving dataframes to directory...')
            data_file = os.path.join(self.data_dir, data_file)
            emotions_file = os.path.join(self.data_dir, emotions_file)
            cascade_size_file = os.path.join(self.data_dir, cascade_size_file)
            df.to_csv(data_file, header=True, index=False)
            df_emo.to_csv(emotions_file, header=True, index=False)
            df_cascade_size.to_csv(cascade_size_file, header=True, index=False)
            self.logger.step_end()

        return df, df_emo


    def __train_test_split(self, df, df_emo, split_ratio=0.85):
        self.logger.step_start('Split train and test data...')
        ids = df.cascade_id.unique().tolist()
        shuffle(ids)

        split = int(len(ids) * split_ratio)
        train_ids, test_ids = ids[:split], ids[split:]
        train_df, test_df = df[df['cascade_id'].isin(train_ids)].copy(), df[df['cascade_id'].isin(test_ids)].copy()
        train_df_emo, test_df_emo = df_emo[df_emo['cascade_id'].isin(train_ids)].copy(), df_emo[df_emo['cascade_id'].isin(test_ids)].copy()

        self.logger.step_end()
        return train_df, test_df, train_df_emo, test_df_emo

    def __compute_logarithm(self, df):
        self.logger.step_start('Compute logarithm for user data...')
        for cname in ['user_followers', 'user_followees', 'user_engagement', 'user_account_age', 'retweet_delay', 'n_children']:
            df[cname + '_log'] = logp(df[cname].values)

        df = df.drop(['user_followers', 'user_followees', 'user_engagement', 'user_account_age', 'retweet_delay', 'n_children'], axis=1)
        self.logger.step_end()
        return df

    def crop(self, df, threshold):
        self.logger.step_start('Crop dataset...')
        cropped = df[df.root_delay < threshold].copy()
        self.logger.step_end()
        return cropped

    def compute_n_children(self, df):
        self.logger.step_start('Compute n_children...')
        children = df['parent_tid'].value_counts().reset_index()
        children.columns = ['tid', 'n_children']

        df = pd.merge(df, children, on='tid', how='left').fillna({'n_children': 0})
        df['isleaf'] = (df['n_children'] == 0).astype(int)
        self.logger.step_end()
        return df

    def to_grouped(self, df, aggs):
        df = df.copy()
        sizes = df['cascade_id'].value_counts().reset_index()
        sizes.columns = ['cascade_id', 'size']

        breadths = df.groupby(['cascade_id', 'depth'], as_index=False).size().groupby('cascade_id')['size'].max().reset_index()
        breadths.columns = ['cascade_id', 'breadth']

        grouped = df.groupby('cascade_id').agg(aggs).reset_index()

        grouped = pd.merge(grouped, sizes, on='cascade_id')
        grouped = pd.merge(grouped, breadths, on='cascade_id')
        grouped['db_ratio'] = grouped['depth'] / grouped['breadth']

        return grouped

    def __save_experiment_data(self, df, df_emo, df_grouped, threshold, test=False, structureless=False):
        name = '_' + threshold
        name_structureless = name + '_structureless'
        
        if test:
            name += '_test'
            name_structureless += '_test'

        self.logger.step_start('Saving cascades ' + name[1:] + '...')
        for cid in df.cascade_id.unique():	
            small = df[df.cascade_id == cid]
            emo = df_emo[df_emo.cascade_id == cid]

            th_emo = th.Tensor(emo.iloc[:, 1:].values)

            X = th.Tensor(small[self.to_encode].values)
            
            is_leaf = th.Tensor(small['isleaf'].values)
            src, dest = small.new_tid[1:].values, small.new_parent_tid[1:].values
                
            c = Cascade(cid, X, th_emo, src, dest, is_leaf)
            th.save(c, self.graphs_dir + str(cid) + name + '.pt')
            
            if structureless:
                X_structureless = th.Tensor(small[self.to_encode_structureless].values)
                c_structureless = Cascade(cid, X_structureless, th_emo, src, dest, is_leaf)
                th.save(c_structureless, self.graphs_dir + str(cid) + name_structureless + '.pt')

        df_grouped.to_csv(self.grouped_dir + 'grouped' + name + '.csv', header=True, index=False)
        self.logger.step_end()


    def generate_experiment_data(self, 
                                tweets, 
                                emotions,
                                crops_dict, 
                                split_ratio=0.85, 
                                structureless=False,
                                graphs_dir='graphs/',
                                grouped_dir='grouped/'):

        ss = StandardScaler()
        ss_grouped = StandardScaler()
        ss_emo = StandardScaler()

        self.graphs_dir = os.path.join(self.data_dir, graphs_dir)
        self.grouped_dir = os.path.join(self.data_dir, grouped_dir)

        self.__cleanup()

        for k,v in crops_dict.items():
            self.logger.make_title('Generate experiment data for ' + k)
            cropped = self.crop(tweets, v)
            cropped = self.compute_n_children(cropped)
            cropped = self.__compute_logarithm(cropped)

            train_df, test_df, train_df_emo, test_df_emo = self.__train_test_split(cropped, emotions, split_ratio)

            for df, df_emo, test in zip([train_df, test_df], [train_df_emo, test_df_emo], [False, True]):

                self.logger.step_start('Standardize data...')
                grouped = self.to_grouped(df, self.to_group)

                if not test:
                    df[self.to_standardize] = ss.fit_transform(df[self.to_standardize].values)
                    df_emo.iloc[:, 1:] = ss_emo.fit_transform(df_emo.iloc[:, 1:].values)
                    grouped.iloc[:, 1:] = ss_grouped.fit_transform(grouped.iloc[:, 1:].values)
                else:
                    df[self.to_standardize] = ss.transform(df[self.to_standardize].values)
                    df_emo.iloc[:, 1:] = ss_emo.transform(df_emo.iloc[:, 1:].values)
                    grouped.iloc[:, 1:] = ss_grouped.transform(grouped.iloc[:, 1:])

                grouped = pd.merge(grouped, df_emo)
                self.logger.step_end()

                self.__save_experiment_data(df, df_emo, grouped, k, test, structureless)
                
    
    def __cleanup(self):
        def confirm():
            answer = ""
            while answer not in ['y', 'n']:
                answer = input('Are you sure you want to continue, this will delete all data in ' + self.graphs_dir + ' and ' + self.grouped_dir + ' ? [Y/N] ').lower()
            return answer == 'y'

        for dir in [self.graphs_dir, self.grouped_dir]:
            Path(dir).mkdir(parents=True, exist_ok=True)

        if not confirm():
            sys.exit(0)

        for dir in [self.graphs_dir, self.grouped_dir]:
            for filename in os.listdir(dir):
                file_path = os.path.join(dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__":
    try: os.chdir(os.path.dirname(sys.argv[0]))
    except: pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest = 'data_dir', default = '../data/', type = str)
    parser.add_argument('--verbose', dest='verbose', default=1, type=int)
    parser.add_argument('--feature_engineer', action='store_true', default=False, dest='feature_engineer')
    parser.add_argument('--data_file', dest = 'data_file', default = 'tweets.csv', type = str)
    parser.add_argument('--emotions_file', dest = 'emotions_file', default = 'emotions.csv', type = str)
    parser.add_argument('--lower', dest='lower', default=100, type=int)
    parser.add_argument('--upper', dest='upper', default=10000, type=int)
    parser.add_argument('--structureless', action='store_true', default = False)
    parser.add_argument('--split', dest='split', default=0.85, type=float)

    args = parser.parse_args()

    crops = {
        '15_mins': 60 * 15,
        '30_mins': 60 * 30,
        '1_hour': 60 * 60,
        '3_hour': 60 * 60 * 3,
        '24_hour': 60 * 60 * 24
    }

    crops_min = {
        '1_hour': 60 * 60,
        '3_hour': 60 * 60 * 3
    }

    data_dir = args.data_dir
    raw_data_file = args.data_dir + 'raw_data_anon.csv'
    raw_emotions_file = args.data_dir + 'emotions_anon.csv'
    data_file = args.data_dir + args.data_file
    emotions_file = args.data_dir + args.emotions_file

    preprocessor = Preprocessor(data_dir, args.verbose)

    if args.feature_engineer:
        df, df_emo = preprocessor.feature_engineer(raw_data_file, raw_emotions_file, args.lower, args.upper, True, data_file, emotions_file)
    else:
        df = pd.read_csv(data_file)
        df_emo = pd.read_csv(emotions_file)
        
    preprocessor.generate_experiment_data(df, df_emo, crops, args.split, args.structureless, 'graphs/', 'grouped/')
