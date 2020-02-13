import sys
import os
import warnings
import numpy as np
from random import shuffle
import pandas as pd
import torch as th
from tqdm import tqdm
import networkx as nx
from collections import Counter
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.append('../code/')
from klasses import Cascade

try: os.chdir(os.path.dirname(sys.argv[0]))
except: pass

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def save_cascades(df, df_emo, to_encode, name_append=''):
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
    
    # loop through cascades id 
    for cid in tqdm(df.cascade_id.unique()):	
        # selects tweets and emotions pertaining to cascade
        small = df[df.cascade_id == cid].copy().reset_index()
        emo = df_emo[df_emo.cascade_id == cid]

        # if there are no emotions for the cascade, skip it
        if emo.empty:
            continue

        th_emo = th.Tensor(emo.iloc[:, 1:].values)
        th_ver = th.tensor([[small.veracity[0]]])
        X = th.Tensor(small[to_encode].values)
        is_leaf = th.Tensor(small.is_leaf.values)
        src, dest = small.new_tid[1:].values, small.new_parent_tid[1:].values

        c = Cascade(cid, X, th_ver, th_emo, src, dest, is_leaf)

        th.save(c, graphs_dir + str(cid) + name_append + '.pt')


def get_parents(df):
    # returns df with number of children per tweet
    df = df.copy()
    parents = df.groupby(['cascade_id', 'new_parent_tid']).agg({'tid': 'count'}).reset_index()
    parents = parents.rename(columns={'tid': 'n_children', 'new_parent_tid': 'new_tid'})
    return parents


def crop(tweets, mode=False, threshold=False):
    """
    Returns df cropped at a certain threshold
    In :
        - tweets: df with tweets
        - mode: 
            *time (keep tweets before time XX has elapsed from first tweet)
            * numer (keep tweets until tweets number XX)
        - threshold: threshold of tweets or time ; same variable for both
    Out: cropped df
    """

    cropped = tweets.copy()

    if threshold: 
        if mode == 'time':
            cropped = cropped[cropped.root_delay < threshold]
        elif mode == 'number':
            cropped = cropped[cropped.new_tid < threshold]

    cropped = pd.merge(cropped, get_parents(cropped), on=['cascade_id', 'new_tid'], how='left')
    cropped = cropped.fillna({'n_children': 0})
    cropped['n_children_log'] = logp(cropped.n_children)
    cropped['is_leaf'] = (cropped.n_children == 0).astype(int)
    return cropped


def get_new_tid(df):
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
    df.loc[df.new_parent_tid.isna(), 'new_parent_tid'] = -1
    return df['new_tid'], df['new_parent_tid']


def get_depths(df):
    """
    get node depth for ALL CASCADES
    In : df with all tweets 
    Out : list with depth of all nodes for every cascade
    """

    print('getting_depths')
    
    all_depths = []
    
    for cid in tqdm(df.cascade_id.unique()):
        small = df[df.cascade_id == cid].copy()
        all_depths.append(get_nodes_depths(small))
    
    return list(chain(*all_depths))        
    
    
def get_nodes_depths(small):
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


def get_cascade_statistics(small):
    # get depth, breadth and depth to breadth ratio for a single cascade
    
    depths = small.depth
    cascade_depth = max(depths)
    cascade_breadth = max(Counter(depths).values())
    db_ratio = cascade_depth / cascade_breadth

    return cascade_depth, cascade_breadth, db_ratio


def to_grouped(df, to_group):
    """
    Create df for standard feature classifiers with 1 cascade per row
    In: 
        - df : all tweets
        - to_group : variables to average per cascade
    Out: df with averaged node vars and aggregate statistics per cascade
    """
    
    aggs = {k: 'mean' for k in to_group}
    aggs['n_children_log'] = 'max' 
        
    grouped = df.groupby(['cascade_id', 'veracity']).agg(aggs).reset_index()

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


logp = lambda x: np.log (x+1)
vprint = lambda *x : print(*x) if verbose else None

source_dir = '../data/'
dest_dir = '../data/'
graphs_dir = dest_dir + 'graphs/'
tweets_file = source_dir + 'tweets.csv'
emotions_file = source_dir + 'emotions.csv'

to_encode = ['user_followers_log',
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

to_encode_structureless = ['user_followers_log',
                           'user_followees_log',
                           'user_account_age_log',
                           'user_engagement_log',
                           'retweet_delay_log',
                           'user_verified',
                           'hour_cos',
                           'hour_sin',
                           'wd_cos',
                           'wd_sin']


to_group = ['user_followers_log',
            'user_followees_log',
            'user_account_age_log',
            'user_engagement_log',
            'retweet_delay_log']

to_standardize = ['user_followers_log', 
                  'user_followees_log',
                  'user_engagement_log', 
                  'user_account_age_log',
                  'retweet_delay_log',
                  'depth']

lower_threshold = 100
upper_threshold = 10000
verbose = True


crop_thresh_time = {
    'half_hour': 60 * 30,
    '1_hour': 60. * 60,
    '2_hour': 60. * 60 * 2,
    '3_hour': 60. * 60 * 3, 
    '6_hour': 60. * 60 * 6,
    '12_hour': 60. * 60 * 12,
    '24_hour': 60. * 60 * 24}

    
crop_thresh_number = {
    '200_tweets': 200,
    '500_tweets': 500,
    '1000_tweets': 1000,
    '2000_tweets': 2000}

ss = StandardScaler()
si = SimpleImputer(strategy='median')

split_ratio = 0.85

tweets = pd.read_csv(tweets_file)
print(tweets.shape)
tweets.head()

#tweets.drop(to_drop, axis=1, inplace=True)

tweets['veracity'] = tweets['veracity'].astype(float)

if lower_threshold != -1 and upper_threshold != -1:
    # get cascade ids of cascades whose size is between lower and upper thesholds<
    cascade_ids = [k for k, v in Counter(tweets.cascade_id).items() if v >= lower_threshold and v < upper_threshold]
    vprint('cascade ids retrieved')
    sieve = pd.DataFrame({'cascade_id': cascade_ids})
    vprint('started merging')
    tweets = pd.merge(sieve, tweets, how='left', on='cascade_id')
    vprint('finished merging')
    
vprint('There are', tweets.shape[0], 'rows after filtering for cascade size')


tweets = tweets.sort_values(['cascade_id', 'datetime']).reset_index(drop=True)
tweets['new_tid'], tweets['new_parent_tid'] = get_new_tid(tweets)
tweets['depth'] = get_depths(tweets)

emos = pd.read_csv(emotions_file)

# ids : all ids of cascades that have emo AND are in size range 
IDs = list(set(tweets.cascade_id).intersection(emos.cascade_id))

shuffle(IDs)
split = int(len(IDs) * split_ratio)
train_ids, test_ids = pd.DataFrame({'cascade_id': IDs[:split]}), pd.DataFrame({'cascade_id': IDs[split:]})

tweets_train = pd.merge(tweets, train_ids, how='inner').reset_index(drop=True)
tweets_test = pd.merge(tweets, test_ids, how='inner').reset_index(drop=True)
emo_train = pd.merge(emos, train_ids, how='inner').reset_index(drop=True)
emo_test = pd.merge(emos, test_ids, how='inner').reset_index(drop=True)

tweets_train[['user_followers', 'user_followees', 'user_account_age']] = si.fit_transform(tweets_train[['user_followers', 'user_followees', 'user_account_age']].values)
tweets_test[['user_followers', 'user_followees', 'user_account_age']] = si.transform(tweets_test[['user_followers', 'user_followees', 'user_account_age']].values)

# get log of vars
for cname in ['user_followers', 'user_followees', 'user_engagement', 'user_account_age', 'retweet_delay']:
    tweets_train[cname + '_log'] = logp(tweets_train[cname].values)
    tweets_test[cname + '_log'] = logp(tweets_test[cname].values)
    
tweets_train[to_standardize] = ss.fit_transform(tweets_train[to_standardize].values)
tweets_test[to_standardize] = ss.transform(tweets_test[to_standardize].values)

emo_train.iloc[:, 1:] = ss.fit_transform(emo_train.iloc[:, 1:].values)
emo_test.iloc[:, 1:] = ss.transform(emo_test.iloc[:, 1:].values)

tweets_train.to_csv('../data/tweets_train.csv')
tweets_test.to_csv('../data/tweets_test.csv')
emo_train.to_csv('../data/emo_train.csv')
emo_test.to_csv('../data/emo_test.csv')

# scaler for grouped data frames
ss_grouped = StandardScaler()


# MAIN CASCADE LOOP

# loop through train and test dfs
for (df_tweets, df_emo, post) in zip(*[(tweets_train.copy(), tweets_test.copy()), (emo_train, emo_test), ('', '_test')]):   
    not_cropped = crop(df_tweets)   
    if post == '':
        not_cropped['n_children_log'] = ss.fit_transform(not_cropped.n_children.values.reshape(-1, 1))
    else:
        not_cropped['n_children_log'] = ss.transform(not_cropped.n_children.values.reshape(-1, 1))
    
    grouped = to_grouped(not_cropped, to_group)
    
    if post == '':
        grouped.iloc[:, 2:] = ss_grouped.fit_transform(grouped.iloc[:, 2:])
    else:
        grouped.iloc[:, 2:] = ss_grouped.transform(grouped.iloc[:, 2:])

    pd.merge(grouped, df_emo, on='cascade_id').to_csv(dest_dir + 'grouped' + post + '.csv', header=True, index=False)

    save_cascades(not_cropped, df_emo, to_encode, post)
    save_cascades(not_cropped, df_emo, to_encode_structureless, '_structureless' + post)

# LOOP FOR CROPPED CASCADES

# loop through cropping modes (train and test)
for (crop_dict, mode) in [(crop_thresh_time, 'time'), (crop_thresh_number, 'number')]:
    # loop through cropping thresholds in cropping mode
    for k, v in crop_dict.items():  
        # loop through train and test
        for (df_tweets, df_emo, post) in zip(*[(tweets_train.copy(), tweets_test.copy()), (emo_train, emo_test), ('', '_test')]):
            cropped = crop(df_tweets, mode, v)
            if post == '':
                cropped['n_children_log'] = ss.fit_transform(cropped.n_children.values.reshape(-1, 1))
            else:
                cropped['n_children_log'] = ss.transform(cropped.n_children.values.reshape(-1, 1))

            grouped = to_grouped(cropped, to_group)            
            if post == '':
                grouped.iloc[:, 2:] = ss_grouped.fit_transform(grouped.iloc[:, 2:])
            else:
                grouped.iloc[:, 2:] = ss_grouped.transform(grouped.iloc[:, 2:])
            
            pd.merge(grouped, df_emo, on='cascade_id').to_csv(dest_dir + 'grouped_' + k + post + '.csv', header=True, index=False)

            save_cascades(cropped, df_emo, to_encode, '_' + k + post)
