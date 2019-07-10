import json
import re
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

top50 = [
    'trees',
    'buildapc',
    'gaming',
    'fantasyfootball',
    'techsupport',
    'atheism',
    'circlejerk',
    'askscience',
    'Fitness',
    'SteamGameSwap',
    'tipofmytongue',
    'NoFap',
    'politics',
    'Music',
    'IAmA',
    'atheismbot',
    'malefashionadvice',
    'funny',
    'explainlikeimfive',
    'r4r',
    'movies',
    'relationships',
    'HITsWorthTurkingFor',
    'RedditRescueForce',
    'relationship_advice',
    'depression',
    'loseit',
    'nosleep',
    'picrequests',
    'keto',
    'TwoXChromosomes',
    'Jokes',
    'BabyBumps',
    'guns',
    'amiugly',
    'self',
    'Drugs',
    'seduction',
    'learnprogramming',
    'Guitar',
    'Steam',
    'offmychest',
    'applehelp',
    'AskWomen',
    'Christianity',
    'bicycling',
    'motorcycles',
    'Poetry',
    'LucidDreaming',
    'iphone',
]

subreddit_to_label = {name : i for i, name in enumerate(top50)}

postfile = 'top50-posts.txt'
commentfile = 'top50-comments.txt'

post_data = []
post_ids = []
post_labels = []

def read_data():
    # regex used to filter out urls
    url_re = r'https?://[^\s)]*'
    with open(postfile, 'r') as f:
        for i, l in enumerate(f):
            if i % 10000 == 0:
                print('Processed %d items from %s' % (i, postfile))
            obj = json.loads(l)
            text = obj['title']
            text = re.sub(url_re, '', text)
            post_data.append(text)
            post_ids.append(obj['name'])
            post_labels.append(subreddit_to_label[obj['subreddit']])

def make_features():
    # create bag-of-words feature
    count_vect = CountVectorizer(max_df=0.99, min_df=0.00005)
    counts = count_vect.fit_transform(post_data)
    # counts is a scipy sparse matrix of shape (num_posts, vocab_size)
    return counts

def write_features(counts):
    # write bag-of-words feature to file 'post-feat.txt'
    with open('post-feat.txt', 'w') as f:
        # first line is the shape
        f.write('%d\t%d\n' % (counts.shape[0], counts.shape[1]))
        counts = counts.tocoo()
        row = counts.row
        col = counts.col
        val = counts.data
        for r, c, v in zip(row, col, val):
            f.write('%d\t%d\t%d\n' % (r, c, v))

def write_post_ids():
    # write ids to file 'post-ids.txt'
    with open('post-ids.txt', 'w') as f:
        for pid in post_ids:
            f.write('%s\n' % pid)

def write_post_labels():
    # write labels to file 'post-ids.txt'
    with open('post-labels.txt', 'w') as f:
        for lbl in post_labels:
            f.write('%d\n' % lbl)

def generate_graph():
    # a map from post name to its id
    post2ids = {p : i for i, p in enumerate(post_ids)}
    # read author-post edge from comment data
    edges = []
    uid = len(post_ids)
    author2ids = dict()
    with open(commentfile, 'r') as f:
        for i, l in enumerate(f):
            if i % 10000 == 0:
                print('Processed %d items from %s' % (i, commentfile))
            obj = json.loads(l)
            if not obj['author'] in author2ids:
                author2ids[obj['author']] = uid
                uid += 1
            edges.append((author2ids[obj['author']], post2ids[obj['link_id']]))
            edges.append((post2ids[obj['link_id']], author2ids[obj['author']]))
    # generate a sparse matrix using author-post pair
    row, col = zip(*edges)
    n = max(max(row), max(col)) + 1
    row = np.array(row)
    col = np.array(col)
    data = np.ones((len(row),))
    adj = sp.coo_matrix((data, (row, col)), shape=(n, n))

    # get second-order graph and slice our subgraph among posts
    adj2 = adj.dot(adj)
    adj2 = adj2[0:len(post_ids), 0:len(post_ids)]
    return adj2

def write_graph(adj):
    # write graph structure to two file 'u.npy' and 'v.npy'
    adj = adj.tocoo()
    adj.row.dump('u.npy')
    adj.col.dump('v.npy')

if __name__ == '__main__':
    print('Load raw data')
    read_data()
    print('Create features')
    feat = make_features()
    print('Feature shape:', feat.shape)
    print('Create graph')
    g = generate_graph()
    print('#Nodes:', g.shape[0])
    print('#Edges:', len(g.data))
    print('Write features')
    write_features(feat)
    print('Write post ids')
    write_post_ids()
    print('Write post labels')
    write_post_labels()
    print('Write graph')
    write_graph(g)
