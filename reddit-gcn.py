import dgl
import dgl.function as fn

import numpy as np
import scipy.sparse as sp

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
import mxnet.ndarray as nd


def load_data():
    print('Features...')
    n = 0
    m = 0
    row = []
    col = []
    val = []
    with open('./reddit-top50/post-feat.txt', 'r') as f:
        for i, l in enumerate(f):
            parts = l.strip().split('\t')
            if i == 0:
                n = int(parts[0])
                m = int(parts[1])
            else:
                row.append(int(parts[0]))
                col.append(int(parts[1]))
                val.append(int(parts[2]))
    feat = sp.coo_matrix((val, (row, col)), shape=(n, m), dtype=np.float32)
    feat = nd.sparse.csr_matrix(feat.tocsr(), shape=(n, m), dtype=np.float32)
    feat = feat / feat.sum(1).reshape(-1,1)

    print('Generating graph')
    u = np.load('./reddit-top50/u.npy')
    v = np.load('./reddit-top50/v.npy')
    # sparsify the graph to keep only 10% of the edges
    keep_idx = np.random.choice(len(u), len(u) // 10, replace=False)
    u = u[keep_idx]
    v = v[keep_idx]
    adj = sp.coo_matrix((np.ones((len(u),)), (u, v)), shape=(n, n))
    # add self-loop
    adj += sp.eye(n, n)
    g = dgl.DGLGraph(adj)
    print('#Nodes:', g.number_of_nodes())
    print('#Edges:', g.number_of_edges())

    print('Labels...')
    label = []
    with open('./reddit-top50/post-labels.txt', 'r') as f:
        for i, l in enumerate(f):
            label.append(int(l.strip()))
    label = np.array(label, dtype=np.int64)

    print('Making training/testing masks')
    train_mask = np.arange(0, n//2)
    test_mask = np.arange(n - 1000, n)
    return g, feat, label, train_mask, test_mask

def evaluate(model, g, feats, labels, mask):
    logits = model(g, feats)
    logits = logits[mask]
    labels = labels[mask]
    indices = logits.argmax(axis=1)
    accuracy = (indices == labels).sum() / labels.shape[0]
    return accuracy.asscalar()

g, feat, label, train_mask, test_mask = load_data()

ctx = mx.gpu(0)
feat = nd.array(feat, ctx=ctx)
label = nd.array(label, ctx=ctx)
train_mask = nd.array(train_mask, ctx=ctx)
test_mask = nd.array(test_mask, ctx=ctx)
n_train_samples = train_mask.shape[0]

# calculate normalization
#degs = g.in_degrees().astype('float32').asnumpy()
#norm = np.power(degs, -0.5).reshape(-1, 1)
#norm[np.isinf(norm)] = 0.
#norm = nd.array(norm, ctx=ctx)
#g.ndata['norm'] = norm

class GraphConv(gluon.Block):
    def __init__(self, n_in, n_out):
        super(GraphConv, self).__init__()
        self.fc = nn.Dense(n_out)
        self.fc2 = nn.Dense(n_out)

    def forward(self, g, feats):
        h = self.fc(feats)
        g.ndata['h'] = h #* g.ndata['norm']
        g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'))
        hh = g.ndata.pop('h') #* g.ndata['norm']
        h = nd.concat(h, hh, dim=1)
        return self.fc2(h)

class GCN(gluon.Block):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(feat.shape[1], 64)
        self.gc2 = GraphConv(64, 50)

    def forward(self, g, feats):
        h = self.gc1(g, feats)
        h = nd.relu(h)
        h = self.gc2(g, h)
        return h

model = GCN()
model.initialize(ctx=ctx)
trainer = gluon.Trainer(model.collect_params(), 'adam',
                        {'learning_rate': 0.01, 'wd': 5e-4})
loss_fcn = gluon.loss.SoftmaxCELoss()

feat = feat.as_in_context(ctx)
label = label.as_in_context(ctx)

for epoch in range(200):
    with autograd.record():
        logits = model(g, feat)
        loss = loss_fcn(logits[train_mask], label[train_mask]).sum() / n_train_samples

    loss.backward()
    trainer.step(batch_size=1)

    train_acc = evaluate(model, g, feat, label, train_mask)
    test_acc = evaluate(model, g, feat, label, test_mask)
    print('Epoch %d, Loss %f, Train acc %f, Test acc %f' % (epoch, loss.asscalar(), train_acc, test_acc))
