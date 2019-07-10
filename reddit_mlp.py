import numpy as np
import scipy.sparse as sp

from mxnet import gluon, autograd
from mxnet.gluon import nn
import mxnet.ndarray as nd

def load_data():
    print('Features...')
    n, m = 0, 0
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
    feat = feat.todense()
    feat = feat / feat.sum(1)

    print('Labels...')
    label = []
    with open('./reddit-top50/post-labels.txt', 'r') as f:
        for i, l in enumerate(f):
            label.append(int(l.strip()))
    label = np.array(label, dtype=np.int64)

    print('Making training/testing sets')
    train_feat = feat[0:n//2, :]
    test_feat = feat[-1001:-1, :]
    train_label = label[0:n//2]
    test_label = label[-1001:-1]
    return train_feat, train_label, test_feat, test_label

def evaluate(model, feats, labels):
    logits = model(feats)
    indices = logits.argmax(axis=1)
    correct = (indices == labels).sum()
    return (correct / labels.shape[0]).asscalar()



class MLP(gluon.Block):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Dense(64)
        self.fc2 = nn.Dense(50)

    def forward(self, feats):
        h = self.fc1(feats)
        h = nd.relu(h)
        h = self.fc2(h)
        return h
    
if __name__ == '__main__':
    
    train_feat, train_label, test_feat, test_label = load_data()
    print(train_feat.shape, train_label.shape)
    print(test_feat.shape, test_label.shape)

    train_feat = nd.array(train_feat)
    train_label = nd.array(train_label)
    test_feat = nd.array(test_feat)
    test_label = nd.array(test_label)

    batch_size = 1024
    dataset = gluon.data.dataset.ArrayDataset(train_feat, train_label)
    dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)

    model = MLP()
    model.initialize()
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.1, 'wd': 5e-4})
    loss_fcn = gluon.loss.SoftmaxCELoss()

    for epoch in range(200):
        for i, (feat, lbl) in enumerate(dataloader):
            with autograd.record():
                logits = model(feat)
                loss = loss_fcn(logits, lbl).sum() / batch_size

            loss.backward()
            trainer.step(batch_size=1)

        train_acc = evaluate(model, feat, lbl)
        test_acc = evaluate(model, test_feat, test_label)
        print('Epoch %d, Loss %f, Train acc %f, Test acc %f' % (epoch, loss.asscalar(), train_acc, test_acc))
