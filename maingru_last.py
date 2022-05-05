# encoding=utf-8
import os
import dataloader_last
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from protein_gru_last import Model
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
loader = dataloader_last.DataMaster()

batch_size = 64
epoch_num = 50

keep_pro = 0.95
init_learning_rate = 0.001


decay_rate = 0.96
decay_steps = loader.training_size / batch_size
model = Model(init_learning_rate, decay_steps, decay_rate)

def validataion(test_X,test_E,test_Y):
    step_size = 300
    outputs = []
    logits_pred = []
    for i in range(0, len(loader.test_Y), step_size):
        batch_X = test_X[i:i + step_size]
        batch_E = test_E[i:i + step_size]
        batch_Y = test_Y[i:i + step_size]

        output, y_logit = sess.run([model.prediction_cnn, model.logits_pred],
                                   feed_dict={model.x: batch_X, model.e: batch_E, model.y: batch_Y,
                                              model.dropout_keep_prob: 1.0})
        outputs.append(output)
        logits_pred.append(y_logit)
    y_pred = np.concatenate(outputs, axis=0)
    logits_pred = np.concatenate(logits_pred, axis=0)

    report = metrics.classification_report(test_Y, y_pred,
                                          target_names=['Trivial', 'Essential'])

    print(report)
    return test_Y, y_pred


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    datasets = np.load('../data')
    dataembs = np.load('../data')
    datalabels = np.load('../data')

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc = []
    training_size = int(0.8 * len(datasets))

    sss = StratifiedShuffleSplit(n_splits=10,test_size=0.2, random_state=2020)
    for train_index, test_index in sss.split(datasets,datalabels):
        train_X, test_X = datasets[train_index], datasets[test_index]
        train_Y, test_Y = datalabels[train_index], datalabels[test_index]
        train_E, test_E = dataembs[train_index], dataembs[test_index]
        for epoch in range(epoch_num):
            for iter, idx in enumerate(range(0, training_size, batch_size)):
                batch_X = train_X[idx:idx + batch_size]
                batch_E = train_E[idx:idx + batch_size]
                batch_Y = train_Y[idx:idx + batch_size]
                batch_loss, y_pred, y_logits, accuracy, _ = sess.run(
                    [model.loss_cnn, model.prediction_cnn, model.logits_pred, model.accuracy, model.optimizer_cnn],
                    feed_dict={model.x: batch_X, model.e: batch_E, model.y: batch_Y,
                               model.dropout_keep_prob: keep_pro})
                if iter % 20 == 0:
                    print("=====epoch:%d iter:%d=====" % (epoch + 1, iter + 1))
                    print('batch_loss %.3f' % batch_loss)
                    print("accuracy %.6f" % metrics.accuracy_score(batch_Y, y_pred))
                    print("Precision %.6f" % metrics.precision_score(batch_Y, y_pred))
                    print("Recall %.6f" % metrics.recall_score(batch_Y, y_pred))
                    print("f1_score %.6f" % metrics.f1_score(batch_Y, y_pred))
                    fpr, tpr, threshold = metrics.roc_curve(batch_Y, y_logits)
                    print("auc_socre %.6f" % metrics.auc(fpr, tpr))
    

        y_test, y_pred = validataion(test_X,test_E,test_Y)

        as1 = metrics.accuracy_score(y_test, y_pred)
        accuracy_scores.append(as1)
        ps = metrics.precision_score(y_test, y_pred)
        precision_scores.append(ps)
        rs = metrics.recall_score(y_test, y_pred)
        recall_scores.append(rs)
        fs = metrics.f1_score(y_test, y_pred)
        f1_scores.append(fs)
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr, tpr)
        roc_auc.append(auc)

    print('Accuracy', np.mean(accuracy_scores))
    print(accuracy_scores)
    print('Precision', np.mean(precision_scores))
    print(precision_scores)
    print('Recall', np.mean(recall_scores))
    print(recall_scores)
    print('F1-measure', np.mean(f1_scores))
    print(f1_scores)
    print('auc', np.mean(roc_auc))
    print(roc_auc)

