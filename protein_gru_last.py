# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np


embedding_fn_size_1 = 512
embedding_fn_size_2 = 312
embedding_fn_size_3 = 128

time_steps = 12
channel_size = 3
channel_size2 =8

embedding_size = 64
loc_size = 11

embedding_fn_size = 64
embedding_loc_size = 512

filter_num = 8
filter_sizes = [1, 3, 5]
gru_cell_size = 3
threshold = 0.5

class Model(object):
    def __init__(self, init_learning_rate, decay_steps, decay_rate):
        weights = {
            'wc1': tf.Variable(tf.truncated_normal([filter_sizes[0], channel_size, filter_num], stddev=0.1)),
            'wc2': tf.Variable(tf.truncated_normal([filter_sizes[1], channel_size2, filter_num], stddev=0.1)),
            'wc3': tf.Variable(tf.truncated_normal([filter_sizes[2], channel_size, filter_num], stddev=0.1))
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc2': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc3': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1))
        }


        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate,
                                                   staircase=True)


        self.x = tf.placeholder(tf.float32, [None, channel_size, time_steps])
        x_emb = tf.transpose(self.x, [0, 2, 1])
        self.e = tf.placeholder(tf.float32, [None, embedding_size])
        self.l = tf.placeholder(tf.float32, [None, loc_size])
        self.y = tf.placeholder(tf.int32, [None, 1])

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        ones = tf.ones_like(self.y)
        zeros = tf.zeros_like(self.y)

        with tf.name_scope("FN_Part"):
            output_e = tf.layers.dense(self.e, embedding_fn_size, activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))


        with tf.name_scope("CNN_Part"):
            conv1 = self.conv1d(x_emb, weights['wc1'], biases['bc1']) #
            print('conv1:', conv1)
            conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
            print('conv2:', conv2)
            x_convs = self.conv2d(conv2, weights['wc2'], biases['bc2'])
            print('x_convs:', x_convs)

            pooled = tf.reduce_max(x_convs, axis=1)
            print('pooled:', pooled)
            pooled = tf.reshape(pooled, shape=[-1,1,8])
            print('pooled:', pooled)

            output_gru = self.BidirectionalgruEncoder(pooled)
            print('after BidirectionalgruEncoder---1: ', output_gru)

            output_gru = tf.reshape(output_gru, [-1, 2 * gru_cell_size])
            print('after BidirectionalgruEncoder---2: ', output_gru)
            print(output_gru)

        with tf.name_scope("Output_Part"):

            concate_v = tf.concat([output_e, output_gru], axis=1)

            fc_1 = tf.layers.dense(concate_v, embedding_fn_size_1, activation=tf.nn.relu,       # 输入
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

            weight_last = tf.Variable(tf.truncated_normal([embedding_fn_size_1, 1]) * np.sqrt(2. / (2 * filter_num)))
            bias_last = tf.Variable(tf.truncated_normal([1], stddev=0.1))
            fc_1 = tf.nn.dropout(fc_1, self.dropout_keep_prob)
            logits_cnn = tf.matmul(fc_1, weight_last) + bias_last

            self.loss_cnn = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y, tf.float32), logits=logits_cnn))
            self.optimizer_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_cnn,
                                                                                              global_step=global_step)
            self.logits_pred = tf.nn.sigmoid(logits_cnn)
            self.prediction_cnn = tf.cast(tf.where(tf.greater(self.logits_pred, threshold), ones, zeros), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction_cnn, self.y), tf.float32))


    def conv1d(sef, x, W, b):
        x = tf.reshape(x, shape=[-1, time_steps, channel_size])
        x = tf.nn.conv1d(x, W, 1, padding='SAME')
        x = tf.nn.bias_add(x, b)
        h = tf.nn.relu(x)
        return h

    def conv2d(sef, x, W, b):
        x = tf.reshape(x, shape=[-1, time_steps, channel_size2])
        x = tf.nn.conv1d(x, W, 1, padding='SAME')
        x = tf.nn.bias_add(x, b)
        h = tf.nn.relu(x)
        return h

    def conv3d(sef, x, W, b):
        x = tf.reshape(x, shape=[-1, time_steps, channel_size3])
        x = tf.nn.conv1d(x, W, 1, padding='SAME')
        x = tf.nn.bias_add(x, b)
        h = tf.nn.relu(x)
        return h

    def multi_conv(self, x, weights, biases):
        conv1 = self.conv1d(x, weights['wc1'], biases['bc1'])
        conv2 = self.conv1d(x, weights['wc2'], biases['bc2'])
        convs = tf.concat([conv1, conv2], 1)
        return convs

    def BidirectionalgruEncoder(self, inputs, name='Bidirectionalgru'):
        print(inputs)
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(gru_cell_size)
            GRU_cell_bw = rnn.GRUCell(gru_cell_size)
            GRU_cell_fw = rnn.DropoutWrapper(GRU_cell_fw, output_keep_prob=self.dropout_keep_prob)
            GRU_cell_bw = rnn.DropoutWrapper(GRU_cell_bw, output_keep_prob=self.dropout_keep_prob)
            ((fw_outputs, bw_outputs), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                               cell_bw=GRU_cell_bw,
                                                                                               inputs=inputs,
                                                                                               sequence_length=self.length(
                                                                                                   inputs),
                                                                                            dtype=tf.float32)

            outputs = tf.concat((fw_state, bw_state), 1)
            return outputs

    def length(self, sequences):
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        self.seq_len = tf.cast(seq_len, tf.int32)
        return self.seq_len

