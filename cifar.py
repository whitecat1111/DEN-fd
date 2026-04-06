'''在Split CIFAR-100数据集上测试不同模型性能'''
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import DEN
import DEN_fd

from keras.datasets import cifar100


# ======================================================
# 1. Split CIFAR-100 Loader
# ======================================================
class SplitCIFAR100Data(object):
    def __init__(self):
        (trainX, trainY), (testX, testY) = cifar100.load_data(label_mode='fine')
        self.trainX = trainX.reshape(-1, 3072).astype(np.float32) / 255.
        self.testX = testX.reshape(-1, 3072).astype(np.float32) / 255.
        self.trainY = trainY.flatten()
        self.testY = testY.flatten()

    def _one_hot(self, y, num_classes=10):
        return np.eye(num_classes)[y]

    def get_task_data(self, task_id, classes_per_task=10):
        start_class = (task_id - 1) * classes_per_task
        end_class = start_class + classes_per_task

        train_idx = np.where((self.trainY >= start_class) & (self.trainY < end_class))[0]
        test_idx = np.where((self.testY >= start_class) & (self.testY < end_class))[0]

        tr_X = self.trainX[train_idx]
        te_X = self.testX[test_idx]

        tr_Y_raw = self.trainY[train_idx] - start_class
        te_Y_raw = self.testY[test_idx] - start_class

        val_size = 1000
        indices = np.random.permutation(len(tr_X))
        tr_X, tr_Y_raw = tr_X[indices], tr_Y_raw[indices]

        v_X = tr_X[:val_size]
        v_Y_raw = tr_Y_raw[:val_size]
        tr_X = tr_X[val_size:]
        tr_Y_raw = tr_Y_raw[val_size:]

        tr_Y = self._one_hot(tr_Y_raw, classes_per_task)
        v_Y = self._one_hot(v_Y_raw, classes_per_task)
        te_Y = self._one_hot(te_Y_raw, classes_per_task)

        return tr_X, tr_Y, v_X, v_Y, te_X, te_Y


# ======================================================
# 2. FLAGS & Helpers
# ======================================================
np.random.seed(1004)

try:
    tf.app.flags.DEFINE_string('f', '', 'kernel')
except:
    pass
flags = tf.app.flags

flags.DEFINE_integer("max_iter", 4000, "")
flags.DEFINE_float("lr", 0.001, "")
flags.DEFINE_integer("batch_size", 256, "")
flags.DEFINE_string("dims", "3072,1000,500,10", "")
flags.DEFINE_integer("n_classes", 10, "")
flags.DEFINE_float("l1_lambda", 0.00001, "")
flags.DEFINE_float("l2_lambda", 0.0001, "")
flags.DEFINE_float("gl_lambda", 0.001, "")
flags.DEFINE_float("regular_lambda", 0.5, "")
flags.DEFINE_integer("ex_k", 10, "")
flags.DEFINE_float('loss_thr', 0.01, "")
flags.DEFINE_float('spl_thr', 0.05, "")

FLAGS = flags.FLAGS
FLAGS.dims = list(map(int, FLAGS.dims.split(',')))


def get_hidden_capacity(params, n_layers, task_id):
    total_units = 0
    for i in range(1, n_layers):
        w_shape = params['layer%d/weight:0' % i].shape
        total_units += w_shape[1]
    return total_units


def get_dnn_hidden_capacity(dims):
    return sum(dims[1:-1])


# ======================================================
# 3. Vanilla DNN
# ======================================================
class VanillaDNN:
    def __init__(self, dims, lr):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, dims[0]])
            self.y = tf.placeholder(tf.float32, [None, dims[-1]])
            h = self.x
            for i in range(1, len(dims)):
                W = tf.Variable(tf.random_normal([dims[i - 1], dims[i]], stddev=0.01))
                b = tf.Variable(tf.zeros([dims[i]]))
                h = tf.matmul(h, W) + b
                if i != len(dims) - 1:
                    h = tf.nn.relu(h)
            self.logits = h
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits))
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
            correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))
            self.init = tf.global_variables_initializer()

    def train_epoch(self, sess, X, Y, batch_size):
        idx = np.random.permutation(len(X))
        X, Y = X[idx], Y[idx]
        losses = []
        for i in range(0, len(X), batch_size):
            bx, by = X[i:i + batch_size], Y[i:i + batch_size]
            _, loss = sess.run([self.train_op, self.loss], feed_dict={self.x: bx, self.y: by})
            losses.append(loss)
        return np.mean(losses)

    def evaluate(self, sess, X, Y):
        return sess.run(self.acc, feed_dict={self.x: X, self.y: Y})


# ======================================================
# 4. Generate 10 Tasks for Split CIFAR-100
# ======================================================
dataset = SplitCIFAR100Data()
num_tasks = 10

trainXs, trainYs = [], []
valXs, valYs = [], []
testXs, testYs = [], []

for t in range(1, num_tasks + 1):
    tr_X, tr_Y, v_X, v_Y, te_X, te_Y = dataset.get_task_data(t)
    trainXs.append(tr_X);
    trainYs.append(tr_Y)
    valXs.append(v_X);
    valYs.append(v_Y)
    testXs.append(te_X);
    testYs.append(te_Y)

# ======================================================
# 5. TRAINING LOOPS
# ======================================================

# --- Original DEN ---
den_model = DEN.DEN(FLAGS)
params = dict()
den_avg_perf, den_hidden_units = [], []
den_expansion_history, den_split_history = [], []

for t in range(num_tasks):
    data = (trainXs[t], trainYs[t], valXs[t], valYs[t], testXs[t], testYs[t])
    den_model.sess = tf.Session()
    print("\n===== Original DEN TASK %d =====" % (t + 1))
    den_model.T += 1
    den_model.task_indices.append(t + 1)
    den_model.load_params(params, time=1)

    perf, sparsity, counts = den_model.add_task(t + 1, data)
    ex_counts, sp_counts = counts
    den_expansion_history.append(sum(ex_counts))
    den_split_history.append(sum(sp_counts))

    params = den_model.get_params()
    den_hidden_units.append(get_hidden_capacity(params, den_model.n_layers, t + 1))

    den_model.destroy_graph()
    den_model.sess.close()

    den_model.sess = tf.Session()
    den_model.load_params(params)
    perfs = []
    for j in range(t + 1):
        acc = den_model.predict_perform(j + 1, testXs[j], testYs[j])
        perfs.append(acc)
    avg = sum(perfs) / (t + 1)
    den_avg_perf.append(avg)
    print("Original DEN avg_perf:", avg)
    den_model.destroy_graph()
    den_model.sess.close()

# --- DEN_fd ---
den_fd_model = DEN_fd.DEN(FLAGS)
fd_params = dict()
den_fd_avg_perf, den_fd_hidden_units = [], []
fd_expansion_history, fd_split_history = [], []

for t in range(num_tasks):
    data = (trainXs[t], trainYs[t], valXs[t], valYs[t], testXs[t], testYs[t])
    den_fd_model.sess = tf.Session()
    print("\n===== DEN_fd TASK %d =====" % (t + 1))
    den_fd_model.T += 1
    den_fd_model.task_indices.append(t + 1)
    den_fd_model.load_params(fd_params, time=1)

    perf, sparsity, counts = den_fd_model.add_task(t + 1, data)
    ex_counts, sp_counts = counts
    fd_expansion_history.append(sum(ex_counts))
    fd_split_history.append(sum(sp_counts))

    fd_params = den_fd_model.get_params()
    den_fd_hidden_units.append(get_hidden_capacity(fd_params, den_fd_model.n_layers, t + 1))

    den_fd_model.destroy_graph()
    den_fd_model.sess.close()

    den_fd_model.sess = tf.Session()
    den_fd_model.load_params(fd_params)
    perfs = []
    for j in range(t + 1):
        acc = den_fd_model.predict_perform(j + 1, testXs[j], testYs[j])
        perfs.append(acc)
    avg = sum(perfs) / (t + 1)
    den_fd_avg_perf.append(avg)
    print("DEN_fd avg_perf:", avg)
    den_fd_model.destroy_graph()
    den_fd_model.sess.close()

# --- Vanilla DNN ---
dnn_avg_perf = []
dnn = VanillaDNN(FLAGS.dims, FLAGS.lr)

with tf.Session(graph=dnn.graph) as sess:
    sess.run(dnn.init)
    for t in range(num_tasks):
        print("\n===== DNN TASK %d =====" % (t + 1))
        for epoch in range(4000):
            dnn.train_epoch(sess, trainXs[t], trainYs[t], FLAGS.batch_size)
        perfs = []
        for j in range(t + 1):
            acc = dnn.evaluate(sess, testXs[j], testYs[j])
            perfs.append(acc)
        avg = sum(perfs) / (t + 1)
        dnn_avg_perf.append(avg)
        print("DNN avg_perf:", avg)

# ======================================================
# 6. Plot comparison (10 Tasks)
# ======================================================
plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
tasks_range = range(1, num_tasks + 1)
tasks_labels = [f"T{i}" for i in tasks_range]

ax1.plot(tasks_range, den_avg_perf, marker='o', linestyle='-', linewidth=2, label="Original DEN")
ax1.plot(tasks_range, den_fd_avg_perf, marker='s', linestyle='-', linewidth=2, label="DEN-fd")
ax1.plot(tasks_range, dnn_avg_perf, marker='^', linestyle='--', color='gray', label="DNN")
ax1.set_xlabel("Number of Tasks")
ax1.set_ylabel("Average Accuracy")
ax1.set_title("Average Accuracy Comparison", fontsize=14)
ax1.set_xticks(tasks_range)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

ax2.plot(tasks_range, den_hidden_units, marker='o', linewidth=2, label="Original DEN")
ax2.plot(tasks_range, den_fd_hidden_units, marker='s', linewidth=2, label="DEN-fd")
dnn_fixed_cap = get_dnn_hidden_capacity(FLAGS.dims)
ax2.plot(tasks_range, [dnn_fixed_cap] * num_tasks, linestyle='--', color='gray', label="DNN")
ax2.set_xlabel("Number of Tasks")
ax2.set_ylabel("Total Hidden Units")
ax2.set_title("Network Capacity Growth", fontsize=14)
ax2.set_xticks(tasks_range)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

x = np.arange(len(tasks_labels))
width = 0.35
ax3.bar(x - width / 2, den_expansion_history, width, label='DEN: Expansion', color='#1f77b4', alpha=0.6,
        edgecolor='white')
ax3.bar(x - width / 2, den_split_history, width, bottom=den_expansion_history, label='DEN: Splitting', color='#aec7e8',
        alpha=0.6, edgecolor='white')
ax3.bar(x + width / 2, fd_expansion_history, width, label='DEN-fd: Expansion', color='#ff7f0e', edgecolor='white')
ax3.bar(x + width / 2, fd_split_history, width, bottom=fd_expansion_history, label='DEN-fd: Splitting', color='#ffbb78',
        edgecolor='white')

ax3.set_ylabel("Newly Added Hidden Units")
ax3.set_title("Growth Source: DEN vs DEN-fd", fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(tasks_labels)
ax3.legend(loc='upper left', fontsize='small', ncol=2)
ax3.grid(axis='y', linestyle=':', alpha=0.6)

for i in range(len(x)):
    total_den = den_expansion_history[i] + den_split_history[i]
    total_fd = fd_expansion_history[i] + fd_split_history[i]
    ax3.text(i - width / 2, total_den + 0.5, str(int(total_den)), ha='center', va='bottom', fontsize=8)
    ax3.text(i + width / 2, total_fd + 0.5, str(int(total_fd)), ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()