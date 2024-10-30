import numpy as np
import tensorflow as tf
from collections import defaultdict


def get_need_model(graph):
    if not graph["A"] or not graph["A"][0] or not graph["A"][1]:
        # Mtrx = tf.eye(1, dtype=tf.float32)
        # H0 = tf.ones([1, 9], dtype=tf.float32)
        return tf.zeros([0, 0]), tf.zeros([0, 0])
        # return Mtrx, H0
    else:
        N = max(max(graph["A"][0]) + 1, max(graph["A"][1]) + 1)

    indices_A = np.transpose(graph["A"])
    values = np.array([v if index[0] != index[1] else 1
                       for v, index in zip(graph["edges_feature"], indices_A)],
                      dtype=np.float32)

    index_dict = defaultdict(float)
    for idx, val in zip(indices_A, values):
        index_dict[tuple(idx)] += val

    indices_A = np.array(list(index_dict.keys()))
    values = np.array(list(index_dict.values()), dtype=np.float32)

    v = np.array(graph['nodes_feature'], dtype=np.float32)

    if len(v) != N:
        v = v[:N] if len(v) > N else np.pad(v, ((0, N - len(v)), (0, 0)), 'constant', constant_values=0)


    max_ = np.max(v, axis=0)
    min_ = np.min(v, axis=0)
    delta_ = max_ - min_

    for i in range(len(v[0])):
        v[:, i] = (max_[i] - v[:, i]) / delta_[i] if delta_[i] != 0 else v[:, i]

    H0 = tf.constant(v, dtype=tf.float32)

    A = tf.SparseTensor(indices=indices_A,
                        values=values,
                        dense_shape=[N, N])

    A = tf.sparse.reorder(A)
    A_with_I = tf.sparse.add(A, tf.sparse.eye(N))

    D_diag_values = tf.sparse.reduce_sum(A_with_I, axis=1)
    D_inv_sqrt_diag_values = tf.math.pow(D_diag_values, -0.5)
    D_inv_sqrt_indices = np.array([[i, i] for i in range(N)], dtype=np.int64)
    D_inv_sqrt = tf.SparseTensor(
        indices=D_inv_sqrt_indices,
        values=D_inv_sqrt_diag_values,
        dense_shape=[N, N]
    )

    temp = tf.sparse.sparse_dense_matmul(D_inv_sqrt, tf.sparse.to_dense(A_with_I))
    Mtrx = tf.sparse.sparse_dense_matmul(D_inv_sqrt, temp)

    return Mtrx, H0


def get_Mtrxs(graph):
    Mtrx, H0 = get_need_model(graph)
    true_label = graph["label"]
    return Mtrx, H0, true_label


def classification_blocks(model, graph):
    Mtrx, H0 = get_need_model(graph)
    label_pred = model(Mtrx, H0)
    print(label_pred)
    a = np.argmax(label_pred)
    return a