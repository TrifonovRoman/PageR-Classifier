import tensorflow as tf

params = {
    "count_neuron_layer_1": 9,
    "count_neuron_layer_2": 12,
    "output_size": 3,
}


def get_model(path=None):
    l1 = params["count_neuron_layer_1"]
    l2 = params["count_neuron_layer_2"]
    output_size = params["output_size"]
    model = BlockClassifier(l1, l2, output_size)
    if path is None:
        return model


class MyGraphConv(tf.Module):
    def __init__(self, input_size, output_size, activation_fun):
        self.W = tf.Variable(tf.random.normal(shape=[input_size, output_size], mean=0.1, stddev=1.0))
        self.B = tf.Variable(tf.random.normal(shape=[input_size, output_size], mean=1.0, stddev=1.0))
        self.activation = activation_fun

    def __call__(self, A, H):
        H_prime = self.activation(
            tf.matmul(tf.matmul(A, H), self.W) - tf.matmul(H, self.B)
        )
        return H_prime


class BlockClassifier(tf.Module):
    def __init__(self, l1, l2, output_size):
        self.conv1 = MyGraphConv(l1, l2, tf.nn.relu)
        self.conv2 = MyGraphConv(l2, output_size, tf.nn.relu)

        self.output_layer = tf.keras.layers.Dense(5, activation='softmax')

    @tf.function
    def __call__(self, A, H0):
        H1 = self.conv1(A, H0)
        H2 = self.conv2(A, H1)

        node_logits = self.output_layer(H2)

        output = tf.reduce_mean(node_logits, axis=0)

        return output

    def save(self, path):
        tf.saved_model.save(self, path)


@tf.function
def my_loss(label_pred, true_label):
    loss = tf.keras.losses.CategoricalCrossentropy()
    return loss(label_pred, true_label)