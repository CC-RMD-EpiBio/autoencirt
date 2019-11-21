import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Dense(object):
    fn = None
    weights = None

    def __init__(self, input_size=None, layer_sizes=None):
        if (input_size is None) or (layer_sizes is None):
            self.fn = lambda x: x
        else:
            self.weights = self.sample_initial_nn_params(
                input_size, layer_sizes)
            self.fn = self.build_network(self.weights)

    def dense(self, X, W, b, activation):
        return activation(tf.matmul(X, W) + b)

    def set_weights(self, weights):
        self.weights = weights
        self.fn = self.build_network(self.weights)

    def build_network(self, weight_tensors, activation=tf.nn.relu):
        @tf.function
        def model(X):
            net = X
            net = tf.cast(net, tf.float32)
            weights_list = weight_tensors[::2]
            biases_list = weight_tensors[1::2]
            for (weights, biases) in zip(weights_list, biases_list):
                net = self.dense(net, weights, biases, activation)
            return net
        return model

    def sample_initial_nn_params(self, input_size, layer_sizes, priors=None):
        """
        Priors should be either none or a list of tuples:
        [(weight prior, bias prior) for layer in layer_sizes]
        """
        architecture = []
        layer_sizes = [input_size] + layer_sizes

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                weigths = tfd.Normal(
                    loc=tf.zeros((layer_sizes[j], layer_size)), scale=1e-1
                ).sample()
                biases = tfd.Normal(
                    loc=tf.zeros((layer_size)), scale=1.
                ).sample()
                architecture += [weigths, biases]
        else:
            pass

        return architecture


class DenseHorseshoe(Dense):
    """Dense horseshoe network of given layer sizes

    Arguments:
        DenseNetwork {[type]} -- [description]
    """
    weight_distributions = None
    surrogate_density = None
    reparameterized = False

    def __init__(self, input_size=None, layer_sizes=None,
                 reparameterized=False, *args, **kwargs):
        super().__init__(input_size, layer_sizes)
        self.assemble_distribution()
        self.reparameterized = reparameterized

    def set_weights(self, weights):
        super().set_weights(weights)
        self.assemble_distribution()

    def assemble_distribution(self):
        w_dict = {
            "W" + str(w): tfd.Normal()}


class DenseHorseshoeAE(DenseHorseshoe):
    pass
