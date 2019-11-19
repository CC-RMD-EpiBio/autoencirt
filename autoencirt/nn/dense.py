import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class DenseNetwork(object):
    fn = None
    parameter_tensors = None
    
    def __init__(self, input_size=None, layer_sizes=None):
        if (input_size is None) or (layer_sizes is None):
            self.fn = lambda x: x
        else:
            self.parameter_tensors = self.sample_initial_nn_params(
                input_size, layer_sizes)
            _ = self.build_network()

    def dense(self, X, W, b, activation):
        return activation(tf.matmul(X, W) + b)

    def build_network(self, activation=tf.nn.selu):
        @tf.function
        def model(X):
            net = X
            net = tf.cast(net, tf.float32)
            weights_list = self.parameter_tensors[::2]
            biases_list = self.parameter_tensors[1::2]
            for (weights, biases) in zip(weights_list, biases_list):
                net = dense(net, weights, biases, activation)
            return net
        self.fn = model
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


class DenseHorseshoeNetwork(DenseNetwork):
    def __init__(self, *args, **kwargs):
        pass
