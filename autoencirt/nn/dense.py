import tensorflow as tf
import tensorflow_probability as tfp

from autoencirt.tools.tf import SqrtInverseGamma

tfd = tfp.distributions

weight_code = """ lambda lambda_{0}_{1}, tau_{0}_{1}: tfd.Independent(
    tfd.Normal(
        loc=tf.zeros({3}),
        scale=lambda_{0}_{1}*tau_{0}_{1}
    ),
    reinterpreted_batch_ndims={2}
)
"""

cauchy_code = """ tfd.Independent(
    tfd.HalfCauchy(
        loc=tf.zeros({0}),
        scale={1}*tf.ones({0})
    ),
    reinterpreted_batch_ndims={2}
)
"""

sq_igamma_code = """ lambda {1}: tfd.Independent(
    SqrtInverseGamma(
        concentration=0.5*tf.ones({0}),
        scale=1.0/{1}
    ),
    reinterpreted_batch_ndims={2}
)
"""

igamma_code = """ tfd.Independent(
    tfd.InverseGamma(
        concentration=0.5*tf.ones({0}),
        scale={1}*tf.ones({0})
    ),
    reinterpreted_batch_ndims={2}
)
"""


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
    distribution = None
    surrogate_distribution = None
    reparameterized = True

    def __init__(self, input_size=None, layer_sizes=None,
                 reparameterized=False, decay=0.5, *args, **kwargs):
        super().__init__(input_size, layer_sizes, *args, **kwargs)
        self.layer_sizes = [input_size] + layer_sizes
        self.reparameterized = reparameterized
        self.assemble_distributions()
        self.decay = decay  # dimensional decay

    def set_weights(self, weights):
        super().set_weights(weights)
        self.assemble_distributions()

    def assemble_distributions(self):
        distribution_dict = {}
        bijectors = {}
        for j, weight in enumerate(self.weights[::2]):
            distribution_dict['w_' + str(j)] = eval(
                weight_code.format(
                    'w',
                    j,
                    2,
                    (self.layer_sizes[j],
                     self.layer_sizes[j+1])))
            distribution_dict['b_' + str(j)] = eval(
                weight_code.format(
                    'b',
                    j,
                    1,
                    (self.layer_sizes[j+1],)
                )
            )
            bijectors['w_' + str(j)] = tfp.bijectors.Identity()
            bijectors['b_' + str(j)] = tfp.bijectors.Identity()
            if not self.reparameterized:
                bijectors['tau_w_{0}'.format(j)] = tfp.bijectors.Softplus()
                bijectors['lambda_w_{0}'.format(j)] = tfp.bijectors.Softplus()
                bijectors['tau_b_{0}'.format(j)] = tfp.bijectors.Softplus()
                bijectors['lambda_b_{0}'.format(j)] = tfp.bijectors.Softplus()
                distribution_dict[
                    'tau_w_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        (1, self.layer_sizes[j+1]),
                        1.,
                        2
                    )
                )
                distribution_dict[
                    'tau_b_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        1,
                        1.,
                        1
                    )
                )
                distribution_dict[
                    'lambda_b_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        (self.layer_sizes[j+1],),
                        1.,
                        1
                    )
                )
                distribution_dict[
                    'lambda_w_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        (self.layer_sizes[j], self.layer_sizes[j+1]),
                        1.,
                        2
                    )
                )
            else:
                distribution_dict[
                    'lambda_b_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        (self.layer_sizes[j+1],),
                        'lambda_b_{0}_a'.format(j),
                        1
                    )
                )

                distribution_dict[
                    'lambda_b_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        (self.layer_sizes[j+1],),
                        1.0,
                        1
                    )
                )

                distribution_dict[
                    'lambda_w_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        (self.layer_sizes[j], self.layer_sizes[j+1]),
                        'lambda_w_{0}_a'.format(j),
                        2
                    )
                )

                distribution_dict[
                    'lambda_w_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        (self.layer_sizes[j], self.layer_sizes[j+1]),
                        1.,
                        2
                    )
                )

                distribution_dict[
                    'tau_w_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        (1, self.layer_sizes[j+1]),
                        'tau_w_{0}_a'.format(j),
                        2
                    )
                )
                distribution_dict[
                    'tau_w_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        (1, self.layer_sizes[j+1]),
                        1.,
                        2
                    )
                )
                distribution_dict[
                    'tau_b_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        1,
                        'tau_b_{0}_a'.format(j),
                        1
                    )
                )

                distribution_dict[
                    'tau_b_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        1,
                        1.,
                        1
                    )
                )
        self.bijectors = bijectors
        self.distribution = tfd.JointDistributionNamed(distribution_dict)


class DenseHorseshoeAE(DenseHorseshoe):
    pass


def main():
    dense = Dense(10, [20, 12, 2])
    denseH = DenseHorseshoe(10, [20, 12, 2],
                            reparameterized=True)
    sample = denseH.distribution.sample()
    return


if __name__ == "__main__":
    main()
