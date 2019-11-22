import tensorflow as tf
import numpy as np
from itertools import product
from autoencirt.nn import Dense, DenseHorseshoe

from factor_analyzer import (
    FactorAnalyzer)


import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from autoencirt.tools.tf import (
    clip_gradients, run_chain, LossLearningRateScheduler,
    build_trainable_InverseGamma_dist,
    build_trainable_normal_dist
)


class IRTModel(object):
    response_type = None
    calibration_data = None
    num_people = None
    num_items = None
    response_data = None
    response_cardinality = None
    dimensions = 1
    weighted_likelihood = None
    calibrated_traits = None
    calibrated_traits_sd = None
    calibrated_discriminations = None
    calibrated_discriminations_sd = None
    calibrated_difficulties = None
    calibrated_likelihood_distribution = None
    calibrated_trait_scale = 1
    bijectors = None
    dimensional_decay = 0.25
    surrogate_sample = None
    scoring_grid = None  # D dimensional
    grid = None

    scoring_network = None

    def __init__(self):
        self.set_dimension(1)

    def set_dimension(self, dim, decay=0.25, bins=50):
        self.dimensions = dim
        self.dimensional_decay = decay

    def set_params_from_samples(self, samples):
        try:
            for key in self.surrogate_sample.keys():
                self.surrogate_sample[key] = samples[key]
        except KeyError:
            print(str(key) + " doesn't exist in your samples")
            return
        self.set_calibration_expectations()

    def load_data(self, response_data):
        self.response_data = response_data
        self.num_people = response_data.shape[0]
        self.num_items = response_data.shape[1]
        self.response_cardinality = int(max(response_data.max())) + 1
        fa = FactorAnalyzer(
            rotation=None, n_factors=self.dimensions)
        fa.fit(response_data)
        self.linear_loadings = fa.loadings_
        if int(min(response_data.min())) == 1:
            print("Warning: responses do not appear to be from zero")
        self.calibration_data = tf.cast(response_data.to_numpy(), tf.int32)

    def create_distributions(self):
        pass

    def set_calibration_expectations(self):
        pass

    def calibrate(self):
        pass

    def score(self, responses):
        pass

    def loss(self, responses):
        pass

    def obtain_scoring_nn(self, hidden_layers=None):
        if calibrated_traits is None:
            print("Please calibrate the IRT model first")
            return
        if hidden_layers is None:
            hidden_layers = [self.num_items*2, self.num_items*2]
        dnn = DenseNetwork(
            self.num_items,
            [self.num_items] + hidden_layers + [self.dimensions]
        )
        ability_distribution = tfd.Independent(
            tfd.Normal(
                loc=tf.reduce_mean(
                    self.surrogate_sample['abilities'],
                    axis=0),
                scale=tf.math.reduce_std(
                    self.surrogate_sample['abilities'],
                    axis=0
                )
            ), reinterpreted_batch_ndims=2
        )
        dnn_params = dnn.weights

        def loss():
            dnn_fun = dnn.build_network(dnn_params, tf.nn.relu)
            return -ability_distribution.log_prob(dnn_fun(self.response_data))

    def calibrate_mcmc(self, init_state=None, step_size=1e-3,
                       num_steps=1000, burnin=500, nuts=True):
        """Calibrate using HMC/NUT

        Keyword Arguments:
            num_chains {int} -- [description] (default: {1})
        """
        if init_state is None:
            surrogate_expectations = {
                k: tf.reduce_mean(v, axis=0)
                for k, v in self.surrogate_sample.items()}
            init_state = surrogate_expectations

        initial_list = tf.nest.flatten(init_state)
        bijectors = [self.bijectors[k] for k in sorted(init_state.keys())]
        samples, sampler_stat = run_chain(
            init_state=initial_list,
            step_size=step_size,
            target_log_prob_fn=self.unormalized_log_prob_list,
            unconstraining_bijectors=bijectors,
            num_steps=num_steps,
            burnin=burnin,
            nuts=nuts
        )
        self.surrogate_sample = tf.nest.pack_sequence_as(
            self.surrogate_sample, samples
        )
        self.set_calibration_expectations()

        return samples, sampler_stat

    @tf.function
    def unormalized_log_prob_list(self, *x):
        return self.unormalized_log_prob(
            **tf.nest.pack_sequence_as(
                self.surrogate_sample,
                x
            ))

    @tf.function
    def unormalized_log_prob(self, **x):
        x['x'] = self.calibration_data
        return self.weighted_likelihood.log_prob(x)

    def calibrate_advi(
            self, num_steps=10, initial_learning_rate=5e-3,
            decay_rate=0.99, learning_rate=None,
            opt=None):
        if learning_rate is None:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=num_steps,
                decay_rate=decay_rate,
                staircase=True)
        if opt is None:
            opt = tf.optimizers.Adam(
                learning_rate=learning_rate)

        @tf.function
        def run_approximation(num_steps):
            losses = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=self.unormalized_log_prob,
                surrogate_posterior=self.surrogate_posterior,
                optimizer=opt,
                num_steps=num_steps,
                sample_size=10
            )
            return(losses)

        losses = run_approximation(num_steps)
        self.set_calibration_expectations()
        print(losses)
        return
