import inspect
from itertools import product

from factor_analyzer import (
    ConfirmatoryFactorAnalyzer,
    FactorAnalyzer,
    ModelSpecificationParser)

import tensorflow as tf
import numpy as np
from autoencirt.nn import Dense, DenseHorseshoe

import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from autoencirt.tools.tf import (
    clip_gradients, run_chain, LossLearningRateScheduler,
    build_trainable_InverseGamma_dist,
    build_trainable_normal_dist
)

tfd = tfp.distributions


class IRTModel(object):
    response_type = None
    calibration_data = None
    num_people = None
    num_items = None
    response_data = None
    response_cardinality = None
    dimensions = 1
    weighted_likelihood = None
    calibrated_expectations = None
    calibrated_sd = None
    calibrated_trait_scale = 1
    bijectors = None
    dimensional_decay = 0.25
    surrogate_sample = None
    scoring_grid = None  # D dimensional
    grid = None
    fa = None
    factor_loadings = None
    xi_scale = None
    kappa_scale = None
    positive_discriminations = True
    scoring_network = None

    def __init__(self,
                 dim=1,
                 decay=0.25,
                 positive_discriminations=True):
        self.set_dimension(1)
        self.var_list = inspect.getfullargspec(
            self.joint_log_prob).args[2:]
        self.set_dimension(dim, decay)
        self.positive_discriminations = positive_discriminations

    def set_dimension(self, dim, decay=0.25):
        self.dimensions = dim
        self.dimensional_decay = decay
        self.kappa_scale *= (decay**tf.cast(
            tf.range(dim), tf.float32)
        )[tf.newaxis, :, tf.newaxis, tf.newaxis]
        self.fa = FactorAnalyzer(n_factors=dim)

    def set_params_from_samples(self, samples):
        try:
            for key in self.var_list:
                self.surrogate_sample[key] = samples[key]
        except KeyError:
            print(str(key) + " doesn't exist in your samples")
            return
        self.set_calibration_expectations()

    def load_data(self, response_data):
        self.response_data = response_data
        self.num_people = response_data.shape[0]
        self.num_items = response_data.shape[1]
        self.fa.fit(response_data)
        self.factor_loadings = self.fa.loadings_
        self.response_cardinality = int(max(response_data.max())) + 1
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

    def calibrate_mcmc(self, init_state=None, step_size=1e-1,
                       num_steps=1000, burnin=500, nuts=True,
                       num_leapfrog_steps=5, clip=None):
        """Calibrate using HMC/NUT

        Keyword Arguments:
            num_chains {int} -- [description] (default: {1})
        """

        if init_state is None:
            init_state = self.calibrated_expectations

        initial_list = [init_state[v] for v in self.var_list]
        bijectors = [self.bijectors[k] for k in self.var_list]

        samples, sampler_stat = run_chain(
            init_state=initial_list,
            step_size=step_size,
            target_log_prob_fn=(
                self.unormalized_log_prob_list if clip is None
                else clip_gradients(self.unormalized_log_prob_list, clip)),
            unconstraining_bijectors=bijectors,
            num_steps=num_steps,
            burnin=burnin,
            num_leapfrog_steps=num_leapfrog_steps,
            nuts=nuts
        )
        self.surrogate_sample = {
            k: sample for k, sample in zip(self.var_list, samples)
        }
        self.set_calibration_expectations()

        return samples, sampler_stat

    def unormalized_log_prob_list(self, *x):
        return self.unormalized_log_prob(
            **{
                v: t for v, t in zip(self.var_list, x)
            }
        )

    def unormalized_log_prob2(self, **x):
        x['x'] = self.calibration_data
        return self.weighted_likelihood.log_prob(x)

    def calibrate_advi(
            self, num_steps=10, initial_learning_rate=5e-3,
            decay_rate=0.99, learning_rate=None,
            opt=None, clip=None):
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
                target_log_prob_fn=(
                    self.unormalized_log_prob if clip is None
                    else clip_gradients(self.unormalized_log_prob, clip)),
                surrogate_posterior=self.surrogate_posterior,
                optimizer=opt,
                num_steps=num_steps,
                sample_size=25
            )
            return(losses)

        losses = run_approximation(num_steps)
        if (not np.isnan(losses[-1])) and (not np.isinf(losses[-1])):
            self.surrogate_sample = self.surrogate_posterior.sample(1000)
            self.set_calibration_expectations()
        return(losses)

    def set_calibration_expectations(self):
        #  self.surrogate_sample = self.surrogate_posterior.sample(1000)

        self.calibrated_expectations = {
            k: tf.reduce_mean(v, axis=0)
            for k, v in self.surrogate_sample.items()
        }

        self.calibrated_sd = {
            k: tf.math.reduce_std(v, axis=0)
            for k, v in self.surrogate_sample.items()
        }

    def simulate_data(self, shape, sparsity=0.5):
        sampling_rv = tfd.Independent(
            tfd.Normal(
                loc=tf.reduce_mean(
                    self.calibrated_expectations['abilities'],
                    axis=0),
                scale=tf.math.reduce_std(
                    self.calibrated_expectations['abilities'],
                    axis=0)
            ),
            reinterpreted_batch_ndims=2
        )
        trait_samples = sampling_rv.sample(shape)
        discrimination = self.calibrated_expectations['discriminations']
        rv = tfd.Bernoulli(tf.ones_like(discrimination)*(1.0-sparsity))
        discrimination = discrimination*tf.cast(rv.sample(), tf.float32)
        probs = self.grm_model_prob_d(
            self.calibrated_expectations['abilities'],
            discrimination,
            self.calibrated_expectations['difficulties0'],
            self.calibrated_expectations['ddifficulties']
        )
        response_rv = tfd.Categorical(
            probs=probs
        )
        responses = response_rv.sample()
        return responses, discrimination, trait_samples

    def unormalized_log_prob(self, **x):
        if self.auxiliary_parameterization:
            return self.joint_log_prob_auxiliary(
                responses=self.calibration_data,
                **x

            )
        else:
            return self.joint_log_prob(
                responses=self.calibration_data,
                **x
            )

    def out_numpy(self):
        out = {k: v.numpy() for k, v in self.surrogate_sample.items()}
        return out

    def in_numpy(self, dict_of_numpy):
        for k, v in dict_of_numpy.items():
            self.surrogate_sample[k] = tf.cast(v, tf.float32)
            self.calibrated_expectations[k] = tf.reduce_mean(
                self.surrogate_sample[k], axis=0
            )
