from itertools import product

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bayesianquilts.nn.dense import Dense, DenseHorseshoe
# from bayesianquilts.model import BayesianModel
from autoencirt.irt.model import BayesianModel
from bayesianquilts.util import (clip_gradients,
                                 fit_surrogate_posterior, run_chain)
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from bayesianquilts.nn.dense import Dense
tfd = tfp.distributions


class IRTModel(BayesianModel):
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
    xi_scale = None
    kappa_scale = None
    positive_discriminations = True
    scoring_network = None
    dtype = tf.float64
    item_keys = []
    weight_exponent = 1.0
    xi_scale = .1
    kappa_scale = .1
    eta_scale = .1

    def __init__(
            self,
            item_keys,
            num_people,
            data=None,
            response_cardinality=5,
            person_key="person",
            dim=1,
            decay=0.25,
            positive_discriminations=True,
            missing_val=-9999,
            xi_scale=1e-2,
            eta_scale=1e-2,
            kappa_scale=1e-2,
            weight_exponent=1.0,
            dtype=tf.float64):
        super(IRTModel, self).__init__(
            data, None, None
        )
        self.dtype = dtype

        self.set_dimension(dim, decay)
        self.item_keys = item_keys
        self.num_items = len(item_keys)
        self.missing_val = missing_val
        self.person_key = person_key
        self.positive_discriminations = positive_discriminations
        self.xi_scale = xi_scale
        self.eta_scale = eta_scale
        self.kappa_scale = kappa_scale
        self.weight_exponent = weight_exponent
        self.response_cardinality = response_cardinality
        self.num_people = num_people
        # self.create_distributions()

    def set_dimension(self, dim, decay=0.25):
        self.dimensions = dim
        self.dimensional_decay = decay
        self.kappa_scale *= (decay**tf.cast(
            tf.range(dim), self.dtype)
        )[tf.newaxis, :, tf.newaxis, tf.newaxis]

    def set_params_from_samples(self, samples):
        try:
            for k in self.var_list:
                self.surrogate_sample[k] = samples[k]
        except KeyError:
            print(str(k) + " doesn't exist in your samples")
            return
        self.set_calibration_expectations()

    def create_distributions(self):
        pass

    def obtain_scoring_nn(self, hidden_layers=None):
        if self.calibrated_traits is None:
            print("Please calibrate the IRT model first")
            return
        if hidden_layers is None:
            hidden_layers = [self.num_items*2, self.num_items*2]
        dnn = Dense(
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
        rv = tfd.Bernoulli(
            tf.ones_like(discrimination, dtype=self.dtype)*(1.0-sparsity))
        discrimination = discrimination*tf.cast(rv.sample(), dtype=self.dtype)
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
