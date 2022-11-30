from itertools import product

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bayesianquilts.model import BayesianModel
from bayesianquilts.nn.dense import Dense, DenseHorseshoe
from bayesianquilts.util import clip_gradients, run_chain
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib

tfd = tfp.distributions


class IRTModel(BayesianModel):

    def __init__(
            self,
            item_keys,
            num_people,
            num_groups=None,
            data=None,
            person_key="person",
            dim=1,
            decay=0.25,
            positive_discriminations=True,
            missing_val=-9999,
            full_rank=False,
            eta_scale=1e-2,
            kappa_scale=1e-2,
            weight_exponent=1.0,
            response_cardinality=5,
            discrimination_guess=None,
            include_independent=True,
            vi_mode='advi',
            dtype=tf.float64):
        super(IRTModel, self).__init__(
            data
        )

        self.dtype = dtype

        self.item_keys = item_keys
        self.num_items = len(item_keys)
        self.missing_val = missing_val
        self.person_key = person_key
        self.positive_discriminations = positive_discriminations
        self.eta_scale = eta_scale
        self.kappa_scale = kappa_scale
        self.weight_exponent = weight_exponent
        self.response_cardinality = response_cardinality
        self.num_people = num_people
        self.full_rank = full_rank
        self.include_independent = include_independent
        self.discrimination_guess = discrimination_guess
        self.vi_mode = vi_mode
        self.num_groups = num_groups
        self.dtype = dtype
        # self.create_distributions()
        self.set_dimension(dim, decay)
  
    def set_dimension(self, dim, decay=0.25):
        self.dimensions = dim
        self.dimensional_decay = decay
        self.kappa_scale *= (decay**tf.cast(
            tf.range(dim), self.dtype)
        )[tf.newaxis, :, tf.newaxis, tf.newaxis]

    def set_params_from_samples(self, samples):
        """_summary_

        Args:
            samples (_type_): _description_
        """
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
