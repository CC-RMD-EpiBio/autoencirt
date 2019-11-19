#!/usr/bin/env python3
import math
import os
from os import path, system
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.mcmc.transformed_kernel import (
    make_transform_fn, make_transformed_log_prob, make_log_det_jacobian_fn)
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import (
    build_factored_surrogate_posterior)

from tensorflow.python import tf2
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()


tfd = tfp.distributions

tfd = tfp.distributions
tfb = tfp.bijectors


def clip_gradients(fn, clip_value):
    def wrapper(*args):
        @tf.custom_gradient
        def grad_wrapper(*flat_args):
            with tf.GradientTape() as tape:
                tape.watch(flat_args)
                ret = fn(*tf.nest.pack_sequence_as(args, flat_args))

            def grad_fn(*dy):
                flat_grads = tape.gradient(ret, flat_args, output_gradients=dy)
                flat_grads = tf.nest.map_structure(lambda g: tf.where(
                    tf.math.is_finite(g), g, tf.zeros_like(g)), flat_grads)
                return tf.clip_by_global_norm(flat_grads, clip_value)[0]
            return ret, grad_fn
        return grad_wrapper(*tf.nest.flatten(args))
    return wrapper


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
    bijectors = None

    def __init__(self):
        pass

    def set_dimension(self, dim):
        self.dimensions = dim

    def load_data(self, response_data):
        self.response_data = response_data
        self.num_people = response_data.shape[0]
        self.num_items = response_data.shape[1]
        self.response_cardinality = int(max(response_data.max())) + 1
        if int(min(response_data.min())) == 1:
            print("Warning: responses do not appear to be from zero")
        self.calibration_data = tf.cast(response_data.to_numpy(), tf.int32)

    def create_distributions(self):
        pass

    def calibrate(self):
        pass

    def score(self, responses):
        pass

    def loss(self, responses):
        pass


class GRModel(IRTModel):
    """Implement and store the graded response model for IRT

    Arguments:
        IRTModel {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    data = None
    response_type = "polytomous"

    def __init__(self, auxiliary_parameterization=True):
        super().__init__()
        self.auxiliary_parameterization = auxiliary_parameterization

    def grm_model_prob(self, abilities, discriminations, difficulties):
        if len(difficulties.shape) == 3:
            offsets = (difficulties[tf.newaxis, :, :, :]
                       - abilities[:, :, tf.newaxis, tf.newaxis])
            scaled = offsets*discriminations[tf.newaxis, :, :, tf.newaxis]
            logits = 1.0/(1+tf.exp(scaled))
            logits = tf.pad(logits, paddings=(
                (0, 0), (0, 0), (0, 0), (1, 0)),
                mode='constant', constant_values=1)
            logits = tf.pad(logits, paddings=(
                (0, 0), (0, 0), (0, 0), (0, 1)),
                mode='constant', constant_values=0)
            probs = logits[:, :, :, :-1] - logits[:, :, :, 1:]
            # weight by discrimination
            # \begin{align}
            #   w_{id} &= \frac{\lambda_{i}^{(d)}}{\sum_d \lambda_{i}^{(d)}}.
            # \end{align}
            weights = (discriminations
                       / tf.reduce_sum(discriminations, axis=0)[tf.newaxis, :]
                       )
            probs = tf.reduce_sum(
                probs*weights[tf.newaxis, :, :, tf.newaxis], axis=1)
        else:
            offsets = (difficulties[:, tf.newaxis, ...]
                       - abilities[..., tf.newaxis, tf.newaxis])
            scaled = offsets*discriminations[:, tf.newaxis, :, :, tf.newaxis]
            logits = 1.0/(1+tf.exp(scaled))
            logits = tf.pad(logits, paddings=(
                (0, 0), (0, 0), (0, 0), (0, 0), (1, 0)),
                mode='constant', constant_values=1)
            logits = tf.pad(logits, paddings=(
                (0, 0), (0, 0), (0, 0), (0, 0), (0, 1)),
                mode='constant', constant_values=0)
            probs = logits[..., :-1] - logits[..., 1:]
            # weight by discrimination
            # \begin{align}
            #   w_{id} &= \frac{\lambda_{i}^{(d)}}{\sum_d \lambda_{i}^{(d)}}.
            # \end{align}
            weights = (
                discriminations
                / tf.reduce_sum(discriminations, axis=1)[:, tf.newaxis, :])
            probs = tf.reduce_sum(
                probs*weights[:, tf.newaxis, :, :, tf.newaxis], axis=2)
        return probs

    @tf.function
    def grm_model_prob_d(self,
                         abilities,
                         discriminations,
                         difficulties0,
                         ddifficulties
                         ):
        d0 = tf.concat(
            [difficulties0[..., tf.newaxis], ddifficulties], axis=-1)
        difficulties = tf.cumsum(d0, axis=-1)
        return self.grm_model_prob(abilities, discriminations, difficulties)

    """
    Probabilities for single items
    discriminations  D (domain) x I (item)
    difficulties D x I x K - 1
    abilities Dx1
    xi (local horseshoe) D x I
    eta (global horseshoe) I
    mu (difficulty local) D x I
    tau (difficulty) D x I x K-2

    """

    @tf.function(autograph=False)
    def joint_log_prob(responses,
                       discriminations,
                       difficulties0,
                       ddifficulties,
                       abilities,
                       xi,
                       eta,
                       mu):
        """[summary]

        Arguments:
            responses {[type]} -- [description]
            discriminations {[type]} -- [description]
            difficulties0 {[type]} -- [description]
            ddifficulties {[type]} -- [description]
            abilities {[type]} -- [description]
            xi {[type]} -- [description]
            eta {[type]} -- [description]
            mu {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        d0 = tf.concat(
            [difficulties0[..., tf.newaxis], ddifficulties], axis=-1)
        difficulties = tf.cumsum(d0, axis=-1)
        return (
            tf.reduce_sum(
                log_likelihood(responses,
                               discriminations,
                               difficulties,
                               abilities)
            )
            + joint_log_prior(
                discriminations, difficulties0,
                ddifficulties, abilities, xi, eta, mu)
        )

    @tf.function(autograph=False)
    def log_likelihood(responses, discriminations, difficulties, abilities):
        rv_responses = tfd.Categorical(
            self.grm_model_prob(
                abilities, discriminations, difficulties), validate_args=True)
        return rv_responses.log_prob(responses)

    @tf.function
    def joint_log_prior(discriminations, difficulties0,
                        ddifficulties, abilities, xi, eta, mu):
        if len(mu.shape) == 2:
            rv_discriminations = tfd.HalfNormal(scale=eta*xi)
            rv_difficulties0 = tfd.Normal(loc=mu, scale=1.)
            rv_ddifficulties = tfd.HalfNormal(
                scale=tf.ones_like(ddifficulties))
            rv_abilities = tfd.Normal(loc=tf.zeros_like(abilities), scale=1.)
            rv_eta = tfd.HalfCauchy(loc=tf.zeros_like(eta),
                                    scale=tf.ones_like(eta))
            rv_xi = tfd.HalfCauchy(loc=tf.zeros_like(xi),
                                   scale=tf.ones_like(xi))
            rv_mu = tfd.Normal(loc=tf.zeros_like(mu), scale=1.)

            return (tf.reduce_sum(rv_discriminations.log_prob(discriminations))
                    + tf.reduce_sum(rv_difficulties0.log_prob(difficulties0))
                    + tf.reduce_sum(rv_ddifficulties.log_prob(ddifficulties))
                    + tf.reduce_sum(rv_abilities.log_prob(abilities))
                    + tf.reduce_sum(rv_eta.log_prob(eta))
                    + tf.reduce_sum(rv_xi.log_prob(xi))
                    + tf.reduce_sum(rv_mu.log_prob(mu)))
        elif len(mu.shape) == 3:
            rv_discriminations = tfd.HalfNormal(
                scale=eta[..., tf.newaxis, :]*xi)
            rv_difficulties0 = tfd.Normal(loc=mu, scale=1.)
            rv_ddifficulties = tfd.HalfNormal(
                scale=tf.ones_like(ddifficulties))
            rv_abilities = tfd.Normal(loc=tf.zeros_like(abilities), scale=1.)
            rv_eta = tfd.HalfCauchy(loc=tf.zeros_like(eta),
                                    scale=tf.ones_like(eta))
            rv_xi = tfd.HalfCauchy(loc=tf.zeros_like(xi),
                                   scale=tf.ones_like(xi))
            rv_mu = tfd.Normal(loc=tf.zeros_like(mu), scale=1.)

            return (
                tf.reduce_sum(
                    rv_discriminations.log_prob(discriminations),
                    axis=list(range(1, (len(discriminations.shape))))
                )
                + tf.reduce_sum(
                    rv_difficulties0.log_prob(difficulties0),
                    axis=list(range(1, (len(difficulties0.shape))))
                )
                + tf.reduce_sum(
                    rv_ddifficulties.log_prob(ddifficulties),
                    axis=list(range(1, (len(ddifficulties.shape))))
                )
                + tf.reduce_sum(
                    rv_abilities.log_prob(abilities),
                    axis=list(range(1, (len(abilities.shape))))
                )
                + tf.reduce_sum(
                    rv_eta.log_prob(eta),
                    axis=list(range(1, (len(eta.shape))))
                )
                + tf.reduce_sum(
                    rv_xi.log_prob(xi),
                    axis=list(range(1, (len(xi.shape))))
                )
                + tf.reduce_sum(
                    rv_mu.log_prob(mu),
                    axis=list(range(1, (len(mu.shape))))
                )
            )

    def joint_prior_distribution(self):
        K = self.response_cardinality
        grm_joint_distribution_dict = dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros((self.dimensions, self.num_items)),
                    scale=1.
                ),
                reinterpreted_batch_ndims=2
            ),  # mu
            eta=tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros((self.num_items)), scale=1.
                ),
                reinterpreted_batch_ndims=1
            ),  # eta
            xi=tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros((self.dimensions, self.num_items)),
                    scale=1.
                ),
                reinterpreted_batch_ndims=2
            ),  # xi
            difficulties0=lambda mu: tfd.Independent(
                tfd.Normal(loc=mu, scale=1.),
                reinterpreted_batch_ndims=2
            ),  # difficulties0
            discriminations=lambda eta, xi: tfd.Independent(
                tfd.HalfNormal(scale=tf.sqrt(eta[..., tf.newaxis, :]*xi)),
                reinterpreted_batch_ndims=2
            ),  # discrimination
            ddifficulties=tfd.Independent(
                tfd.HalfNormal(
                    scale=tf.ones(
                        (self.dimensions,
                         self.num_items,
                         self.response_cardinality-2
                         )
                    )
                ),
                reinterpreted_batch_ndims=3
            ),
            abilities=tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros((self.num_people, self.dimensions)),
                    scale=1.
                ),
                reinterpreted_batch_ndims=2
            ),
            x=lambda abilities, discriminations, difficulties0, ddifficulties:
            tfd.Independent(
                tfd.Categorical(
                    self.grm_model_prob_d(
                        abilities,
                        discriminations,
                        difficulties0,
                        ddifficulties
                    ),
                    validate_args=True),
                reinterpreted_batch_ndims=2
            )
        )
        if self.auxiliary_parameterization:
            grm_joint_distribution_dict["xi_a"] = tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((self.dimensions, self.num_items)),
                    tf.ones((self.dimensions, self.num_items))
                ),
                reinterpreted_batch_ndims=2
            )
            grm_joint_distribution_dict["xi"] = lambda xi_a: tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((self.dimensions, self.num_items)),
                    1.0/xi_a
                ),
                reinterpreted_batch_ndims=2
            )
            grm_joint_distribution_dict["eta_a"] = tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((self.num_items)),
                    tf.ones((self.num_items))
                ),
                reinterpreted_batch_ndims=1
            )
            grm_joint_distribution_dict["eta"] = lambda eta_a: tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((self.num_items)),
                    1.0/eta_a
                ),
                reinterpreted_batch_ndims=1
            )

        return tfd.JointDistributionNamed(grm_joint_distribution_dict)

    def create_distributions(self):
        self.weighted_likelihood = self.joint_prior_distribution()
        self.bijectors = {
            k: tfp.bijectors.Identity() for k in self.weighted_likelihood.parameters['model'].keys()}
        del self.bijectors['x']
        self.bijectors['eta'] = tfp.bijectors.Softplus()
        self.bijectors['xi'] = tfp.bijectors.Softplus()
        self.bijectors['discriminations'] = tfp.bijectors.Softplus()
        self.bijectors['ddifficulties'] = tfp.bijectors.Softplus()
        if self.auxiliary_parameterization:
            self.bijectors['eta_a'] = tfp.bijectors.Softplus()
            self.bijectors['xi_a'] = tfp.bijectors.Softplus()

        event_shape = self.weighted_likelihood.event_shape_tensor()
        del event_shape['x']
        variable_names = event_shape.keys()

        self.surrogate_posterior = build_factored_surrogate_posterior(
            event_shape=event_shape,
            constraining_bijectors=self.bijectors
        )

    def calibrate_advi(self, num_steps=10):
        def unormalized_log_prob(**x):
            x['x'] = self.calibration_data
            return self.weighted_likelihood.log_prob(x)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-1,
            decay_steps=5,
            decay_rate=0.99,
            staircase=True)
        opt = tf.optimizers.Adam(
            learning_rate=learning_rate,
            clipvalue=3.)

        @tf.function
        def run_approximation(num_steps):
            losses = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=unormalized_log_prob,
                surrogate_posterior=self.surrogate_posterior,
                optimizer=opt,
                num_steps=num_steps,
                sample_size=5
            )
            return(losses)

        losses = run_approximation(num_steps)
        posterior_samples = self.surrogate_posterior.sample(1000)
        self.calibrated_traits = tf.reduce_mean(
            posterior_samples['abilities'], axis=0
            )
        self.calibrated_traits_sd = tf.math.reduce_std(
            posterior_samples['abilities'], axis=0
            )            
        self.calibrated_discriminations = tf.reduce_mean(
            posterior_samples['discriminations'], axis=0
            )
        self.calibrated_discriminations_sd = tf.math.reduce_std(
            posterior_samples['discriminations'], axis=0
            )

        difficulty_samples = None
        self.calibrated_difficulties = tf.cumsum(
            tf.concat([
                tf.reduce_mean(
                    posterior_samples['difficulties0'], axis=0
                )[..., tf.newaxis],
                tf.reduce_mean(
                    posterior_samples['ddifficulties'], axis=0
                )
            ], axis=-1), axis=-1
        )
        return(losses)

    def calibrate_mcmc(self, num_chains=1):
        initial_state_dict = self.weighted_likelihood.sample(num_chains)

    def score(self, responses):
        pass

    def loss(self, responses, scores):
        pass


def get_data():
    if not path.exists('RWAS/data.csv'):
        system("wget https://openpsychometrics.org/_rawdata/RWAS.zip")
        system("unzip RWAS.zip")
    data = pd.read_csv('RWAS/data.csv', low_memory=False)
    item_responses = data.loc[:, map(lambda x: 'Q'+str(x), list(range(1, 23)))]
    # system("rm -r RWAS")
    return item_responses


def main():
    data = get_data()
    kf = KFold(n_splits=4)
    kf.get_n_splits(data)
    for train_index, test_index in kf.split(data):
        grm = GRModel()
        grm.load_data(data.iloc[train_index, :])
        grm.set_dimension(2)
        grm.create_distributions()
        losses = grm.calibrate_advi(10)
        print(losses)

        scores = grm.score(
            tf.cast(data.iloc[test_index, :].to_numpy(), tf.int32)
        )
        prediction_loss = grm.loss(
            tf.cast(data.iloc[test_index, :].to_numpy(), tf.int32),
            scores
        )


if __name__ == "__main__":
    main()
