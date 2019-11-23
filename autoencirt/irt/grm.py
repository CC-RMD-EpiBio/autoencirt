#!/usr/bin/env python3

import itertools
import numpy as np
import pandas as pd

from autoencirt.irt import IRTModel
from autoencirt.tools.tf import (
    clip_gradients, run_chain, LossLearningRateScheduler,
    build_trainable_InverseGamma_dist,
    build_trainable_normal_dist
)


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.mcmc.transformed_kernel import (
    make_transform_fn, make_transformed_log_prob, make_log_det_jacobian_fn)

from tensorflow_probability.python.bijectors import softplus as softplus_lib

tfd = tfp.distributions

tfd = tfp.distributions
tfb = tfp.bijectors

LogNormal = tfd.LogNormal


class GRModel(IRTModel):
    """Implement and store the graded response model for IRT

    Arguments:
        IRTModel {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    response_type = "polytomous"

    def __init__(self, auxiliary_parameterization=True):
        super().__init__()
        self.auxiliary_parameterization = auxiliary_parameterization

    @tf.function
    def grm_model_prob(self, abilities, discriminations, difficulties):
        offsets = difficulties - abilities  # N x D x I x K-1
        scaled = offsets*discriminations
        logits = 1.0/(1+tf.exp(scaled))
        logits = tf.pad(
            logits,
            paddings=(
                [(0, 0)]*(len(tf.shape(logits)) - 1) + [(1, 0)]),
            mode='constant',
            constant_values=1)
        logits = tf.pad(
            logits,
            paddings=(
                [(0, 0)]*(len(tf.shape(logits)) - 1) + [(0, 1)]),
            mode='constant', constant_values=0)
        probs = logits[..., :-1] - logits[..., 1:]

        # weight by discrimination
        # \begin{align}
        #   w_{id} &= \frac{\lambda_{i}^{(d)}}{\sum_d \lambda_{i}^{(d)}}.
        # \end{align}
        weights = (
            discriminations
            / tf.reduce_sum(
                discriminations, axis=-3)[..., tf.newaxis, :, :])
        probs = tf.reduce_sum(probs*weights, axis=-3)
        return probs

    @tf.function
    def grm_model_prob_d(self,
                         abilities,
                         discriminations,
                         difficulties0,
                         ddifficulties
                         ):
        d0 = tf.concat(
            [difficulties0, ddifficulties], axis=-1)
        difficulties = tf.cumsum(d0, axis=-1)
        return self.grm_model_prob(abilities, discriminations, difficulties)

    @tf.function
    def joint_log_prob(self, responses, discriminations,
                       difficulties0, ddifficulties, abilities,
                       xi, eta, mu):
        d0 = tf.concat(
            [difficulties0, ddifficulties], axis=-1)
        difficulties = tf.cumsum(d0, axis=-1)
        return (
            self.log_likelihood(
                responses, discriminations,
                difficulties, abilities)
            + self.joint_log_prior(
                discriminations, difficulties0,
                ddifficulties, abilities, xi, eta, mu))

    @tf.function
    def joint_log_prob_auxiliary(self, responses, discriminations,
                                 difficulties0, ddifficulties, abilities,
                                 xi, eta, mu, xi_a, eta_a):
        d0 = tf.concat(
            [difficulties0, ddifficulties], axis=-1)
        difficulties = tf.cumsum(d0, axis=-1)
        return (
            self.log_likelihood(
                    responses, discriminations,
                    difficulties, abilities)
            + self.joint_log_prior_auxiliary(
                discriminations, difficulties0,
                ddifficulties, abilities, xi**2, eta**2,
                mu, xi_a, eta_a))

    @tf.function
    def log_likelihood(self, responses, discriminations,
                       difficulties, abilities):
        rv_responses = tfd.Independent(
            tfd.Categorical(
                self.grm_model_prob(
                    abilities, discriminations, difficulties)),
            reinterpreted_batch_ndims=2
        )
        return rv_responses.log_prob(responses)

    @tf.function
    def joint_log_prior(self, discriminations, difficulties0,
                        ddifficulties, abilities, xi, eta, mu):
        D = discriminations.shape[0]
        rv_discriminations = tfd.Independent(
            tfd.HalfNormal(scale=eta*xi),
            reinterpreted_batch_ndims=4)
        rv_difficulties0 = tfd.Independent(
            tfd.Normal(
                loc=mu,
                scale=tf.ones_like(mu)),
            reinterpreted_batch_ndims=4
        )
        rv_ddifficulties = tfd.Independent(
            tfd.HalfNormal(
                scale=tf.ones_like(ddifficulties)),
            reinterpreted_batch_ndims=4)
        rv_abilities = tfd.Independent(
            tfd.Normal(
                loc=tf.zeros_like(abilities),
                scale=tf.ones_like(abilities)),
            reinterpreted_batch_ndims=4)
        rv_eta = tfd.Independent(tfd.HalfCauchy(
            loc=tf.zeros_like(eta),
            scale=tf.ones_like(eta)),
            reinterpreted_batch_ndims=4)
        rv_xi = tfd.Independent(tfd.HalfCauchy(
            loc=tf.zeros_like(xi),
            scale=tf.ones_like(xi)),
            reinterpreted_batch_ndims=4)
        rv_mu = tfd.Independent(tfd.Normal(
            loc=tf.zeros_like(mu),
            scale=tf.ones_like(mu)),
            reinterpreted_batch_ndims=4)

        return (rv_discriminations.log_prob(discriminations)
                + rv_difficulties0.log_prob(difficulties0)
                + rv_ddifficulties.log_prob(ddifficulties)
                + rv_abilities.log_prob(abilities)
                + rv_eta.log_prob(eta)
                + rv_xi.log_prob(xi)
                + rv_mu.log_prob(mu))

    def joint_log_prior_auxiliary(self, discriminations, difficulties0,
                                  ddifficulties, abilities, xi, eta, mu,
                                  xi_a, eta_a):
        rv_discriminations = tfd.Independent(
            tfd.HalfNormal(scale=eta*xi),
            reinterpreted_batch_ndims=4)
        rv_difficulties0 = tfd.Independent(
            tfd.Normal(loc=mu, scale=1.),
            reinterpreted_batch_ndims=4)
        rv_ddifficulties = tfd.Independent(
            tfd.HalfNormal(scale=tf.ones_like(ddifficulties)),
            reinterpreted_batch_ndims=4)
        rv_abilities = tfd.Independent(
            tfd.Normal(loc=tf.zeros_like(abilities), scale=1.),
            reinterpreted_batch_ndims=4)
        rv_eta = tfd.Independent(
            tfd.InverseGamma(concentration=0.5*tf.ones_like(
                eta), scale=1.0/eta_a),
            reinterpreted_batch_ndims=4)  # global
        rv_xi = tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5*tf.ones_like(xi),
                scale=1.0/xi_a),
            reinterpreted_batch_ndims=4)  # local
        rv_eta_a = tfd.Independent(
            tfd.InverseGamma(concentration=0.5*tf.ones_like(
                eta_a), scale=tf.ones_like(eta_a)),
            reinterpreted_batch_ndims=4)  # global
        rv_xi_a = tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5*tf.ones_like(xi_a),
                scale=tf.ones_like(xi_a)),
            reinterpreted_batch_ndims=4)  # local
        rv_mu = tfd.Independent(
            tfd.Normal(loc=tf.zeros_like(mu), scale=1.),
            reinterpreted_batch_ndims=4)

        return (rv_discriminations.log_prob(discriminations)
                + rv_difficulties0.log_prob(difficulties0)
                + rv_ddifficulties.log_prob(ddifficulties)
                + rv_abilities.log_prob(abilities)
                + rv_eta.log_prob(eta**2)
                + rv_xi.log_prob(xi**2)
                + rv_mu.log_prob(mu)
                + rv_eta_a.log_prob(eta_a)
                + rv_xi_a.log_prob(xi_a))

    def joint_prior_distribution(self):
        """Joint probability with measure over observations

        Returns:
            tf.distributions.JointDistributionNamed -- Joint distribution
        """
        K = self.response_cardinality
        xi_scale = tf.ones((1, self.dimensions, self.num_items, 1))
        difficulties0 = np.sort(
            np.random.normal(
                size=(1,
                      self.dimensions,
                      self.num_items,
                      self.response_cardinality-1)
            ),
            axis=-1)
        abilities0 = np.random.normal(
            size=(self.num_people, self.dimensions, 1, 1))

        grm_joint_distribution_dict = dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros((1, self.dimensions, self.num_items, 1)),
                    scale=tf.ones((1, self.dimensions, self.num_items, 1))
                ),
                reinterpreted_batch_ndims=4
            ),  # mu
            eta=tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros((1, 1, self.num_items, 1)),
                    scale=tf.ones((1, 1, self.num_items, 1))
                ),
                reinterpreted_batch_ndims=4
            ),  # eta
            xi=tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros((1, self.dimensions, self.num_items, 1)),
                    scale=xi_scale
                ),
                reinterpreted_batch_ndims=4
            ),  # xi
            difficulties0=lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=tf.ones((1, self.dimensions, self.num_items, 1))),
                reinterpreted_batch_ndims=4
            ),  # difficulties0
            discriminations=lambda eta, xi: tfd.Independent(
                tfd.HalfNormal(scale=eta*xi),
                reinterpreted_batch_ndims=4
            ),  # discrimination
            ddifficulties=tfd.Independent(
                tfd.HalfNormal(
                    scale=tf.ones(
                        (1, self.dimensions, self.num_items,
                         self.response_cardinality-2)
                    )
                ),
                reinterpreted_batch_ndims=4
            ),
            abilities=tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros((self.num_people, self.dimensions, 1, 1)),
                    scale=tf.ones((self.num_people, self.dimensions, 1, 1))
                ),
                reinterpreted_batch_ndims=4
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
        surrogate_distribution_dict = {
            'abilities': build_trainable_normal_dist(
                tf.cast(abilities0, tf.float32),
                1e-2*tf.ones((self.num_people, self.dimensions, 1, 1)),
                4
            ),
            'ddifficulties': self.bijectors['ddifficulties'](
                build_trainable_normal_dist(
                    tf.cast(
                        difficulties0[..., 1:]-difficulties0[..., :-1],
                        tf.float32),
                    1e-2*tf.ones(
                        (1,
                         self.dimensions,
                         self.num_items,
                         self.response_cardinality-2
                         )),
                    4
                )
            ),
            'discriminations': self.bijectors['discriminations'](
                build_trainable_normal_dist(
                    tf.ones((1, self.dimensions, self.num_items, 1)),
                    1e-2*tf.ones((1, self.dimensions, self.num_items, 1)),
                    4
                )
            ),
            'mu': build_trainable_normal_dist(
                tf.ones((1, self.dimensions, self.num_items, 1)),
                1e-2*tf.ones((1, self.dimensions, self.num_items, 1)),
                4
            ),
            'eta': self.bijectors['eta'](
                build_trainable_InverseGamma_dist(
                    0.5*tf.ones((1, 1, self.num_items, 1)),
                    tf.ones((1, 1, self.num_items, 1)),
                    4
                )
            ),
            'xi': self.bijectors['xi'](
                build_trainable_InverseGamma_dist(
                    0.5*tf.ones((1, self.dimensions, self.num_items, 1)),
                    tf.ones((1, self.dimensions, self.num_items, 1)),
                    4
                )
            ),
            'difficulties0': build_trainable_normal_dist(
                tf.ones((1, self.dimensions, self.num_items, 1)),
                1e-2*tf.ones((1, self.dimensions, self.num_items, 1)),
                4
            )
        }

        if self.auxiliary_parameterization:
            grm_joint_distribution_dict["xi_a"] = tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((1, self.dimensions, self.num_items, 1)),
                    1.0/tf.math.square(xi_scale)
                ),
                reinterpreted_batch_ndims=4
            )
            surrogate_distribution_dict["xi_a"] = self.bijectors['xi_a'](
                build_trainable_InverseGamma_dist(
                    0.5*tf.ones((1, self.dimensions, self.num_items, 1)),
                    tf.ones((1, self.dimensions, self.num_items, 1)),
                    4
                )
            )

            grm_joint_distribution_dict["xi"] = lambda xi_a: tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((1, self.dimensions, self.num_items, 1)),
                    1.0/xi_a
                ),
                reinterpreted_batch_ndims=4
            )

            surrogate_distribution_dict["xi"] = self.bijectors['xi'](
                build_trainable_InverseGamma_dist(
                    0.5*tf.ones((1, self.dimensions, self.num_items, 1)),
                    tf.ones((1, self.dimensions, self.num_items, 1)),
                    4
                )
            )

            grm_joint_distribution_dict["eta_a"] = tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((1, 1, self.num_items, 1)),
                    tf.ones((1, 1, self.num_items, 1))
                ),
                reinterpreted_batch_ndims=4
            )

            surrogate_distribution_dict["eta_a"] = self.bijectors['eta_a'](
                build_trainable_InverseGamma_dist(
                    0.5*tf.ones((1, 1, self.num_items, 1)),
                    tf.ones((1, 1, self.num_items, 1)),
                    4
                )
            )

            grm_joint_distribution_dict["eta"] = lambda eta_a: tfd.Independent(
                tfd.InverseGamma(
                    0.5*tf.ones((1, 1, self.num_items, 1)),
                    1.0/eta_a
                ),
                reinterpreted_batch_ndims=4
            )

            surrogate_distribution_dict["eta"] = self.bijectors['eta'](
                build_trainable_InverseGamma_dist(
                    0.5*tf.ones((1, 1, self.num_items, 1)),
                    tf.ones((1, 1, self.num_items, 1)),
                    4
                )
            )

            grm_joint_distribution_dict["discriminations"] = lambda eta, xi: tfd.Independent(
                tfd.HalfNormal(scale=tf.sqrt(eta*xi)),
                reinterpreted_batch_ndims=4
            )

        return (tfd.JointDistributionNamed(grm_joint_distribution_dict),
                tfd.JointDistributionNamed(surrogate_distribution_dict))

    def set_calibration_expectations(self):
        self.surrogate_sample = self.surrogate_posterior.sample(1000)
        self.calibrated_traits = tf.reduce_mean(
            self.surrogate_sample['abilities'], axis=0
        )
        self.calibrated_traits_sd = tf.math.reduce_std(
            self.surrogate_sample['abilities'], axis=0
        )
        self.calibrated_discriminations = tf.reduce_mean(
            self.surrogate_sample['discriminations'], axis=0
        )
        self.calibrated_discriminations_sd = tf.math.reduce_std(
            self.surrogate_sample['discriminations'], axis=0
        )
        self.calibrated_difficulties0 = tf.reduce_mean(
            self.surrogate_sample['difficulties0'], axis=0
        )

        self.calibrated_ddifficulties = tf.reduce_mean(
            self.surrogate_sample['ddifficulties'], axis=0
        )
        self.calibrated_difficulties = tf.cumsum(
            tf.concat([
                self.calibrated_difficulties0,
                self.calibrated_ddifficulties
            ], axis=-1), axis=-1
        )

        self.calibrated_trait_scale = tf.math.reduce_std(
            self.calibrated_traits, axis=-4)
        self.calibrated_trait_loc = tf.reduce_mean(
            self.calibrated_traits, axis=-4
        )

    def create_distributions(self):
        """Create the relevant prior distributions
        and the surrogate distribution
        """

        self.bijectors = {
            k: tfp.bijectors.Identity()
            for
            k
            in
            ['abilities', 'mu', 'difficulties0']}

        self.bijectors['eta'] = tfp.bijectors.Softplus()
        self.bijectors['xi'] = tfp.bijectors.Softplus()
        self.bijectors['discriminations'] = tfp.bijectors.Softplus()
        self.bijectors['ddifficulties'] = tfp.bijectors.Softplus()
        if self.auxiliary_parameterization:
            self.bijectors['eta_a'] = tfp.bijectors.Softplus()
            self.bijectors['xi_a'] = tfp.bijectors.Softplus()

        (
            self.weighted_likelihood,
            self.surrogate_posterior
        ) = self.joint_prior_distribution()

        event_shape = self.weighted_likelihood.event_shape_tensor()
        del event_shape['x']
        variable_names = event_shape.keys()

        """
        self.surrogate_posterior = build_factored_surrogate_posterior(
            event_shape=event_shape,
            constraining_bijectors=self.bijectors
        )
        """
        self.set_calibration_expectations()

    def score(self, responses, samples=150):
        responses = tf.cast(responses, np.int32)
        """Compute expections by importance sampling

        Arguments:
            responses {[type]} -- [description]

        Keyword Arguments:
            samples {int} -- Number of samples to use (default: {1000})
        """
        sampling_rv = tfd.Independent(
            tfd.Normal(
                loc=self.calibrated_trait_loc,
                scale=self.calibrated_trait_scale*3
            ),
            reinterpreted_batch_ndims=3
        )
        trait_samples = tf.expand_dims(
            sampling_rv.sample(samples),
            axis=-4
        )

        sample_log_p = sampling_rv.log_prob(trait_samples)

        response_probs = self.grm_model_prob_d(
            abilities=tf.expand_dims(trait_samples, 1),
            discriminations=self.surrogate_sample[
                'discriminations'
            ][tf.newaxis, ...],
            difficulties0=self.surrogate_sample[
                'difficulties0'
            ][tf.newaxis, ...],
            ddifficulties=self.surrogate_sample[
                'ddifficulties'
            ][tf.newaxis, ...]
        )

        response_probs = tf.reduce_mean(response_probs, axis=-4)

        response_rv = tfd.Independent(
            tfd.Categorical(response_probs),
            reinterpreted_batch_ndims=1
        )
        lp = response_rv.log_prob(responses)
        l_w = lp - sample_log_p
        l_w = l_w - tf.reduce_max(l_w, axis=0)[tf.newaxis, :]
        w = tf.math.exp(l_w)/tf.reduce_sum(tf.math.exp(l_w), axis=0)
        mean = tf.reduce_sum(
            w[..., tf.newaxis, tf.newaxis, tf.newaxis]*trait_samples,
            axis=0)
        mean2 = tf.math.reduce_sum(
            w[..., tf.newaxis, tf.newaxis, tf.newaxis]*trait_samples**2,
            axis=0)
        std = tf.sqrt(mean2-mean**2)
        return mean, std

    def loss(self, responses, scores):
        pass

    @tf.function
    def unormalized_log_prob_list(self, *x):
        return self.unormalized_log_prob(
            **tf.nest.pack_sequence_as(
                self.surrogate_sample,
                x
            ))

    @tf.function
    def unormalized_log_prob(self, **x):
        if self.auxiliary_parameterization:
            return self.joint_log_prob_auxiliary(
                **x,
                responses=self.calibration_data
            )
        else:
            return self.joint_log_prob(
                **x,
                responses=self.calibration_data
            )


def main():
    pass


if __name__ == "__main__":
    main()
