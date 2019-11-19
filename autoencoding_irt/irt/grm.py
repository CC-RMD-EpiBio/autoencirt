#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util
import numpy as np
import pandas as pd

tfd = tfp.distributions

tfd = tfp.distributions
tfb = tfp.bijectors


class IRTModel(object):
    response_type = None
    calibration_data = None
    num_people = None
    num_items = None
    response_data = None
    response_cardinality = None
    dimensions = 1

    def __init__(self):
        pass

    def set_dimension(self, dim):
        self.dimensions = dim

    def load_data(self, response_data):
        self.response_data = response_data
        self.num_people = response_data.shape[0]
        self.num_items = response_data.shape[1]
        self.response_cardinality = int(response_data.max()) + 1
        if int(response_data.min()) == 1:
            print("Warning: responses do not appear to be from zero")


class GRModel(IRTModel):
    """Implement and store the graded response model for IRT

    Arguments:
        IRTModel {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    data = None
    response_type = "polytomous"

    def __init__(self):
        super().__init__()

    def load_data(self):
        pass

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
            weights = discriminations / \
                tf.reduce_sum(discriminations, axis=1)[:, tf.newaxis, :]
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
        return grm_model_prob(abilities, discriminations, difficulties)

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
        rv_responses = tfd.Categorical(grm_model_prob(
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

    def joint_distribution():

        K = self.response_cardinality

        grm_joint_distribution = tfd.JointDistributionNamed(dict(
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
                tfd.HalfNormal(scale=eta[..., tf.newaxis, :]*xi),
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
                    grm_model_prob_d(
                        abilities,
                        discriminations,
                        difficulties0,
                        ddifficulties
                        ),
                    validate_args=True),
                reinterpreted_batch_ndims=2
            )
        ))
        return grm_joint_distribution


def main():
    grm = GRModel()
    grm.set_dimension(2)


if __name__ == "__main__":
    main()
