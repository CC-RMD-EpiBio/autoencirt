#!/usr/bin/env python3

import jax.numpy as jnp
import numpy as np
from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from jax.scipy.special import xlogy
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from autoencirt.irt import IRTModel


def make_shifted_softplus(min_value, hinge_softness=1.0, name="shifted_softplus"):
    """Creates a Softplus bijector with a specified minimum value."""
    # The chain of bijectors is applied from right to left:
    # 1. Softplus transforms input from (-inf, inf) to (0, inf).
    # 2. Shift adds `min_value`, transforming the range to (min_value, inf).
    return tfb.Chain(
        [tfb.Softplus(hinge_softness=hinge_softness), tfb.Shift(shift=min_value)],
        name=name,
    )


class GRModel(IRTModel):
    """Implement and store the graded response model for IRT

    Arguments:
        IRTModel {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    response_type = "polytomous"

    def __init__(self, *args, **kwargs):
        super(GRModel, self).__init__(*args, **kwargs)
        self.create_distributions()

    def grm_model_prob(self, abilities, discriminations, difficulties):
        if self.include_independent:
            abilities = jnp.pad(
                abilities,
                [(0, 0)] * (len(discriminations.shape) - 3) + [(1, 0)] + [(0, 0)] * 2,
            )
        offsets = difficulties - abilities  # N x D x I x K-1
        scaled = offsets * discriminations
        logits = 1.0 / (1 + jnp.exp(scaled))
        logits = jnp.pad(
            logits,
            ([(0, 0)] * (len(logits.shape) - 1) + [(1, 0)]),
            mode="constant",
            constant_values=1,
        )
        logits = jnp.pad(
            logits,
            ([(0, 0)] * (len(logits.shape) - 1) + [(0, 1)]),
            mode="constant",
            constant_values=0,
        )
        probs = logits[..., :-1] - logits[..., 1:]

        # weight by discrimination
        # \begin{align}
        #   w_{id} &= \frac{\lambda_{i}^{(d)}}{\sum_d \lambda_{i}^{(d)}}.
        # \end{align}
        weights = (
            jnp.abs(discriminations) ** self.weight_exponent
            / jnp.sum(jnp.abs(discriminations) ** self.weight_exponent, axis=-3)[
                ..., jnp.newaxis, :, :
            ]
        )
        probs = jnp.sum(probs * weights, axis=-3)
        return probs

    def grm_model_prob_d(
        self, abilities, discriminations, difficulties0, ddifficulties
    ):
        d0 = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(d0, axis=-1)
        return self.grm_model_prob(abilities, discriminations, difficulties)

    def predictive_distribution(
        self,
        data,
        discriminations,
        difficulties0,
        ddifficulties,
        abilities,
        **kwargs
    ):
        ddifficulties = jnp.where(
            ddifficulties < 1e-1, 1e-1 * jnp.ones_like(ddifficulties), ddifficulties
        )
        # check if responses are a batch
        difficulties = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(difficulties, axis=-1)

        # gather abilities and item parameters corresponding to responses

        rank = len(abilities.shape)
        batch_shape = abilities.shape[: (rank - 4)]
        batch_ndims = len(batch_shape)

        people = data[self.person_key].astype(jnp.int32)
        choices = jnp.concat([data[i][:, jnp.newaxis] for i in self.item_keys], axis=-1)

        bad_choices = (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)

        choices = jnp.where(bad_choices, jnp.zeros_like(choices), choices)

        for _ in range(batch_ndims):
            choices = choices[jnp.newaxis, ...]

        abilities = abilities[:, people, ...]

        response_probs = self.grm_model_prob(abilities, discriminations, difficulties)
        discrimination_weights = jnp.abs(discriminations) / jnp.sum(
            jnp.abs(discriminations), axis=-3, keepdims=True)

        response_probs = jnp.sum(response_probs*discrimination_weights, axis=-3)
        imputed_lp = jnp.sum(xlogy(response_probs, response_probs), axis=-1)

        rv_responses = tfd.Categorical(probs=response_probs)

        log_probs = rv_responses.log_prob(choices)
        log_probs = jnp.where(
            bad_choices[jnp.newaxis, ...], imputed_lp, log_probs
        )

        log_probs = jnp.sum(log_probs, axis=-1)
        # log_probs = jnp.sum(log_probs, axis=-1)

        return {
            "log_likelihood": log_probs,
            "discriminations": discriminations,
            "rv": rv_responses,
        }

    def log_likelihood(
        self,
        data,
        discriminations,
        difficulties0,
        ddifficulties,
        abilities,
        *args,
        **kwargs
    ):
        prediction = self.predictive_distribution(
            data,
            discriminations,
            difficulties0,
            ddifficulties,
            abilities,
            *args,
            **kwargs
        )
        return prediction["log_likelihood"]

    def create_distributions(self, grouping_params=None):
        """Joint probability with measure over observations
        groupings: Vector of same size as the data

        Returns:
            jnp.distributions.JointDistributionNamed -- Joint distribution
        """
        self.bijectors = {
            k: tfb.Identity() for k in ["abilities", "mu", "difficulties0"]
        }

        self.bijectors["eta"] = tfb.Softplus()
        self.bijectors["kappa"] = tfb.Softplus()

        self.bijectors["discriminations"] = tfb.Softplus()
        self.bijectors["ddifficulties"] = tfb.Softplus() # make_shifted_softplus(1e-3)

        self.bijectors["eta_a"] = tfb.Softplus()
        self.bijectors["kappa_a"] = tfb.Softplus()
        self.bijectors["xi"] = tfb.Softplus()

        K = self.response_cardinality
        difficulties0 = np.sort(
            np.random.normal(size=(1, self.dimensions, self.num_items, K - 1)), axis=-1
        )

        grm_joint_distribution_dict = dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),  # mu
            difficulties0=lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),  # difficulties0
            # discriminations=(
            #    lambda eta, xi, kappa: tfd.Independent(
            #        tfd.HalfNormal(scale=eta*xi*kappa),
            #        reinterpreted_batch_ndims=4
            #    )) if self.positive_discriminations else (
            #    lambda eta, xi, kappa: tfd.Independent(
            #        tfd.Normal(
            #            loc=jnp.zeros(
            #                (1, self.dimensions, self.num_items, 1),
            #                dtype=self.dtype),
            #            scale=eta*xi*kappa),
            #        reinterpreted_batch_ndims=4
            #    )),  # discrimination
            discriminations=(
                (
                    lambda eta, kappa: tfd.Independent(
                        AbsHorseshoe(scale=eta * kappa), reinterpreted_batch_ndims=4
                    )
                )
                if self.positive_discriminations
                else (
                    lambda eta, xi, kappa: tfd.Independent(
                        tfd.Horseshoe(
                            loc=jnp.zeros(
                                (1, self.dimensions, self.num_items, 1),
                                dtype=self.dtype,
                            ),
                            scale=eta * kappa,
                        ),
                        reinterpreted_batch_ndims=4,
                    )
                )
            ),  # discrimination
            ddifficulties=tfd.Independent(
                tfd.HalfNormal(
                    scale=jnp.ones(
                        (
                            1,
                            self.dimensions,
                            self.num_items,
                            self.response_cardinality - 2,
                        ),
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=4,
            ),
            eta=tfd.Independent(
                tfd.HalfNormal(
                    scale=self.eta_scale
                    * jnp.ones((1, 1, self.num_items, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            ),
            # xi=tfd.Independent(
            #    tfd.HalfCauchy(
            #        loc=jnp.zeros(
            #            (1, self.dimensions, self.num_items, 1),
            #            dtype=self.dtype),
            #        scale=1.
            #    ),
            #    reinterpreted_batch_ndims=4
            # ),  # xi
            kappa=lambda kappa_a: tfd.Independent(
                SqrtInverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    1.0 / kappa_a,
                ),
                reinterpreted_batch_ndims=4,
            ),
            kappa_a=tfd.Independent(
                tfd.InverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype)
                    / self.kappa_scale**2,
                ),
                reinterpreted_batch_ndims=4,
            ),
        )
        if grouping_params is not None:
            grm_joint_distribution_dict["probs"] = tfd.Independent(
                tfd.Dirichlet(jnp.cast(grouping_params, self.dtype)),
                reinterpreted_batch_ndims=1,
            )
            grm_joint_distribution_dict["mu_ability"] = lambda sigma: tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros((self.dimensions, self.num_groups), self.dtype),
                    scale=sigma,
                ),
                reinterpreted_batch_ndims=2,
            )
            self.bijectors["sigma"] = tfb.Softplus()
            grm_joint_distribution_dict["sigma"] = tfd.Independent(
                tfd.HalfNormal(
                    scale=0.5 * jnp.ones((self.dimensions, self.num_groups), self.dtype)
                ),
                reinterpreted_batch_ndims=2,
            )

            grm_joint_distribution_dict["abilities"] = (
                lambda probs, mu_ability, sigma: tfd.Independent(
                    tfd.Mixture(
                        cat=tfd.Categorical(probs=probs),
                        components=[
                            tfd.Independent(
                                tfd.Normal(
                                    loc=(
                                        jnp.squeeze(
                                            mu_ability[..., jnp.newaxis, :, 0:1]
                                            + jnp.zeros(
                                                shape=(
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                dtype=self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                    scale=(
                                        jnp.squeeze(
                                            sigma[..., jnp.newaxis, :, 0:1]
                                            + jnp.zeros(
                                                shape=(
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                dtype=self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                ),
                                reinterpreted_batch_ndims=3,
                            ),
                            tfd.Independent(
                                tfd.Normal(
                                    loc=(
                                        jnp.squeeze(
                                            mu_ability[..., jnp.newaxis, :, 1:2]
                                            + jnp.zeros(
                                                (
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                    scale=(
                                        jnp.squeeze(
                                            sigma[..., jnp.newaxis, :, 1:2]
                                            + jnp.zeros(
                                                (
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                ),
                                reinterpreted_batch_ndims=3,
                            ),
                        ],
                    ),
                    reinterpreted_batch_ndims=1,
                )
            )
        else:
            grm_joint_distribution_dict["abilities"] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        (
                            self.num_people,
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            1,
                            1,
                        ),
                        dtype=self.dtype,
                    ),
                    scale=jnp.ones(
                        (
                            self.num_people,
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            1,
                            1,
                        ),
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=4,
            )


        self.joint_prior_distribution = tfd.JointDistributionNamed(
            grm_joint_distribution_dict
        )

        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(self.joint_prior_distribution)
        )
        self.params = self.surrogate_parameter_initializer()

    def score(self, responses, samples=400, mm_iterations=10):
        responses = jnp.cast(responses, jnp.int32)
        """Compute expections by importance sampling

        Arguments:
            responses {[type]} -- [description]

        Keyword Arguments:
            samples {int} -- Number of samples to use (default: {1000})
        """
        sampling_rv = tfd.Independent(
            tfd.Normal(
                loc=jnp.mean(self.calibrated_expectations["abilities"], axis=0),
                scale=jnp.std(
                    self.calibrated_expectations["abilities"], axis=0
                ),
            ),
            reinterpreted_batch_ndims=2,
        )
        trait_samples = sampling_rv.sample(samples)

        sample_log_p = sampling_rv.log_prob(trait_samples)

        response_probs = self.grm_model_prob_d(
            abilities=trait_samples[..., jnp.newaxis, jnp.newaxis, :, :, :],
            discriminations=jnp.expand_dims(self.surrogate_sample["discriminations"], 0),
            difficulties0=jnp.expand_dims(self.surrogate_sample["difficulties0"], 0),
            ddifficulties=jnp.expand_dims(self.surrogate_sample["ddifficulties"], 0),
        )

        response_probs = jnp.mean(response_probs, axis=-4)

        response_rv = tfd.Independent(
            tfd.Categorical(probs=response_probs), reinterpreted_batch_ndims=1
        )
        lp = response_rv.log_prob(responses)
        l_w = lp[..., jnp.newaxis] - sample_log_p[:, jnp.newaxis, :]
        # l_w = l_w - jnp.reduce_max(l_w, axis=0, keepdims=True)
        w = jnp.math.exp(l_w) / jnp.sum(jnp.math.exp(l_w), axis=0, keepdims=True)
        mean = jnp.sum(w * trait_samples[:, jnp.newaxis, :, 0, 0], axis=0)
        mean2 = jnp.sum(
            w * trait_samples[:, jnp.newaxis, :, 0, 0] ** 2, axis=0
        )
        std = jnp.sqrt(mean2 - mean**2)
        return mean, std, w, trait_samples
    
    def fit_dim(self, *args, dim: int,  **kwargs):
        if dim >= self.dimensions:
            raise ValueError("Dimension to fit must be less than model dimensions")
        # Only optimize parameters for the selected dimension `dim`
        optimizing_keys = [
            k
            for k in self.params.keys()
            if (
            not any(
                k.startswith(prefix)
                and not k.startswith(f"{prefix}{dim}")
                for prefix in [
                "discriminations_",
                "ddifficulties_",
                "difficulties0_",
                "kappa_",
                "kappa_a_",
                ]
            )
            )
        ]
        return self.fit(*args, **kwargs, optimize_keys=optimizing_keys)
    def unormalized_log_prob(self, data, prior_weight=1., **params):
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]
        weights = prediction["discriminations"]
        weights = weights / jnp.sum(weights, axis=-3, keepdims=True)
        entropy = -xlogy(weights, weights) / params["eta"]
        entropy = jnp.sum(entropy, axis=[-1, -2, -3, -4])

        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )
        min_val = jnp.min(finite_portion) - 1.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )
        return jnp.cast(prior_weight, log_prior.dtype) * (log_prior - entropy) + jnp.sum(
            log_likelihood, axis=-1
        )

    def dumpyaml(self, item_info, scale_info):
        """

        Args:
            item_info (_type_): _description_
            scale_info (_type_): _description_
        """

        pass


def main():
    pass


if __name__ == "__main__":
    main()
