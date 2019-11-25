#!/usr/bin/env python3

"""
Multidimensional IRT using horseshoe prior without coupling a neural network
"""

import tensorflow as tf
import tensorflow_probability as tfp
from os import path, system
import pickle
import gzip
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

tfd = tfp.distributions

tfd = tfp.distributions
tfb = tfp.bijectors

sparcity_decay_ratio = 0.25
K = 10


@tf.function
def grm_model_prob(abilities, discriminations, difficulties):
    offsets = difficulties[tf.newaxis, :, :, :] - \
        abilities[:, :, tf.newaxis, tf.newaxis]
    scaled = offsets*discriminations[tf.newaxis, :, :, tf.newaxis]
    logits = 1.0/(1+tf.exp(scaled))
    logits = tf.pad(logits, paddings=(
        (0, 0), (0, 0), (0, 0), (1, 0)), mode='constant', constant_values=1)
    logits = tf.pad(logits, paddings=(
        (0, 0), (0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
    probs = logits[:, :, :, :-1] - logits[:, :, :, 1:]

    # weight by discrimination
    # \begin{align}
    #   w_{id} &= \frac{\lambda_{i}^{(d)}}{\sum_d \lambda_{i}^{(d)}}.
    # \end{align}
    weights = discriminations**2 / \
        tf.reduce_sum(discriminations, axis=0)[tf.newaxis, :]**2
    probs = tf.reduce_sum(probs*weights[tf.newaxis, :, :, tf.newaxis], axis=1)
    return probs


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


@tf.function
def joint_log_prob(responses, discriminations, difficulties0, ddifficulties, abilities, xi, eta, mu):
    d0 = tf.concat([difficulties0[:, :, tf.newaxis], ddifficulties], axis=2)
    difficulties = tf.cumsum(d0, axis=2)
    return tf.reduce_sum(log_likelihood(responses, discriminations, difficulties, abilities)) + \
        joint_log_prior(discriminations, difficulties0,
                        ddifficulties, abilities, xi, eta, mu)


@tf.function
def log_likelihood(responses, discriminations, difficulties, abilities):
    rv_responses = tfd.Categorical(grm_model_prob(
        abilities, discriminations, difficulties))
    return rv_responses.log_prob(responses)

# return vector of length D
def dimensional_decay(D):
    pass


@tf.function
def joint_log_prior(discriminations, difficulties0, ddifficulties, abilities, xi, eta, mu):
    D = discriminations.shape[0]
    rv_discriminations = tfd.HalfNormal(scale=eta*xi)
    rv_difficulties0 = tfd.Normal(loc=mu, scale=1.)
    rv_ddifficulties = tfd.HalfNormal(scale=tf.ones_like(ddifficulties))
    rv_abilities = tfd.Normal(loc=tf.zeros_like(abilities), scale=1.)
    rv_eta = tfd.HalfCauchy(loc=tf.zeros_like(
        eta), scale=tf.ones_like(eta))  # global
    rv_xi = tfd.HalfCauchy(loc=tf.zeros_like(xi), scale=(
        sparcity_decay_ratio**tf.cast(tf.range(D)+2, tf.float32))[..., :, tf.newaxis])  # local
    rv_mu = tfd.Normal(loc=tf.zeros_like(mu), scale=1.)

    return tf.reduce_sum(rv_discriminations.log_prob(discriminations)) + \
        tf.reduce_sum(rv_difficulties0.log_prob(difficulties0)) + \
        tf.reduce_sum(rv_ddifficulties.log_prob(ddifficulties)) + \
        tf.reduce_sum(rv_abilities.log_prob(abilities)) + \
        tf.reduce_sum(rv_eta.log_prob(eta)) + \
        tf.reduce_sum(rv_xi.log_prob(xi)) + \
        tf.reduce_sum(rv_mu.log_prob(mu))


# Since MCMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Softplus(),       # R^+ \to R
    tfp.bijectors.Identity(),       # Maps R to R.
    tfp.bijectors.Softplus(),       # Maps R to R.
    tfp.bijectors.Identity(),       # Maps R to R.
    tfp.bijectors.Softplus(),       # R^+ \to R
    tfp.bijectors.Softplus(),       # R^+ \to R
    tfp.bijectors.Identity()       # Maps R to R.
]


@tf.function
def graph_hmc(*args, **kwargs):
    """Compile static graph for tfp.mcmc.sample_chain.
    Since this is bulk of the computation, using @tf.function here
    signifcantly improves performance (empirically about ~5x).
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)


def trace_fn_nuts(_, pkr):
    return (pkr.inner_results.inner_results.is_accepted,
            pkr.inner_results.inner_results.step_size)


def trace_fn_hmc(_, pkr):
    return (pkr.inner_results.inner_results.is_accepted,
            pkr.inner_results.inner_results.accepted_results.step_size)


def main():
    """
    Probabilities for a single person
    discriminations  D (domain) x I (item)
    difficulties D x I x K
    abilities N x D

    returns
    prob N x I x K
    """

    print("GPU Available: ", tf.test.is_gpu_available())

    if not path.exists('RWAS/data.csv'):
        system("wget https://openpsychometrics.org/_rawdata/RWAS.zip")
        system("unzip RWAS.zip")

    data = pd.read_csv('RWAS/data.csv', low_memory=False)
    full_data = data.loc[:, map(lambda x: 'Q'+str(x), list(range(1, 23)))]
    full_data.head()

    Qs = ["The established authorities generally turn out to be right about things, while the radicals and protestors are usually just \"loud mouths\" showing off their ignorance.",
          "Women should have to promise to obey their husbands when they get married.",
          "Our country desperately needs a mighty leader who will do what has to be done to destroy the radical new ways and sinfulness that are ruining us.",
          "Gays and lesbians are just as healthy and moral as anybody else.",
          "It is always better to trust the judgement of the proper authorities in government and religion than to listen to the noisy rabble-rousers in our society who are trying to create doubt in people's minds.",
          "Atheists and others who have rebelled against the established religions are no doubt every bit as good and virtuous as those who attend church regularly.",
          "The only way our country can get through the crisis ahead is to get back to our traditional values, put some tough leaders in power, and silence the troublemakers spreading bad ideas.",
          "There is absolutely nothing wrong with nudist camps.",
          "Our country needs free thinkers who have the courage to defy traditional ways, even if this upsets many people.",
          "Our country will be destroyed someday if we do not smash the perversions eating away at our moral fiber and traditional beliefs.",
          "Everyone should have their own lifestyle, religious beliefs, and sexual preferences, even if it makes them different from everyone else.",
          "The \"old-fashioned ways\" and the \"old-fashioned values\" still show the best way to live.",
          "You have to admire those who challenged the law and the majority's view by protesting for women's abortion rights, for animal rights, or to abolish school prayer.",
          "What our country really needs is a strong, determined leader who will crush evil, and take us back to our true path.",
          "Some of the best people in our country are those who are challenging our government, criticizing religion, and ignoring the \"normal way things are supposed to be done.\"",
          "God's laws about abortion, pornography and marriage must be strictly followed before it is too late, and those who break them must be strongly punished.",
          "There are many radical, immoral people in our country today, who are trying to ruin it for their own godless purposes, whom the authorities should put out of action.",
          "A \"woman's place\" should be wherever she wants to be. The days when women are submissive to their husbands and social conventions belong strictly in the past.",
          "Our country will be great if we honor the ways of our forefathers, do what the authorities tell us to do, and get rid of the \"rotten apples\" who are ruining everything.",
          "There is no \"one right way\" to live life; everybody has to create their own way.",
          "Homosexuals and feminists should be praised for being brave enough to defy \"traditional family values.\"",
          "This country would work a lot better if certain groups of troublemakers would just shut up and accept their group's traditional place in society."]

    

    kf = KFold(n_splits=5)
    kf.get_n_splits(full_data)
    for j, (train_index, test_index) in enumerate(kf.split(data)):
        results = []
        item_responses = full_data.iloc[train_index, :]
        N = item_responses.shape[0]
        I = item_responses.shape[1]
        for D in range(1, 4):
            data = tf.cast(item_responses.head(N).to_numpy(), tf.int32)
            difficulties0 = np.sort(np.random.normal(size=(D, I, K-1)), axis=2)
            abilities0 = np.random.normal(size=(N, D))

            # Define a closure over our joint_log_prob.
            unnormalized_posterior_log_prob = lambda *args: joint_log_prob(
                data, *args)

            # Set the chain's start state.
            initial_chain_state = [
                tf.ones((D, I), name='init_discriminations'),
                # may be causing problems
                tf.cast(difficulties0[:, :, 0],
                        tf.float32, name='init_difficulties0'),
                tf.cast(difficulties0[:, :, 1:]-difficulties0[:,
                                                              :, :-1], tf.float32, name='init_ddifficulties'),
                tf.cast(abilities0, tf.float32, name='init_abilities'),
                tf.ones((D, I), name='init_xi'),
                tf.ones((I), name='init_eta'),
                tf.zeros((D, I), name='init_mu')
            ]

            # Initialize the step_size. (It will be automatically adapted.)

            number_of_steps = 12000
            burnin = 10000
            num_leapfrog_steps = 5
            hmc = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_posterior_log_prob,
                    num_leapfrog_steps=num_leapfrog_steps,
                    step_size=0.1,
                    state_gradients_are_stopped=True),
                bijector=unconstraining_bijectors)

            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=hmc, num_adaptation_steps=10000)

            # with tf.device('/device:GPU:0'):

            [discriminations, difficulties0, ddifficulties,
             abilities, xi, eta, mu], kernel_results = graph_hmc(
                num_results=number_of_steps,
                num_burnin_steps=burnin,
                current_state=initial_chain_state,
                trace_fn=trace_fn_hmc,
                num_steps_between_results=1,
                kernel=kernel)

            [
                discriminations_,
                difficulties0_,
                ddifficulties_,
                abilities_, xi_, eta_, mu_
            ] = [v.numpy() for v in
                 [discriminations,
                  difficulties0,
                  ddifficulties,
                  abilities,
                  xi,
                  eta, mu]]

            results += [
                {
                    'train_index': train_index,
                    'test_index': test_index,
                    'D': D,
                    'discriminations': discriminations_,
                    'difficulties0': difficulties0_,
                    'ddifficulties_': ddifficulties_,
                    'abilities_': abilities_,
                    'kernel_results': kernel_results,
                    'xi': xi_,
                    'eta': eta_,
                    'mu': mu_,
                    'burnin': burnin
                }
            ]

        with gzip.open("irt_cv_mcmc_results" + str(j) + ".gz", 'wb') as outfile:
            pickle.dump([results], outfile)


if __name__ == '__main__':
    main()
