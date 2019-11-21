
import tensorflow as tf
import tensorflow_probability as tfp
from autoencirt.data.rwa import get_data
from autoencirt.irt import GRModel


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
    weights = discriminations / \
        tf.reduce_sum(discriminations, axis=0)[tf.newaxis, :]
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


def main():
    data = get_data()
    unnormalized_posterior_log_prob = lambda *args, **kwargs: joint_log_prob(
                data, *args, **kwargs)
    grm = GRModel(auxiliary_parameterization=False)
    grm.set_dimension(2)
    grm.load_data(data)
    grm.create_distributions()
    test_state = grm.surrogate_sample.copy()
    test_state['responses'] = tf.cast(data.to_numpy(), tf.float32)
, *    print(unnormalized_posterior_log_prob(**test_state))
    1




if __name__ == "__main__":
    main()
