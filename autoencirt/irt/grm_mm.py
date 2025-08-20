import jax
import numpy as np
from jax import numpy as jnp
from jax.lax.linalg import tridiagonal_solve
from jax.nn import one_hot, sigmoid
from jax.random import categorical, normal
from jax.scipy.special import xlogy
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

SMALLEST_DIFFICULTY_GAP = 1e-3

dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
ddsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))


def p(abilities, difficulties, discriminations):
    p_cum = sigmoid(discriminations * (abilities - difficulties))
    p_cum = jnp.pad(
        p_cum, ([[0, 0]] * (len(difficulties.shape) - 1) + [[1, 0]]), constant_values=1
    )
    p_cum = jnp.pad(
        p_cum, ([[0, 0]] * (len(difficulties.shape) - 1) + [[0, 1]]), constant_values=0
    )
    return p_cum[..., :-1] - p_cum[..., 1:]


def rescale_params(abilities, difficulties, discriminations):
    ability_mean = jnp.mean(abilities, axis=-3, keepdims=True)
    ability_std = jnp.std(abilities, axis=-3, keepdims=True)
    if chk_nan(ability_mean):
        pass
    if chk_nan(ability_std):
        pass
    abilities = (abilities - ability_mean) / ability_std

    difficulties = (difficulties - ability_mean) / ability_std
    discriminations *= ability_std
    return abilities, difficulties, discriminations


def nlp_and_majorize(ability, discrimination, difficulties):
    def _nlp(abilities, discriminations, diff):
        return -jnp.log(p(abilities, diff, discriminations))

    def q(abilities, discriminations, diff):

        pass

    return _nlp


#
# N x I x K
# Evaluate the function, and the derivatives of the function wrt theta_n,
# lambda_i, tau_k, tau_k-1
def p_nik(abilities, difficulties, discriminations):  # dimensio
    N = abilities.shape[0]
    I = discriminations.shape[1]
    K = difficulties.shape[-1] + 1
    mu = discriminations * (abilities - difficulties)
    p_cum = sigmoid(mu)  # Pr(x>=k)
    # first partials, will be N x I x K x d where d is the dimension of the parameter
    # d_cum_abilites will be N x I x K x N
    first = {}
    first["abilities"] = dsigmoid(mu) * discriminations  # theta is abilities
    first["difficulties"] = -dsigmoid(mu) * discriminations  # tau is difficulty
    first["discriminations"] = dsigmoid(mu) * (abilities - difficulties)

    # second partials for the cumulative probs

    second = {}
    second[("abilities", "abilities")] = ddsigmoid(mu) * discriminations**2
    # Diagonal terms
    second[("discriminations", "discriminations")] = (
        ddsigmoid(mu) * (abilities - difficulties) ** 2
    )
    second[("difficulties", "difficulties")] = ddsigmoid(mu) * discriminations**2
    second[("abilities", "difficulties")] = -ddsigmoid(mu) * discriminations**2
    second[("abilities", "discriminations")] = dsigmoid(mu) + ddsigmoid(mu) * mu
    second[("difficulties", "discriminations")] = -dsigmoid(mu) - ddsigmoid(mu) * mu

    p_cum = jnp.pad(p_cum, ((0, 0), (0, 0), (1, 0)), constant_values=1)
    p_cum = jnp.pad(p_cum, ((0, 0), (0, 0), (0, 1)), constant_values=0)

    p = p_cum[..., :-1] - p_cum[..., 1:]
    if np.isnan(jnp.mean(p)):
        pass

    # compute derivatives

    def pad_grad_p_(mat):
        _mat = jnp.pad(mat, ((0, 0), (0, 0), (1, 1)), constant_values=0)
        return _mat[..., :-1] - _mat[..., 1:]

    gradients = {
        "abilities": pad_grad_p_(first["abilities"]),
        "discriminations": pad_grad_p_(first["discriminations"]),
        "tau1": (
            jnp.pad(first["difficulties"][..., :-1], ((0, 0), (0, 0), (1, 0)))
            - jnp.pad(first["difficulties"][..., 1:], ((0, 0), (0, 0), (0, 1)))
        ),
    }

    gradients["difficulties0"] = jnp.pad(
        first["difficulties"], ((0, 0), (0, 0), (1, 0)), constant_values=0
    )
    gradients["difficulties1"] = -jnp.pad(
        first["difficulties"], ((0, 0), (0, 0), (0, 1)), constant_values=0
    )
    

    gradients["delta"] = (
        jnp.tril(jnp.ones((1, 1, K-2, K-2)),k=-1)*first["difficulties"][..., :-1, jnp.newaxis]
        - jnp.tril(jnp.ones((1, 1, K-2, K-2)),k=-0)*first["difficulties"][..., 1:, jnp.newaxis]
        )

    grad_log_p = {k: v / (p + 1e-10) for k, v in gradients.items()}

    grad2_p = {
        ("abilities", "abilities"): pad_grad_p_(second[("abilities", "difficulties")]),
        ("discriminations", "discriminations"): pad_grad_p_(
            second[("discriminations", "discriminations")]
        ),
        ("abilities", "discriminations"): pad_grad_p_(
            second[("abilities", "discriminations")]
        ),
        ("difficulties0", "difficulties1"): jnp.zeros_like(p),
        ("difficulties0", "difficulties0"): jnp.pad(
            second[("difficulties", "difficulties")],
            ((0, 0), (0, 0), (1, 0)),
            constant_values=0,
        ),
        ("difficulties1", "difficulties1"): -jnp.pad(
            second[("difficulties", "difficulties")],
            ((0, 0), (0, 0), (0, 1)),
            constant_values=0,
        ),
        ("abilities", "difficulties0"): jnp.pad(
            second[("abilities", "difficulties")],
            ((0, 0), (0, 0), (1, 0)),
            constant_values=0,
        ),
        ("abilities", "difficulties1"): -jnp.pad(
            second[("abilities", "difficulties")],
            ((0, 0), (0, 0), (0, 1)),
            constant_values=0,
        ),
        ("discriminations", "difficulties0"): jnp.pad(
            second[("difficulties", "discriminations")],
            ((0, 0), (0, 0), (1, 0)),
            constant_values=0,
        ),
        ("discriminations", "difficulties1"): -jnp.pad(
            second[("difficulties", "discriminations")],
            ((0, 0), (0, 0), (0, 1)),
            constant_values=0,
        ),
    }
    grad2_log_p = {
        k: (v * p - gradients[k[0]] * gradients[k[1]]) / (p + 1e-8) ** 2
        for k, v in grad2_p.items()
    }

    return {
        "N": N,
        "I": I,
        "K": K,
        "p": p,
        "log(p)": jnp.log(p),
        "grad(p)": gradients,
        "grad(log(p))": grad_log_p,
        "grad(grad(p))": grad2_p,
        "grad(grad(log(p)))": grad2_log_p,
    }


def a_b_c_reparam(abilities, difficulties, discriminations):
    vals = p_nik(abilities, difficulties, discriminations)

    pass


def a_b_c(abilities, difficulties, discriminations):
    vals = p_nik(abilities, difficulties, discriminations)
    vars = ["abilities", "discriminations", "difficulties0", "difficulties1"]

    # H = [[ vals["grad(grad(log(p)))"].get((l1, l2), vals["grad(grad(log(p)))"].get((l2, l1))) for l1 in vars] for l2 in vars]
    def retrive_H_term(a, b):
        try:
            v = vals["grad(grad(log(p)))"][(a, b)]
        except KeyError:
            v = vals["grad(grad(log(p)))"][(b, a)]
        return -v

    H = [[retrive_H_term(l1, l2) for l1 in vars] for l2 in vars]
    H = jnp.transpose(jnp.array(H), (2, 3, 4, 0, 1))
    Heigen = jnp.sum(jnp.abs(H), axis=[-1, -2])
    # Heigen = jnp.real(jnp.linalg.eig(H)[0])
    a = discriminations**2 / 4 + jnp.zeros_like(Heigen)  # Heigen*1.5
    if np.isnan(jnp.mean(a)):
        print(discriminations)
        # print(difficulties)
        pass

    b4 = -vals["grad(log(p))"]["discriminations"] - a * discriminations**2
    tau0 = jnp.pad(
        difficulties, ((0, 0), (0, 0), (1, 0)), constant_values=0
    )  # \tau_{k-1}
    tau = jnp.pad(difficulties, ((0, 0), (0, 0), (0, 1)), constant_values=0)
    b1 = -vals["grad(log(p))"]["abilities"] - a * abilities**2
    b2 = -vals["grad(log(p))"]["difficulties0"] - a * tau0**2
    b3 = -vals["grad(log(p))"]["difficulties1"] - a * tau**2

    c = -jnp.log(vals["p"]) - (
        a / 2 * (discriminations**2 + tau0**2 + tau**2)
        + b2 * tau0
        + b3 * tau
        + b4 * discriminations
        + b1 * abilities
    )

    if chk_nan(a):
        pass
    if chk_nan(b2):
        pass
    if chk_nan(b3):
        pass
    if chk_nan(b4):
        pass
    return a, b2, b3, b4, c, vals


def ldens_to_density(ldens, grid):
    #
    ndens = -ldens - jnp.min(-ldens, axis=0, keepdims=True)
    pi = jnp.exp(-ndens)
    Z = jnp.trapezoid(pi, grid, axis=0)
    pi /= Z
    mean = jnp.sum(pi * grid, axis=0)
    second = jnp.sum(pi * grid**2, axis=0)
    return pi, mean, second


def ability_step(X, difficulties, discriminations):
    # Get the next expected ability for params_next
    grid = jnp.linspace(-5, 5, 50)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    _p = p(grid, difficulties[jnp.newaxis, ...], discriminations[jnp.newaxis, ...])
    _X = one_hot(X, difficulties.shape[-1] + 1)

    lq = xlogy(_X, _p + 1e-8)
    lq = jnp.sum(lq, axis=[-1, -2])
    _, theta, var = ldens_to_density(lq, grid[..., 0, 0])
    if np.isnan(jnp.mean(theta)):
        pass

    return theta, var


def chk_nan(x):
    return np.isnan(jnp.mean(x))


def params_next(X, abilities, difficulties, discriminations, eta, key):

    a, b2, b3, b4, c, vals = a_b_c(abilities, difficulties, discriminations)
    # calculate next lambda values
    _X = one_hot(X, vals["K"])

    W = _X + (jnp.sum(_X, axis=-1, keepdims=True) < 1).astype(int) * vals["p"]

    discrim_nu = (
        jnp.sum(
            W * (discriminations * a + b4)
            + 2 * discriminations / (1 + discriminations**2),
            axis=[0, 2],
            keepdims=True,
        )
        / vals["N"]
        - eta / discriminations
    )
    if chk_nan(discrim_nu):
        pass
    discrim_den = (
        jnp.sum(
            W * a + 2 * discriminations / (1 + discriminations**2),
            axis=[0, 2],
            keepdims=True,
        )
        / vals["N"]
        + eta / discriminations**2
    )
    if chk_nan(discrim_den):
        pass
    first_select = jnp.zeros_like(difficulties).at[..., :1].set(1)
    last_select = jnp.zeros_like(difficulties).at[..., -1:].set(1)

    gaps = difficulties[..., 1:] - difficulties[..., :-1]

    g = (
        2 * (2 - last_select) * difficulties
        + jnp.sum(
            W[..., :-1] * (2 * a[..., :-1] * difficulties + b3[..., :-1])
            + W[..., 1:] * (2 * a[..., 1:] * difficulties + b2[..., 1:]),
            axis=0,
            keepdims=True,
        )
    ) / vals["N"] - eta * (
        (1 - first_select) * jnp.pad(1 / gaps, ((0, 0), (0, 0), (1, 0)))
        - (1 - last_select) * jnp.pad(1 / gaps, ((0, 0), (0, 0), (0, 1)))
    )
    g += jnp.pad(difficulties[..., 1:], ((0, 0), (0, 0), (0, 1))) + jnp.pad(
        difficulties[..., :-1], ((0, 0), (0, 0), (1, 0))
    )  # TODO CHECK

    Hdiag = 2 * (
        2
        - last_select
        + jnp.sum(W[..., :-1] * a[..., :-1] + W[..., :-1] * a[..., 1:], axis=0)
    ) / vals["N"] + eta * (
        (1 - first_select) * jnp.pad(1 / gaps**2, ((0, 0), (0, 0), (1, 0)))
        - (1 - last_select) * jnp.pad(1 / gaps**2, ((0, 0), (0, 0), (0, 1)))
    )
    Hupper = ((last_select - 1)) / vals["N"] - eta * (1 - last_select) * jnp.pad(
        1 / gaps**2, ((0, 0), (0, 0), (0, 1))
    ) ** 2
    Hlower = jnp.pad(Hupper[..., :-1], ((0, 0), (0, 0), (1, 0)))

    diff_ = tridiagonal_solve(
        Hlower[0, ...], Hdiag[0, ...], Hupper[0, ...], g[0, :, :, jnp.newaxis]
    )[jnp.newaxis, :, :, 0]
    delta = difficulties[..., 1:] - difficulties[..., :-1]

    diff_delta = diff_[..., 1:] - diff_[..., :-1]
    t = min(1e-3, jax.random.uniform(key=key) / jnp.max(diff_delta / delta))

    if jnp.max(t * diff_) > 10:
        # print(jnp.max(t*diff_))
        pass
    # print(t)
    _diff = difficulties - t * diff_
    theta, var = ability_step(X, difficulties, discriminations)
    theta = theta[:, jnp.newaxis, jnp.newaxis]
    _discrim = discriminations - t * discrim_nu / discrim_den
    candidate = p(theta, _diff, _discrim)
    if np.isnan(jnp.sum(candidate)):
        print("skipping")
        lp = xlogy(_X, p(abilities, difficulties, discriminations) + 1e-8)
        lp = jnp.sum(lp)
        return abilities, difficulties, discriminations, lp

    if chk_nan(theta):
        pass
    if chk_nan(_discrim):
        pass

    if chk_nan(_diff):
        pass
    # theta, diff, discrim = rescale_params(theta, diff, discrim)
    lp = xlogy(_X, p(theta, _diff, _discrim) + 1e-8)
    lp = jnp.sum(lp)
    if chk_nan(lp):
        pass
    return theta, _diff, _discrim, lp


def params_next_batched(
    X, abilities, difficulties, discriminations, eta, key, batch_size
):
    indices = jax.random.choice(
        key, np.arange(abilities.shape[0]), [batch_size], replace=False
    )
    _X = one_hot(X, difficulties.shape[-1] + 1)
    _, diff, discrim, _ = params_next(
        X[indices], abilities[indices], difficulties, discriminations, eta, key
    )
    theta_, _ = ability_step(X, diff, discrim)
    lp = xlogy(_X, p(theta_[:, jnp.newaxis, jnp.newaxis], diff, discrim) + 1e-8)
    if chk_nan(lp):
        pass
    return theta_[:, jnp.newaxis, jnp.newaxis], diff, discrim, jnp.sum(lp)


def main():
    N = 10000
    K = 5
    I = 30
    key = jax.random.key(np.random.randint(0, 10000))
    abilities = normal(key, N)[:, jnp.newaxis, jnp.newaxis]
    # abilities = jnp.array([0, 0.5, 0.25])[:, jnp.newaxis, jnp.newaxis]
    difficulties0 = normal(key, (1, I, 1)) - K / 2
    ddifficulties = 0.5 + jnp.abs(normal(key, (1, I, K - 2)))
    difficulties = jnp.concat([difficulties0, ddifficulties], axis=-1)
    difficulties = jnp.cumsum(difficulties, axis=-1)
    discriminations = 0.5 + jnp.abs(normal(key, I))[jnp.newaxis, :, jnp.newaxis]
    probs = p(abilities, difficulties, discriminations)
    # difficulties = np.array([[0, 1, 2, 3], [-2, 0, 3, 4]])[np.newaxis, ...]
    X = categorical(key, logits=jnp.log(probs / (1 - probs)))
    # X = np.array([[2, 1], [3, 2], [0, 1]])
    _abilities = jnp.zeros_like(abilities)
    _discriminations = jnp.ones_like(discriminations)
    _difficulites = jnp.transpose(
        jnp.tile(jnp.linspace(-2, 2, num=K - 1)[:, jnp.newaxis], I)
    )[jnp.newaxis, ...]
    batch_size = 1000
    for j in range(5000):
        _abilities, _difficulites, _discriminations, _lp = params_next_batched(
            X,
            _abilities,
            _difficulites,
            _discriminations,
            1000 / ((1 + j // (N / batch_size))),
            key,
            batch_size,
        )
        if j % 500 == 0:
            print(_discriminations)
            print(_abilities)
            print(_difficulites)

        if j % 50 == 0:
            print(j, -_lp)
    _abilities, _difficulites, _discriminations = rescale_params(
        _abilities, _difficulites, _discriminations
    )

    print("ORIGINAL")
    _ = plt.figure(figsize=(5, 5))
    _ = plt.scatter(discriminations[0, :, 0], _discriminations[0, :, 0])
    _ = plt.title("discriminations")

    _ = plt.figure(figsize=(5, 5))
    _ = plt.scatter(abilities[:, 0, 0], _abilities[:, 0, 0])
    _ = plt.title("abilities")

    _ = plt.figure(figsize=(5, 5))
    _ = plt.scatter(difficulties[0, :, 0], _difficulites[0, :, 0])
    _ = plt.title("first difficulty")
    _ = plt.figure(figsize=(5, 5))
    _ = plt.scatter(difficulties[0, :, -1], _difficulites[0, :, -1])
    _ = plt.title("last difficulty")

    print(discriminations)


if __name__ == "__main__":
    main()
