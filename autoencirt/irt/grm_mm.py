from jax import numpy as jnp
from jax.lax.linalg import tridiagonal_solve
from jax.nn import one_hot, sigmoid
from jax.random import PRNGKey, categorical, normal

SMALLEST_DIFFICULTY_GAP = 1e-3

dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
ddsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))


def p(abilities, difficulties, discriminations):
    p_cum = sigmoid(discriminations * (abilities - difficulties))
    p_cum = jnp.pad(p_cum, ((0, 0), (0, 0), (1, 0)), constant_values=1)
    p_cum = jnp.pad(p_cum, ((0, 0), (0, 0), (0, 1)), constant_values=0)
    return p_cum[..., :-1] - p_cum[..., 1:]


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

    # compute derivatives

    def pad_grad_p_(mat):
        _mat = jnp.pad(mat, ((0, 0), (0, 0), (1, 1)), constant_values=0)
        return _mat[..., :-1] - _mat[..., 1:]

    gradients = {
        "abilities": pad_grad_p_(first["abilities"]),
        "discriminations": pad_grad_p_(first["discriminations"]),
    }
    gradients["difficulties0"] = jnp.pad(
        first["difficulties"], ((0, 0), (0, 0), (1, 0)), constant_values=0
    )
    gradients["difficulties1"] = jnp.pad(
        first["difficulties"], ((0, 0), (0, 0), (0, 1)), constant_values=0
    )

    grad_log_p = {k: v / p for k, v in gradients.items()}

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
        k: (v * p - gradients[k[0]] * gradients[k[1]]) / p**2
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


def a_b(abilities, difficulties, discriminations):
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
    Heigen = jnp.real(jnp.linalg.eig(H)[0])
    a = jnp.max(Heigen, axis=-1) + 1

    b4 = -vals["grad(log(p))"]["discriminations"] - a * discriminations**2
    tau0 = jnp.pad(
        difficulties[..., :-1], ((0, 0), (0, 0), (1, 0)), constant_values=0
    )  # \tau_{k-1}


    b2 = jnp.pad(
        -vals["grad(log(p))"]["difficulties0"][..., :-1] - a[..., :-1] * tau0**2,
        ((0, 0), (0, 0), (1, 0)),
    )
    b3 = jnp.pad(
        -vals["grad(log(p))"]["difficulties1"][..., 1:] - a[..., 1:] * difficulties**2,
        ((0, 0), (0, 0), (0, 1)),
    )

    return a, b2, b3, b4, vals


def params_next(X, abilities, difficulties, discriminations, eta):
    a, b2, b3, b4, vals = a_b(abilities, difficulties, discriminations)
    # calculate next lambda values
    observed = (X >= 0).astype(int)
    _X = one_hot(X, vals["K"])

    W = _X + (jnp.sum(_X, axis=-1, keepdims=True) < 1).astype(int) * vals["p"]

    discrim_nu = jnp.sum(
        W * (discriminations * a + b4)
        + 2 * discriminations / (1 + discriminations**2)
        - eta / discriminations,
        axis=[0, 2],
        keepdims=True,
    )
    discrim_den = jnp.sum(
        W * a
        + 2 * discriminations / (1 + discriminations**2)
        + eta / discriminations**2,
        axis=[0, 2],
        keepdims=True,
    )
    discrim = discriminations - discrim_nu / discrim_den
    first_select = jnp.zeros_like(difficulties).at[..., :1].set(1)
    last_select = jnp.zeros_like(difficulties).at[..., -1:].set(1)

    gaps = difficulties[..., 1:] - difficulties[..., :-1]
    g = (
        2 * (2 - last_select) * difficulties
        + jnp.sum(
            W[..., :-1]
            * (
                2 * (a[..., :-1] + a[..., 1:]) * difficulties
                + b3[..., :-1]
                + b2[..., 1:]
            ),
            axis=0,
            keepdims=True,
        )
        - eta
        * (
            (1 - first_select) * jnp.pad(1 / gaps, ((0, 0), (0, 0), (1, 0)))
            - (1 - last_select) * jnp.pad(1 / gaps, ((0, 0), (0, 0), (0, 1)))
        )
    )
    g += (last_select - 1) * difficulties + (first_select - 1) * difficulties
    Hdiag = 2 * (
        2 - last_select + jnp.sum(W[..., :-1] * (a[..., 1:] + a[..., :-1]), axis=0)
    ) + eta * (
        (1 - first_select) * jnp.pad(1 / gaps**2, ((0, 0), (0, 0), (1, 0)))
        - (1 - last_select) * jnp.pad(1 / gaps**2, ((0, 0), (0, 0), (0, 1)))
    )
    Hupper = (last_select - 1) - eta * (1 - last_select) * jnp.pad(
        1 / gaps**2, ((0, 0), (0, 0), (0, 1))
    ) ** 2
    Hlower = jnp.pad(Hupper[..., :-1], ((0, 0), (0, 0), (1, 0)))
    diff = tridiagonal_solve(
        Hlower[0, ...], Hdiag[0, ...], Hupper[0, ...], g[0, :, :, jnp.newaxis]
    )
    diff = difficulties - diff[jnp.newaxis, :, :, 0]
    
    return {
        "discriminations": discrim[jnp.newaxis, :, jnp.newaxis],
        "difficulties": diff,
    }


def main():
    N = 1000
    K = 4
    I = 50
    key = PRNGKey(0)
    abilities = normal(key, N)[:, jnp.newaxis, jnp.newaxis]
    # abilities = jnp.array([0, 0.5, 0.25])[:, jnp.newaxis, jnp.newaxis]
    difficulties = normal(key, (1, I, K - 1))
    difficulties = jnp.sort(difficulties, axis=-1)
    discriminations = jnp.abs(normal(key, I))[jnp.newaxis, :, jnp.newaxis]
    probs = p(abilities, difficulties, discriminations)
    # difficulties = np.array([[0, 1, 2, 3], [-2, 0, 3, 4]])[np.newaxis, ...]
    X = categorical(key, logits=jnp.log(probs))
    # X = np.array([[2, 1], [3, 2], [0, 1]])
    _abilities = jnp.zeros_like(abilities)
    _discriminations = jnp.ones_like(discriminations)
    _difficulites = jnp.transpose(
        jnp.tile(jnp.linspace(-2, 2, num=K - 1)[:, jnp.newaxis], I)
    )[jnp.newaxis, ...]
    for _ in range(10):
        _abilities, _difficulites, _discriminations = params_next(X, _abilities, _difficulites, _discriminations, 1e-1)

    print(_discriminations)


if __name__ == "__main__":
    main()
