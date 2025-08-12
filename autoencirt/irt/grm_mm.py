import numpy as np
from jax import numpy as jnp
from jax.nn import one_hot, sigmoid

dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
ddsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))


#
# N x I x K
# Evaluate the function, and the derivatives of the function wrt theta_n,
# lambda_i, tau_k, tau_k-1
def p_nik(abilities, difficulties, discriminations):  # dimensio
    N = abilities.shape[0]
    I = discriminations.shape[1]
    K = difficulties.shape[-1] + 1
    p_cum = sigmoid(discriminations * (abilities - difficulties))  # Pr(x>=k)
    # first partials, will be N x I x K x d where d is the dimension of the parameter
    # d_cum_abilites will be N x I x K x N
    dp_cum_dtheta = (p_cum * (1 - p_cum)) * discriminations
    dp_cum_dtau = -(p_cum * (1 - p_cum)) * discriminations
    dp_cum_dlambda = (p_cum * (1 - p_cum)) * (abilities - difficulties)

    dp_dtau_0 = jnp.pad(dp_cum_dtau, ((0, 0), (0, 0), (1, 0)), constant_values=0)
    dp_dtau_1 = -jnp.pad(dp_cum_dtau, ((0, 0), (0, 0), (0, 1)), constant_values=0)

    # second partials
    # Diagonal terms
    d2p_cum_dtheta2 = (p_cum * (1 - p_cum) * (1 - 2 * p_cum)) * discriminations**2
    d2p_cum_ddlambda2 = (p_cum * (1 - p_cum) * (1 - 2 * p_cum)) * (
        abilities - difficulties
    ) ** 2
    d2p_cum_dtau2 = p_cum * (1 - p_cum) * (1 - 2 * p_cum) * discriminations**2

    # mixed partials
    d2p_cum_dabilities_difficulties = (
        -(p_cum * (1 - p_cum) * (1 - 2 * p_cum)) * discriminations**2
    )

    d2p_cum_dabilities_discriminations = (p_cum * (1 - p_cum)) + p_cum * (1 - p_cum) * (
        1 - 2 * p_cum
    ) * discriminations**2

    d2p_cum_ddifficulties_discriminations = -p_cum * (1 - p_cum) - p_cum * (
        1 - p_cum
    ) * (1 - 2 * p_cum) * discriminations * (abilities - difficulties)

    p_cum = jnp.pad(p_cum, ((0, 0), (0, 0), (1, 0)), constant_values=1)
    p_cum = jnp.pad(p_cum, ((0, 0), (0, 0), (0, 1)), constant_values=0)

    # padding for gradient
    dp_cum_dabilities = jnp.pad(
        dp_cum_dtheta, ((0, 0), (0, 0), (1, 1)), constant_values=0
    )
    dp_cum_ddifficulties = jnp.pad(
        dp_cum_dtau,
        ((0, 0), (0, 0), (1, 1)),
        constant_values=0,
    )
    dp_cum_ddiscrimintations = jnp.pad(
        dp_cum_dlambda, ((0, 0), (0, 0), (1, 1)), constant_values=0
    )

    dp_dabilities = dp_cum_dabilities[:, :, 1:, ...] - dp_cum_dabilities[:, :, :-1, ...]
    dp_ddifficulties = dp_cum_ddifficulties[:, :, 1:, ...]
    dp_ddifficulties = -dp_cum_ddifficulties[:, :, :-1, ...]
    dp_ddiscrimintations = (
        dp_cum_ddiscrimintations[:, :, 1:, ...]
        - dp_cum_ddiscrimintations[:, :, :-1, ...]
    )

    # diagonal hessian
    d2p_cum_dabilities2 = jnp.pad(
        d2p_cum_dtheta2, ((0, 0), (0, 0), (1, 1)), constant_values=0
    )
    d2p_cum_ddifficulties2 = jnp.pad(
        d2p_cum_dtau2,
        ((0, 0), (0, 0), (1, 1)),
        constant_values=0,
    )
    d2p_cum_ddiscrimintations2 = jnp.pad(
        d2p_cum_ddlambda2, ((0, 0), (0, 0), (1, 1)), constant_values=0
    )

    d2p_cum_ddifficulties0 = dp_cum_ddifficulties[..., :-1]
    d2p_cum_ddifficulties1 = dp_cum_ddifficulties[..., 1:]
    ####################################################
    # pad the mixed terms

    d2p_cum_dabilities_discriminations = jnp.pad(
        d2p_cum_dabilities_discriminations,
        ((0, 0), (0, 0), (1, 1)),
        constant_values=0,
    )

    d2p_cum_dabilities_difficulties = jnp.pad(
        d2p_cum_dabilities_difficulties,
        ((0, 0), (0, 0), (1, 1)),
        constant_values=0,
    )
    d2p_cum_ddifficulties_discriminations = jnp.pad(
        d2p_cum_ddifficulties_discriminations,
        ((0, 0), (0, 0), (1, 1)),
        constant_values=0,
    )

    d2p_dabilities = (
        d2p_cum_dabilities2[:, :, 1:, ...] - d2p_cum_dabilities2[:, :, :-1, ...]
    )
    d2p_ddiscriminations = (
        d2p_cum_ddiscrimintations2[:, :, 1:, ...]
        - d2p_cum_ddiscrimintations2[:, :, :-1, ...]
    )

    d2p_dabilities_ddifficulties = (
        d2p_cum_dabilities_difficulties[:, :, 1:, ...]
        - d2p_cum_dabilities_difficulties[:, :, :-1, ...]
    )
    d2p_dabilities_ddiscrimintations = (
        d2p_cum_dabilities_discriminations[:, :, 1:, ...]
        - d2p_cum_dabilities_discriminations[:, :, :-1, ...]
    )

    d2p_ddifficulties_ddiscrimintations = (
        d2p_cum_ddifficulties_discriminations[:, :, 1:, ...]
        - d2p_cum_ddifficulties_discriminations[:, :, :-1, ...]
    )

    p = p_cum[..., 1:] - p_cum[..., :-1]

    # compute derivatives

    gradients = {
        "abilities": dp_dabilities,
        "difficulties0": dp_dtau_0,
        "difficulties1": dp_dtau_1,
        "discriminations": dp_ddiscrimintations,
    }

    grad_log_p = {k: v / p for k, v in gradients.items()}

    grad2_p = {
        ("abilities", "abilities"): d2p_dabilities,
        ("discriminations", "discriminations"): d2p_ddiscriminations,
        ("difficulties0", "difficulties0"): d2p_cum_ddifficulties0,
        ("difficulties0", "difficulties1"): d2p_cum_ddifficulties0,
        ("difficulties1", "difficulties1"): d2p_cum_ddifficulties1,
        ("abilities", "discriminations"): d2p_dabilities_ddiscrimintations,
        ("abilities", "difficulties0"): d2p_dabilities_ddifficulties,
        ("abilities", "difficulties1"): d2p_dabilities_ddifficulties,
        ("discriminations", "difficulties0"): d2p_ddifficulties_ddiscrimintations,
        ("discriminations", "difficulties1"): d2p_ddifficulties_ddiscrimintations,
    }
    grad2_log_p = {
        k: (grad2_p[k] * p - gradients[k[0]] * gradients[k[1]]) / p**2
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
    k = [(k, k) for k in vals["grad(log(p))"].keys()]
    diagonals = [vals["grad(grad(log(p)))"][j] for j in k]
    a = jnp.min(jnp.stack(diagonals, axis=-1), axis=-1)

    b4 = vals["grad(p)"]["discriminations"] - a * discriminations**2
    tau0 = jnp.pad(
        difficulties[..., :-1], ((0, 0), (0, 0), (1, 0)), constant_values=0
    )  # \tau_{k-1}

    tau1 = jnp.pad(
        difficulties[..., 1:], ((0, 0), (0, 0), (0, 1)), constant_values=0
    )  # \tau_k
    b2 = vals["grad(p)"]["difficulties0"][..., 1:] - a[..., 1:] * tau0**2
    b3 = vals["grad(p)"]["difficulties1"][..., :-1] - a[..., :-1] * tau1**2

    return a, b2, b3, b4, vals


def params_next(X, abilities, difficulties, discriminations):
    a, b2, b3, b4, vals = a_b(abilities, difficulties, discriminations)
    # calculate next lambda values
    observed = (X >= 0).astype(int)
    _X = one_hot(X, vals['K'])
    discrim_numerator = jnp.sum(b4*_X, axis=[0, 2]) + jnp.sum(
        jnp.sum(b4 * vals["p"], axis=-1) * (1 - observed), axis=0
    )
    discrim_denominator = jnp.sum(a*_X, axis=[0, 2]) + jnp.sum(
        jnp.sum(a * vals["p"], axis=-1) * (1 - observed), axis=0
    )
    discrim = -discrim_numerator/discrim_denominator
    
    difficulty_midpoints = (difficulties[..., 1:] + difficulties[..., :-1])/2
    difficulty_midpoints = jnp.pad(difficulty_midpoints, ((0,0), (0, 0), (1, 1)), constant_values=0)
    difficulty_numerator = (
        2*difficulty_midpoints[..., 1:] + 2*difficulty_midpoints[..., :-1]
        + jnp.sum(b3*_X[..., :-1], axis=[0]) + jnp.sum(b2*_X[..., 1:], axis=[0])
        + jnp.sum(b2*vals["p"][..., 1:]  * (1 - observed[... ,jnp.newaxis]), axis=0)
        + jnp.sum(b3*vals["p"][..., :-1]  * (1 - observed[... ,jnp.newaxis]), axis=0)
    )
    difficulty_denominator = (
        -2 + jnp.sum(a[..., :-1]*_X[..., :-1], axis=[0]) + jnp.sum(a[..., 1:]*_X[..., 1:], axis=[0])
        + jnp.sum(a[..., 1:]*vals["p"][..., 1:]  * (1 - observed[... ,jnp.newaxis]), axis=0)
        + jnp.sum(a[..., :-1]*vals["p"][..., :-1]  * (1 - observed[... ,jnp.newaxis]), axis=0) 
    )
    difficult =- difficulty_numerator/difficulty_denominator
    return {"discriminations": discrim, "difficulties": difficult}


def main():
    abilities = np.array([0, 0.5, 0.25])[:, np.newaxis, np.newaxis]
    difficulties = np.array([[0, 1, 2, 3], [-2, 0, 3, 4]])[np.newaxis, ...]
    discriminations = np.array([1, 2])[np.newaxis, :, np.newaxis]

    X = np.array([[2, 1], [3, 2], [0, 1]])

    res = params_next(X, abilities, difficulties, discriminations)
    print(res)


if __name__ == "__main__":
    main()
