

import jax.numpy as jnp
from factor_analyzer import FactorAnalyzer

from autoencirt.data.rwa import get_data
from autoencirt.irt import FactorizedGRModel

dim = 2

def main():
    pd_data = get_data(reorient=True, pandas=True)
    responses = pd_data[0].iloc[:, :22]
    fa = FactorAnalyzer(n_factors=dim)
    fa.fit(responses)
    loadings = fa.loadings_
    dataset, num_people = get_data(reorient=True)
    item_names = [f"Q{j}" for j in range(1, 23)]
    scale_indices = [
        [1, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20],
        [0, 2, 4, 6, 9, 11, 13, 16, 18, 21],
    ]
    fgrm = FactorizedGRModel(
        data=dataset,
        item_keys=item_names,
        num_people=num_people,
        dim=dim,
        eta_scale=1e-3,
        kappa_scale=1e-3,
        weight_exponent=1,
        response_cardinality=9,
        scale_indices=scale_indices,
        discrimination_guess=0.5*jnp.abs(loadings).T.astype(jnp.float64)[
            jnp.newaxis, :, :, jnp.newaxis
        ],
    )
    batched = dataset.batch(20)
    p = fgrm.sample(20)
    ll = fgrm.log_likelihood(batched, **p)
    print(ll)

if __name__ == "__main__":
    main()