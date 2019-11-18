# Probabilistically-autoencoded horseshoe-disentangled multidomain item-response theory models

This code aims to factor item response data using the graded response model and sparse coding. Currently, the code is going through a refactoring into a package. Right now, only the `autoencoding_irt/scripts/rwas_monolithic.py` standalone script works. It fits the IRT model using ADVI and using MCMC.

Note that using logarithmic parameterization causes numerical issues - SoftPlus seems to not cause those issues or at the very least these issues are ameliorated by gradient clipping.