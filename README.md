# Probabilistically-autoencoded horseshoe-disentangled multidomain item-response theory models

This code aims to factor item response data using the graded response model and sparse coding.

Note that using logarithmic parameterization causes numerical issues - SoftPlus seems to not cause those issues or at the very least these issues are ameliorated by gradient clipping.
