from .grm import GRModel


class AEGRModel(GRModel):

    def __init__(self,
                 auxiliary_parameterization=True,
                 xi_scale=1e-2,
                 eta_scale=1e-2,
                 kappa_scale=1e-2,
                 weight_exponent=1.0,
                 dim=2,
                 decay=0.25
                 ):
        super(AEGRModel, self).__init__(
            auxiliary_parameterization=True,
            xi_scale=xi_scale,
            eta_scale=eta_scale,
            kappa_scale=kappa_scale,
            weight_exponent=weight_exponent,
            dim=dim,
            decay=decay
        )


def main():
    pass


if __name__ == "__main__":
    main()
