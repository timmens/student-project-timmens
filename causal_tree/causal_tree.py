import numpy as np
import pandas as pd


# from pandas import Series, DataFrame
def causal_tree(data, treatment):
    '''Input args:
                data: matrix(y, x1, ..., xk),
                        y: outcome
                        x1,...,xk: covariates (plus intercept)
                treatment: treatment indicator vector
        Output:
        Assumptions: No missing values; ...
    '''

    # check stuff about data matrix
    if isinstance(data, pd.DataFrame):
        data = data.values
    # else we assume (for now) that the data is given in a numpy 2D array
    y = data[:, 0]
    x = data[:, 1:]
    nobs, nvar = data.shape

    # check stuff about treatment vector
    treatment = np.array(treatment)
    if not np.isin(treatment, (0, 1)).all():
        raise ValueError("Treatment vector is supposed to be binary.")
    if treatment.sum() == 0 or treatment.sum() == nobs:
        raise ValueError("Data contains only treated cases or only controlled cases.")

    # do some more computations
    xvar = np.apply_along_axis(np.var, 0, x)
    

    return None
