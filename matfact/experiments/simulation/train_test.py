def train_test_split(X, M):
    "Simple train-test splitting of data."

    # TODO: the split should not be hardcoded
    # what if the set is not 1000?
    # better to have optinoal arg with ratio or number

    X_train = X[:800]
    M_train = M[:800]

    X_test = X[-200:]
    M_test = M[-200:]

    return X_train, X_test, M_train, M_test
