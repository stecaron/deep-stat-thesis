from sklearn.neighbors import LocalOutlierFactor


def lof_scoring(dt, n_neighbors=20, pourc=0.01):

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=pourc)
    score = lof.fit_predict(dt)

    return score