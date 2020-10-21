import matplotlib.pyplot as plt
import pandas


mu_outliers = pandas.read_csv("mu_train_outliers.csv").drop(columns = ["Unnamed: 0"])
mu_inliers = pandas.read_csv("mu_train_inliers.csv").drop(columns = ["Unnamed: 0"])

sigma_outliers = pandas.read_csv("sigma_train_outliers.csv").drop(columns = ["Unnamed: 0"])
sigma_inliers = pandas.read_csv("sigma_train_inliers.csv").drop(columns = ["Unnamed: 0"])


mu_outliers["outliers"] = True
mu_inliers["outliers"] = False

sigma_outliers["outliers"] = True
sigma_inliers["outliers"] = False


cols = mu_outliers.loc[:, "0":"24"]
mu_outliers["mean_mu"] = cols.mean(axis=1)

cols = mu_inliers.loc[:, "0":"24"]
mu_inliers["mean_mu"] = cols.mean(axis=1)

cols = sigma_outliers.loc[:, "0":"24"]
sigma_outliers["mean_sigma"] = cols.mean(axis=1)

cols = sigma_inliers.loc[:, "0":"24"]
sigma_inliers["mean_sigma"] = cols.mean(axis=1)

dt_mu = pandas.concat([mu_inliers, mu_outliers], ignore_index=True)
dt_sigma = pandas.concat([sigma_inliers, sigma_outliers], ignore_index=True)

plt.scatter(dt_mu.loc[dt_mu["outliers"] == True], dt_sigma["mean_sigma"])




