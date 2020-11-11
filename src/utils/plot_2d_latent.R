
library(data.table)
library(dplyr)
library(ggplot2)
library(latex2exp)


DATE <- "2020-11-05"
FOLDER <- file.path("~/university/deep-stat-thesis/results", DATE, "da_vae")
LATENT_DIM <- 2
NB_OBS_PLOT <- 1000

# Import data

# mu_outliers_cars <- fread(file.path(FOLDER, "mu_train_outliers_scenario_cars_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L]
# mu_inliers_cars <- fread(file.path(FOLDER, "mu_train_inliers_scenario_cars_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L]
# sigma_outliers_cars <- fread(file.path(FOLDER, "sigma_train_outliers_scenario_cars_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L]
# sigma_inliers_cars <- fread(file.path(FOLDER, "sigma_train_inliers_scenario_cars_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L]

mu_outliers_mnist <- fread(file.path(FOLDER, "mu_train_outliers_scenario3_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L][-1,]
mu_inliers_mnist <- fread(file.path(FOLDER, "mu_train_inliers_scenario3_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L][-1,]
sigma_outliers_mnist <- fread(file.path(FOLDER, "sigma_train_outliers_scenario3_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L][-1,]
sigma_inliers_mnist <- fread(file.path(FOLDER, "sigma_train_inliers_scenario3_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L][-1,]

#data_all_cars <- rbind(cbind(mu_outliers_cars[, outliers := NULL], sigma_outliers_cars), cbind(mu_inliers_cars[, outliers := NULL], sigma_inliers_cars))[sample(1:(nrow(mu_inliers_cars) + nrow(mu_outliers_cars)), NB_OBS_PLOT, F)]
data_all_mnist <- rbind(cbind(mu_outliers_mnist[, outliers := NULL], sigma_outliers_mnist), cbind(mu_inliers_mnist[, outliers := NULL], sigma_inliers_mnist))[sample(1:(nrow(mu_inliers_mnist) + nrow(mu_outliers_mnist)), NB_OBS_PLOT, F)]


mnist_mu_2d <- ggplot(data_all_mnist, aes(x = mu_1, y = mu_2, color = as.factor(outliers))) +
  geom_point(alpha = 0.7) +
  scale_x_continuous(TeX("$\\mu_1$")) +
  scale_y_continuous(TeX("$\\mu_2$")) +
  scale_color_discrete("Anomalie") +
  theme_bw() +
  theme(
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 16),
    legend.position = "bottom",
    axis.title = element_text(size = 20, face = "bold"),
    axis.text = element_text(size = 14),
    strip.text.x = element_text(size = 12))

mnist_sigma_2d <- ggplot(data_all_mnist, aes(x = sigma_1, y = sigma_2, color = as.factor(outliers))) +
  geom_point(alpha = 0.7) +
  scale_x_continuous(TeX("$\\sigma_1$")) +
  scale_y_continuous(TeX("$\\sigma_2$")) +
  scale_color_discrete("Anomalie") +
  theme_bw() +
  theme(
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 16),
    legend.position = "bottom",
    axis.title = element_text(size = 20, face = "bold"),
    axis.text = element_text(size = 14),
    strip.text.x = element_text(size = 12))

ggsave("mnist_mu_2d.pdf", mnist_mu_2d, width = 40, height = 40, units = "cm")
ggsave("mnist_sigma_2d.pdf", mnist_sigma_2d, width = 40, height = 40, units = "cm")
