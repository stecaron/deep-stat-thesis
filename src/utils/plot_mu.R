
library(data.table)
library(dplyr)
library(ggplot2)
library(latex2exp)


DATE <- "2020-10-24"
FOLDER <- file.path("~/university/deep-stat-thesis/results", DATE, "da_vae")
LATENT_DIM <- 25
NB_OBS_PLOT <- 1500

# Import data

mu_outliers_cars <- fread(file.path(FOLDER, "mu_train_outliers_scenario_cars_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L]
mu_inliers_cars <- fread(file.path(FOLDER, "mu_train_inliers_scenario_cars_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L]
sigma_outliers_cars <- fread(file.path(FOLDER, "sigma_train_outliers_scenario_cars_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L]
sigma_inliers_cars <- fread(file.path(FOLDER, "sigma_train_inliers_scenario_cars_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L]

mu_outliers_mnist <- fread(file.path(FOLDER, "mu_train_outliers_scenario3_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L]
mu_inliers_mnist <- fread(file.path(FOLDER, "mu_train_inliers_scenario3_plus.csv"), col.names = c("index", paste0("mu_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L]
sigma_outliers_mnist <- fread(file.path(FOLDER, "sigma_train_outliers_scenario3_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 1L]
sigma_inliers_mnist <- fread(file.path(FOLDER, "sigma_train_inliers_scenario3_plus.csv"), col.names = c("index", paste0("sigma_", 1:LATENT_DIM)))[, index := NULL][, outliers := 0L]


data_all_cars <- rbind(cbind(mu_outliers_cars[, outliers := NULL], sigma_outliers_cars)[-1,], cbind(mu_inliers_cars[, outliers := NULL], sigma_inliers_cars)[2:NB_OBS_PLOT,])
data_all_mnist <- rbind(cbind(mu_outliers_mnist[, outliers := NULL], sigma_outliers_mnist)[-1,], cbind(mu_inliers_mnist[, outliers := NULL], sigma_inliers_mnist)[2:NB_OBS_PLOT,])


data_all_cars <- mutate(data_all_cars, mean_mu = rowMeans(select(data_all_cars, starts_with("mu_")), na.rm = TRUE))
data_all_cars <- mutate(data_all_cars, mean_sigma = rowMeans(select(data_all_cars, starts_with("sigma_")), na.rm = TRUE))

data_all_mnist <- mutate(data_all_mnist, mean_mu = rowMeans(select(data_all_mnist, starts_with("mu_")), na.rm = TRUE))
data_all_mnist <- mutate(data_all_mnist, mean_sigma = rowMeans(select(data_all_mnist, starts_with("sigma_")), na.rm = TRUE))


# Plot data

(plot_mu <- ggplot(data_all_cars, aes(x = mean_mu, fill = as.factor(outliers))) +
  geom_density(alpha = 0.6) +
  scale_x_continuous(TeX("$\\mu$ moyen sur les 25 dimensions latentes")) +
  scale_y_continuous("Densité") +
  scale_fill_discrete("Anomalie") +
  theme_classic() +
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14),
        legend.text = element_text(size = 12)))

(plot_sigma <- ggplot(data_all_cars, aes(x = mean_sigma, fill = as.factor(outliers))) +
    geom_density(alpha = 0.6) +
    scale_x_continuous(TeX("$\\sigma$ moyen sur les 25 dimensions latentes")) +
    scale_y_continuous("Densité") +
    scale_fill_discrete("Anomalie") +
    theme_classic() +
    theme(axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.text = element_text(size = 12)))

(plot_mu_mnist <- ggplot(data_all_mnist, aes(x = mean_mu, fill = as.factor(outliers))) +
    geom_density(alpha = 0.6) +
    scale_x_continuous(TeX("$\\mu$ moyen sur les 25 dimensions latentes")) +
    scale_y_continuous("Densité") +
    scale_fill_discrete("Anomalie") +
    theme_classic() +
    theme(axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.text = element_text(size = 12)))

(plot_sigma_mnist <- ggplot(data_all_mnist, aes(x = mean_sigma, fill = as.factor(outliers))) +
    geom_density(alpha = 0.6) +
    scale_x_continuous(TeX("$\\sigma$ moyen sur les 25 dimensions latentes")) +
    scale_y_continuous("Densité") +
    scale_fill_discrete("Anomalie") +
    theme_classic() +
    theme(axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.text = element_text(size = 12)))



ggsave("plot_mu.pdf", plot_mu, width = 40, height = 20, units = "cm")
ggsave("plot_sigma.pdf", plot_sigma, width = 40, height = 20, units = "cm")

ggsave("plot_mu_mnist.pdf", plot_mu_mnist, width = 40, height = 20, units = "cm")
ggsave("plot_sigma_mnist.pdf", plot_sigma_mnist, width = 40, height = 20, units = "cm")



# PCA plots ---------------------------------------------------------------

# On calcule le PCA
cars_pca <- prcomp(data_all_cars %>% select(-outliers,-mean_mu, -mean_sigma), center = TRUE, scale. = TRUE)
mnist_pca <- prcomp(data_all_mnist %>% select(-outliers,-mean_mu, -mean_sigma), center = TRUE, scale. = TRUE)

summary(cars_pca)
summary(mnist_pca)

cars_var_1 <- paste(round(100 * (summary(cars_pca)$sdev^2/sum(summary(cars_pca)$sdev^2))[1], 2), "%")
cars_var_2 <- paste(round(100 * (summary(cars_pca)$sdev^2/sum(summary(cars_pca)$sdev^2))[2], 2), "%")

mnist_var_1 <- paste(round(100 * (summary(mnist_pca)$sdev^2/sum(summary(mnist_pca)$sdev^2))[1], 2), "%")
mnist_var_2 <- paste(round(100 * (summary(mnist_pca)$sdev^2/sum(summary(mnist_pca)$sdev^2))[2], 2), "%")

cars_pca_plot <- data.frame(
  PC1 = cars_pca$x[, 1],
  PC2 = cars_pca$x[, 2],
  outlier = data_all_cars$outliers
)

mnist_pca_plot <- data.frame(
  PC1 = mnist_pca$x[, 1],
  PC2 = mnist_pca$x[, 2],
  outlier = data_all_mnist$outliers
)

cars_plot <- ggplot(cars_pca_plot, aes(x = PC1, y = PC2, color = as.factor(outlier))) +
  geom_point(alpha = 0.8) + 
  scale_x_continuous(paste0("1er CP (variance expliquée : ", cars_var_1, ")")) +
  scale_y_continuous(paste0("2e CP (variance expliquée : ", cars_var_2, ")")) +
  scale_color_discrete("Anomalie") +
  theme_bw() +
  theme(
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 10),
    strip.text.x = element_text(size = 12))

mnist_plot <- ggplot(mnist_pca_plot, aes(x = PC1, y = PC2, color = as.factor(outlier))) +
  geom_point() + 
  scale_color_discrete("Anomalie") +
  scale_x_continuous(paste0("1er CP (variance expliquée : ", mnist_var_1, ")")) +
  scale_y_continuous(paste0("2e CP (variance expliquée : ", mnist_var_2, ")")) +
  theme_bw() +
  theme(
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 10),
    strip.text.x = element_text(size = 12))

