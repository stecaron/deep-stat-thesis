library(data.table)
library(dplyr)
library(ggplot2)


# On importe les données
mu_inliers <- fread("university/deep-stat-thesis/mu_inliers.csv")[-1,][, V1 := NULL][, outlier := 0L]
mu_outliers <- fread("university/deep-stat-thesis/mu_outliers.csv")[-1,][, V1 := NULL][, outlier := 1L]
df_mu <- rbind(mu_inliers, mu_outliers)

sigma_inliers <- fread("university/deep-stat-thesis/sigma_inliers.csv")[-1,][, V1 := NULL][, outlier := 0L]
sigma_outliers <- fread("university/deep-stat-thesis/sigma_outliers.csv")[-1,][, V1 := NULL][, outlier := 1L]
df_sigma <- rbind(sigma_inliers, sigma_outliers)

# On valide les dimensions de df
dim(df_mu)
dim(df_sigma)


# On valide les moyennes outliers versus inliers
df_mu %>% group_by(outlier) %>% summarise(mu_moyen = mean(c(V2, V3, V4, V5, V6)))
df_sigma %>% group_by(outlier) %>% summarise(sigma_moyen = mean(c(V2, V3, V4, V5, V6)))


# On calcule le PCA
mu_pca <- prcomp(df_mu %>% select(-outlier))
sigma_pca <- prcomp(df_sigma %>% select(-outlier))

summary(mu_pca)
summary(sigma_pca)

mu_pca_plot <- data.frame(
  PC1 = mu_pca$x[, 1],
  PC2 = mu_pca$x[, 2],
  outlier = df_mu$outlier
)

sigma_pca_plot <- data.frame(
  PC1 = sigma_pca$x[, 1],
  PC2 = sigma_pca$x[, 2],
  outlier = df_sigma$outlier
)

mu_plot <- ggplot(mu_pca_plot, aes(x = PC1, y = PC2, color = as.factor(outlier))) +
  geom_point() + 
  scale_color_discrete("Outlier") +
  ggtitle("2D PCA-plot for mu with 5 latent dimensions")

sigma_plot <- ggplot(sigma_pca_plot, aes(x = PC1, y = PC2, color = as.factor(outlier))) +
  geom_point() + 
  scale_color_discrete("Outlier") +
  ggtitle("2D PCA-plot for sigma with 5 latent dimensions")


ggsave("mu_pca_2d.png", mu_pca)