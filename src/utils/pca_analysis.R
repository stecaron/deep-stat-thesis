library(data.table)
library(dplyr)
library(ggplot2)
library(e1071)


# On importe les données
mu_inliers <- fread("university/deep-stat-thesis/mu_inliers.csv")[-1,][, V1 := NULL][, outlier := 0L]
mu_outliers <- fread("university/deep-stat-thesis/mu_outliers.csv")[-1,][, V1 := NULL][, outlier := 1L]
df_mu <- rbind(mu_inliers, mu_outliers)

sigma_inliers <- fread("university/deep-stat-thesis/sigma_inliers.csv")[-1,][, V1 := NULL][, outlier := 0L]
sigma_outliers <- fread("university/deep-stat-thesis/sigma_outliers.csv")[-1,][, V1 := NULL][, outlier := 1L]
df_sigma <- rbind(sigma_inliers, sigma_outliers)

colnames(df_sigma) <- paste("sigma", colnames(df_sigma), sep = "_")
data_tot <- cbind(df_mu, copy(df_sigma)[, sigma_outlier := NULL])
data_tot[, outlier := as.factor(outlier)]

# On valide les dimensions de df
dim(df_mu)
dim(df_sigma)


# On valide les moyennes outliers versus inliers
df_mu %>% group_by(outlier) %>% summarise(mu_moyen = mean(c(V2, V3, V4, V5, V6)))
df_sigma %>% group_by(outlier) %>% summarise(sigma_moyen = mean(c(V2, V3, V4, V5, V6)))


# On calcule le PCA
mu_pca <- prcomp(df_mu %>% select(-outlier), scale. = TRUE, center = TRUE)
sigma_pca <- prcomp(df_sigma %>% select(-sigma_outlier), scale. = TRUE, center = TRUE)
both_pca <- prcomp(data_tot %>% select(-outlier), scale. = TRUE, center = TRUE)

summary(mu_pca)
summary(sigma_pca)
summary(both_pca)

mu_pca_plot <- data.frame(
  PC1 = mu_pca$x[, 1],
  PC2 = mu_pca$x[, 2],
  PC3 = mu_pca$x[, 3],
  PC4 = mu_pca$x[, 4],
  PC5 = mu_pca$x[, 5],
  outlier = df_mu$outlier
)

sigma_pca_plot <- data.frame(
  PC1 = sigma_pca$x[, 1],
  PC2 = sigma_pca$x[, 2],
  PC3 = sigma_pca$x[, 3],
  PC4 = sigma_pca$x[, 4],
  PC5 = sigma_pca$x[, 5],
  outlier = df_sigma$sigma_outlier
)

both_pca_plot <- data.frame(
  PC1 = both_pca$x[, 1],
  PC2 = both_pca$x[, 2],
  PC3 = both_pca$x[, 3],
  PC4 = both_pca$x[, 4],
  PC5 = both_pca$x[, 5],
  PC6 = both_pca$x[, 6],
  PC7 = both_pca$x[, 7],
  PC8 = both_pca$x[, 8],
  PC9 = both_pca$x[, 9],
  PC10 = both_pca$x[, 10],
  outlier = data_tot$outlier
)

mu_plot <- ggplot(mu_pca_plot, aes(x = PC1, y = PC2, color = as.factor(outlier))) +
  geom_point(alpha = 0.2) + 
  scale_color_discrete("Outlier") +
  ggtitle("2D PCA-plot for mu with 5 latent dimensions")

sigma_plot <- ggplot(sigma_pca_plot, aes(x = PC1, y = PC2, color = as.factor(outlier))) +
  geom_point(alpha = 0.2) + 
  scale_color_discrete("Outlier") +
  ggtitle("2D PCA-plot for sigma with 5 latent dimensions")

ggsave("mu_pca_2d.png", mu_plot)
ggsave("sigma_pca_2d.png", sigma_plot)


my_cols <- c("#00AFBB", "#E7B800")

jpeg("mu.jpeg")
pairs(mu_pca_plot[, 1:5], 
      main = "PCA pairwise plot for MU with 5 latent dimensions",
      pch = 21, 
      col = alpha(my_cols[mu_pca_plot$outlier + 1], 0.5))
dev.off()

jpeg("sigma.jpeg")
pairs(sigma_pca_plot[, 1:5], 
      main = "PCA pairwise plot for SIGMA with 5 latent dimensions",
      pch = 21, 
      col = alpha(my_cols[sigma_pca_plot$outlier + 1], 0.5))
dev.off()

jpeg("both-mu-sigma.jpeg")
pairs(both_pca_plot[, 1:5], 
      main = "PCA pairwise plot for MU SIGMA with 5 latent dimensions",
      pch = 21, 
      col = alpha(my_cols[sigma_pca_plot$outlier + 1], 0.5))
dev.off()


## Fitter svm sur dimensions

train.index <- caret::createDataPartition(data_tot$outlier, p = 0.8, list = FALSE)
train <- data_tot[ train.index,]
test  <- data_tot[-train.index,]

table(train$outlier)
table(test$outlier)

svm <- svm(outlier ~ ., data = train, kernel = "radial", cost = 100000, scale = FALSE)
train_preds <- predict(svm, train)
test_preds <- predict(svm, test)
table(preds)

caret::confusionMatrix(train_preds, train$outlier)
caret::confusionMatrix(test_preds, test$outlier)


## Fit a density

kde <- density()


## Analysis density

density_df <- fread("university/deep-stat-thesis/train_density.csv")

summary(density_df[outliers == 0,]$density)
summary(density_df[outliers == 1,]$density)

ggplot(density_df, aes(x = density, fill = as.factor(outliers), color = as.factor(outliers))) +
  geom_density(alpha = 0.5)

density_df$density_norm <- (density_df$density - min(density_df$density))/(max(density_df$density) - min(density_df$density))
density_df$order <- rank(density_df$density_norm)
density_df[, point_size := 1]
density_df[outliers == 1, point_size := 5]

ggplot(density_df, aes(x = order, y = density_norm, color = as.factor(outliers))) +
  geom_point(aes(size = point_size)) + 
  scale_color_manual(values=c("#ffff00", "#000066")) +
  scale_size(guide = "none")

table(density_df$outliers)


alpha <- 0.1
cutoff <- sort(density_df$density)[alpha * nrow(density_df)]

preds_density <- as.factor(ifelse(density_df$density <= cutoff, 1, 0))

caret::confusionMatrix(preds_density, as.factor(density_df$outliers))


