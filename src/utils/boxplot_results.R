
library(data.table)
library(ggplot2)
library(cowplot)
library(gridExtra)
library(grid)

DATE <- "2020-10-09"
FOLDER <- file.path("university/deep-stat-thesis/results", DATE)
MODELS <- c("kpca", "ae", "vae_isof", "da_vae")

results_list <- list()

for (model in MODELS) {
  
  temp_path <- file.path(FOLDER, model)
  results_files <- grep("^results_", list.files(temp_path), value = TRUE)
  
  for (file in results_files) {
    
    temp_file_path <- file.path(temp_path, file)
    temp_file <- fread(temp_file_path)
    scenario <- gsub("results_|\\.csv", "", file)
    method <- ifelse(model == "vae_isof", "isof_vae", model)
    temp_file[, method := method]
    temp_file[, scenario := scenario]
    temp_file[, dataset := "mnist"]
    temp_file[grep("cars", scenario), dataset := "imagenet"]
    temp_file[, scenario2 := strsplit(scenario, "_")[[1]][1]]
    temp_file[grep("cars", scenario), scenario2 := "cars"]
    temp_file[, contamination := strsplit(scenario, "_")[[1]][2]]
    temp_file[scenario2 == "cars", contamination := strsplit(scenario, "_")[[1]][3]]
    
    results_list[[paste(model, scenario, sep = "_")]] <- temp_file
    
  }
  
}

# Some transofs for better graphs
compiled_results <- rbindlist(results_list, use.names = TRUE, fill = FALSE)
compiled_results$method <- factor(compiled_results$method, levels=c("kpca", "ae", "isof_vae", "da_vae"))
compiled_results[contamination == "egal", contamination := "Égal"]
compiled_results[contamination == "moins", contamination := "Moins"]
compiled_results[contamination == "plus", contamination := "Plus"]
compiled_results$contamination <- factor(compiled_results$contamination, levels=c("Moins", "Égal", "Plus"))
compiled_results[, scenario2 := gsub("scenario", "", scenario2)]
compiled_results[scenario2 <=3, scenario3 := "Scenario 1 à 3"]
compiled_results[scenario2 >3, scenario3 := "Scenario 4 à 6"]
compiled_results[method == "kpca", method := "kPCA"]
compiled_results[method == "ae", method := "AE"]
compiled_results[method == "isof_vae", method := "ISOF-VAE"]
compiled_results[method == "da_vae", method := "DA-VAE"]


# Plot AUC ImagetNet
(auc_cars <- ggplot(compiled_results[dataset == "imagenet",], aes(x = method, y = auc, group = method)) +
              geom_boxplot(position = position_dodge()) +
              facet_grid(. ~ contamination) +
              scale_y_continuous("Aire sous la courbe ROC") +
              scale_x_discrete("Méthode") +
              theme_bw() + 
              theme(
                legend.title = element_text(size = 12),
                legend.text = element_text(size = 10),
                axis.title = element_text(size = 12, face = "bold"),
                axis.text.x = element_text(size = 10),
                strip.text.x = element_text(size = 12)))

ggsave("auc_cars.pdf", auc_cars, width = 40, height = 20, units = "cm")

# Plot AUC MNIST
auc_mnist_1 <- ggplot(compiled_results[dataset == "mnist" & scenario2 <= 3,], aes(x = scenario2, y = auc, fill = method)) +
    geom_boxplot(position = position_dodge(1)) +
    facet_grid(. ~ contamination, scales = "free") +
    ylab(NULL) +
    xlab(NULL) +
    scale_fill_discrete("Méthode") +
    theme_bw() +
  theme(
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 10),
    strip.text.x = element_text(size = 12))


auc_mnist_2 <- ggplot(compiled_results[dataset == "mnist" & scenario2 > 3,], aes(x = scenario2, y = auc, fill = method)) +
    geom_boxplot(position = position_dodge(1)) +
    facet_grid(. ~ contamination, scales = "free") +
    ylab(NULL) +
    scale_x_discrete("Scénario de test") +
    scale_fill_discrete("Méthode") +
    theme_bw() + 
    theme(
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      legend.position = "bottom",
      axis.title = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(size = 10),
      strip.text.x = element_text(size = 12))

legend <- get_legend(
  # create some space to the left of the legend
  auc_mnist_2 + theme(legend.box.margin = margin(0, 4, 0, 0))
)

y.grob <- textGrob("Aire sous la courbe ROC", 
                   gp=gpar(fontface="bold", fontsize=12), rot=90)

auc_mnist <- plot_grid(
  auc_mnist_1 + theme(legend.position="none"),
  auc_mnist_2 + theme(legend.position="none"),
  align = 'vh',
  nrow = 2
)

auc_mnist <- plot_grid(
  auc_mnist,
  legend,
  nrow = 2,
  rel_heights = c(3, .4)
)

auc_mnist_final <- grid.arrange(arrangeGrob(auc_mnist, left = y.grob))

ggsave("auc_mnist.pdf", auc_mnist_final, width = 40, height = 20, units = "cm")

# Plot precision ImagetNet
(precision_cars <- ggplot(compiled_results[dataset == "imagenet",], aes(x = method, y = precision, group = method)) +
    geom_boxplot(position = position_dodge()) +
    facet_grid(. ~ contamination) +
    scale_y_continuous("Précision") +
    scale_x_discrete("Méthode") +
    theme_bw() +
    theme(
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(size = 10),
      strip.text.x = element_text(size = 12)))

ggsave("precision_cars.pdf", precision_cars, width = 40, height = 20, units = "cm")

# Plot precision MNIST
precision_mnist_1 <- ggplot(compiled_results[dataset == "mnist" & scenario2 <= 3,], aes(x = scenario2, y = precision, fill = method)) +
    geom_boxplot(position = position_dodge(1)) +
    facet_grid(contamination ~ ., scales = "free") +
    scale_y_continuous("Précision") +
    xlab(NULL) +
    scale_fill_discrete("Méthode") +
    theme_bw() +
    theme(
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(size = 10),
      strip.text.x = element_text(size = 12))

precision_mnist_2 <- ggplot(compiled_results[dataset == "mnist" & scenario2 > 3,], aes(x = scenario2, y = precision, fill = method)) +
    geom_boxplot(position = position_dodge(1)) +
    facet_grid(contamination ~ ., scales = "free") +
    ylab(NULL) +
    xlab(NULL) +
    scale_fill_discrete("Méthode") +
    theme_bw() +
    theme(
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      legend.position = "bottom",
      axis.title = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(size = 10),
      strip.text.x = element_text(size = 12))

legend <- get_legend(
  # create some space to the left of the legend
  precision_mnist_2 + theme(legend.box.margin = margin(0, 4, 0, 0))
)

x.grob <- textGrob("Scénario de test", 
                   gp=gpar(fontface="bold", fontsize=12))

precision_minst <- plot_grid(
  precision_mnist_1 + theme(legend.position="none"),
  precision_mnist_2 + theme(legend.position="none"),
  align = 'vh',
  nrow = 1
)

precision_minst <- grid.arrange(arrangeGrob(precision_minst, bottom = x.grob))

precision_minst_final <- plot_grid(
  precision_minst,
  legend,
  nrow = 2,
  rel_heights = c(3, .4)
)

ggsave("precision_mnist.pdf", precision_minst_final, width = 40, height = 20, units = "cm")


# Plot recall ImagetNet
(recall_cars <- ggplot(compiled_results[dataset == "imagenet",], aes(x = method, y = recall, group = method)) +
    geom_boxplot(position = position_dodge()) +
    facet_grid(. ~ contamination) +
    scale_y_continuous("Rappel") +
    scale_x_discrete("Méthode") +
    theme_bw() +
    theme(
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(size = 10),
      strip.text.x = element_text(size = 12)))

ggsave("recall_cars.pdf", recall_cars, width = 40, height = 20, units = "cm")

# Plot precision MNIST
recall_mnist_1 <- ggplot(compiled_results[dataset == "mnist" & scenario2 <= 3,], aes(x = scenario2, y = recall, fill = method)) +
    geom_boxplot(position = position_dodge(1)) +
    facet_grid(contamination ~ ., scales = "free") +
    scale_y_continuous("Rappel") +
    xlab(NULL) +
    scale_fill_discrete("Méthode") +
    theme_bw() +
    theme(
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(size = 10),
      strip.text.x = element_text(size = 12))


recall_mnist_2 <- ggplot(compiled_results[dataset == "mnist" & scenario2 > 3,], aes(x = scenario2, y = recall, fill = method)) +
    geom_boxplot(position = position_dodge(1)) +
    facet_grid(contamination ~ ., scales = "free") +
  ylab(NULL) +
  xlab(NULL) +
  scale_fill_discrete("Méthode") +
  theme_bw() +
  theme(
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "bottom",
    axis.title = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 10),
    strip.text.x = element_text(size = 12))

legend <- get_legend(
  # create some space to the left of the legend
  recall_mnist_2 + theme(legend.box.margin = margin(0, 4, 0, 0))
)

x.grob <- textGrob("Scénario de test", 
                   gp=gpar(fontface="bold", fontsize=12))

recall_minst <- plot_grid(
  recall_mnist_1 + theme(legend.position="none"),
  recall_mnist_2 + theme(legend.position="none"),
  align = 'vh',
  nrow = 1
)

recall_minst <- grid.arrange(arrangeGrob(recall_minst, bottom = x.grob))

recall_minst_final <- plot_grid(
  recall_minst,
  legend,
  nrow = 2,
  rel_heights = c(3, .4)
)

ggsave("recall_mnist.pdf", recall_minst_final, width = 40, height = 20, units = "cm")
