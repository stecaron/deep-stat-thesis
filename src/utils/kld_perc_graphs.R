

library(data.table)
library(ggplot2)
library(scales)


FOLDER <- "~/university/deep-stat-thesis/results/2020-10-21/da_vae/"


# Import data -------------------------------------------------------------

mnist_scenario_3_plus <- fread(file.path(FOLDER, "mnist_vae_modelscenario3_plus_kld_percentage.csv"))
mnist_scenario_3_moins <- fread(file.path(FOLDER, "mnist_vae_modelscenario3_moins_kld_percentage.csv"))
mnist_scenario_3_egal <- fread(file.path(FOLDER, "mnist_vae_modelscenario3_egal_kld_percentage.csv"))

cars_scenario_plus <- fread(file.path(FOLDER, "vae_model_carsscenario_cars_plus_kld_percentage.csv"))
cars_scenario_moins <- fread(file.path(FOLDER, "vae_model_carsscenario_cars_moins_kld_percentage.csv"))
cars_scenario_egal <- fread(file.path(FOLDER, "vae_model_carsscenario_cars_egal_kld_percentage.csv"))

mnist_scenario_3_moins[, scenario := "Moins"]
mnist_scenario_3_egal[, scenario := "Égal"]
mnist_scenario_3_plus[, scenario := "Plus"]
minst_scenario_3 <- rbind(mnist_scenario_3_moins, mnist_scenario_3_egal, mnist_scenario_3_plus)

cars_scenario_moins[, scenario := "Moins"]
cars_scenario_egal[, scenario := "Égal"]
cars_scenario_plus[, scenario := "Plus"]
cars_scenario <- rbind(cars_scenario_moins, cars_scenario_egal, cars_scenario_plus)


# Plot KLD percentage per epoch -------------------------------------------

(mnist_scenario_3 <- ggplot(minst_scenario_3, aes(x = epoch, y = kld_percentage, color = scenario)) +
  geom_line(alpha = 0.8) +
  geom_point(alpha = 0.8) +
  scale_x_continuous("Itération (ou epoch)") +
  scale_y_continuous("Pourcentage perte KLD", label=percent) +
  scale_color_discrete("Scénario de contamination") +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10)))

ggsave("kld_mnist_scenario_3.pdf", mnist_scenario_3, width = 40, height = 20, units = "cm")


(cars_scenario_graph <- ggplot(cars_scenario, aes(x = epoch, y = kld_percentage, color = scenario)) +
    geom_line(alpha = 0.8) +
    geom_point(alpha = 0.8) +
    scale_x_continuous("Itération (ou epoch)") +
    scale_y_continuous("Pourcentage perte KLD", label=percent) +
    scale_color_discrete("Scénario de contamination") +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10)))

ggsave("kld_cars.pdf", cars_scenario_graph, width = 40, height = 20, units = "cm")

                     