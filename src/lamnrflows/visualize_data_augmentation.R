# Installation si nécessaire : install.packages(c("ggplot2", "dplyr", "tidyr", "viridis"))
library(ggplot2)
library(dplyr)
library(tidyr)

# 1. Configuration des paramètres (extraits de votre script)
aug_iterations <- 1000  # Valeur de la variable ${aug_iterations}
t <- 0:aug_iterations

# 2. Fonctions de calcul des trajectoires
calc_linear <- function(start, end, t, T_max) {
  start + (t / T_max) * (end - start)
}

calc_cosine <- function(start, end, t, T_max) {
  end + 0.5 * (start - end) * (1 + cos(pi * t / T_max))
}

calc_exponential <- function(start, end, t, T_max) {
  start * (end / start)^(t / T_max)
}

calc_power <- function(start, end, t, T_max, p = 2) {
  (start - end) * (1 - t / T_max)^p + end
}

# # Augmentation schedule
# aug_params_phase1="noise_std:cos:0.05->0.015@${aug_iterations},\
# sd_affine:cos:0.05->0.01@${aug_iterations},\
# sd_deformation:linear:12.0->0.6@${aug_iterations},\
# sd_simulated_bias_field:cos:0.20->0.03@${aug_iterations},\
# sd_histogram_warping:cos:0.04->0.008@${aug_iterations}"

# 3. Création des données basées sur aug_params_phase1
df_trajectories <- data.frame(iteration = t) %>%
  mutate(
    sd_simulated_bias_field = calc_cosine(0.20, 0.03, iteration, aug_iterations),
    sd_histogram_warping = calc_cosine(0.04, 0.008, iteration, aug_iterations),
    noise_std = calc_cosine(0.05, 0.015, iteration, aug_iterations),
    sd_affine = calc_cosine(0.05, 0.01, iteration, aug_iterations),
    sd_deformation = calc_linear(12.0, 0.6, iteration, aug_iterations)
  )

# 4. Préparation du format long pour ggplot
df_long <- df_trajectories %>%
  pivot_longer(cols = -iteration, names_to = "parametre", values_to = "valeur")

# 5. Graphique principal avec facettes (échelles libres)
aug_plot <- ggplot(df_long, aes(x = iteration, y = valeur, color = parametre)) +
  geom_line(linewidth = 1.1, alpha = 0.9) +
  facet_wrap(~parametre, scales = "free_y", ncol = 3) +
  theme_minimal(base_size = 14) +
  scale_color_viridis_d(option = "viridis") +
  labs(
    title = "Augmentation schedule",
    x = "Iterations",
    y = "Parameter value"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank()
  )
ggsave("augmentation_schedule.png", aug_plot, width = 12, height = 4, dpi = 300)  