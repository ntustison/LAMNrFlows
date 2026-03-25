library(ggplot2)
library(dplyr)
library(stringr)

# 1. Chargement des données
df <- read.csv("clinical_bootstrap_results.csv")

# 2. Nettoyage et préparation des données
df_plot <- df %>%
  mutate(
    # Nettoyage des noms (ex: recall.delayed -> Recall Delayed)
    Outcome = str_replace_all(Outcome, "\\.", " "),
    Outcome = str_to_title(Outcome),
    # Création d'un label unique pour le tri
    Label = paste0(Dataset, ": ", Outcome)
  ) %>%
  # Tri par Dataset puis par valeur de Delta
  arrange(Dataset, Delta) %>%
  mutate(Label = factor(Label, levels = Label))

# 3. Génération du Forest Plot
ggplot(df_plot, aes(x = Delta, y = Label, color = Dataset)) +
  # Ligne verticale de référence à zéro
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", size = 0.8, alpha = 0.6) +
  # Barres d'erreur (95% CI)
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper), height = 0.3, size = 0.8) +
  # Points de données (Delta r)
  geom_point(size = 3.5) +
  # Couleurs personnalisées
  scale_color_manual(values = c("NNL" = "#1f77b4", "PPMI" = "#ff7f0e")) +
  # Thème et labels
  labs(
    title = "Clinical Prediction Uplift: LAMNr vs SiMLR Baseline",
    subtitle = "95% Confidence Intervals from 1000 Bootstrap Resamples",
    x = expression(paste("Uplift in Prediction (", Delta, "r = ", r[LAMNr], " - ", r[SiMLR], ")")),
    y = NULL,
    color = "Cohort"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.text.y = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 16)
  )

# 4. Sauvegarde
ggsave("clinical_uplift_forest_plot.png", width = 10, height = 7, dpi = 300)