library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(latex2exp)

# 1. Chargement des données de bootstrapping
df <- read.csv("full_clinical_comparison.csv")

# 2. Préparation des données pour ggplot
# On sépare les colonnes pour avoir une ligne par comparaison (Linear vs Ablation)
df_plot <- df %>%
  pivot_longer(
    cols = c(starts_with("Delta_"), starts_with("CI_"), starts_with("p_")),
    names_to = c(".value", "Comparison"),
    names_sep = "_"
  ) %>%
  mutate(
    # Clarification des types de comparaison
    Comparison = ifelse(Comparison == "Lin", "vs. SiMLR (Linear)", "vs. Baseline (lambda = 0)"),
    # Nettoyage cosmétique des noms de variables
    Outcome = str_replace_all(Outcome, "\\.", " "),
    Outcome = str_to_title(Outcome),
    Outcome = str_replace_all(Outcome, c("Mmse" = "MMSE", "Moca" = "MoCA", "Adas" = "ADAS", 
                                         "Mpacc" = "mPACC", "Updrs" = "UPDRS", "Cdr" = "CDR"))
  ) %>%
  # Extraction des bornes de l'Intervalle de Confiance (CI)
  mutate(
    CI_Clean = str_remove_all(CI, "\\[|\\]"),
    CI_Lower = as.numeric(str_split_fixed(CI_Clean, ", ", 2)[,1]),
    CI_Upper = as.numeric(str_split_fixed(CI_Clean, ", ", 2)[,2])
  )

# 3. Labeller personnalisé pour les titres de panels
cohort_labels <- c(
  "NNL" = "NNL (Cognition)",
  "PPMI" = "PPMI (Pathology)"
)

# 4. Création du graphique
p <- ggplot(df_plot, aes(x = Delta, y = Outcome, color = Comparison, group = Comparison)) +
  # Séparation physique des mesures NNL et PPMI
  facet_grid(Dataset ~ ., scales = "free_y", space = "free_y", 
             labeller = labeller(Dataset = cohort_labels)) +
  
  # Ligne de référence à zéro (pas d'effet)
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.4) +
  
  # Barres d'erreur (95% CI) avec décalage pour éviter la superposition
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper), 
                 height = 0.4, size = 0.8, 
                 position = position_dodge(width = 0.6)) +
  
  # Points de données (Delta r)
  geom_point(size = 3, position = position_dodge(width = 0.6)) +
  
  # Palette de couleurs contrastée
scale_color_manual(values = c("vs. SiMLR (Linear)" = "#1f77b4", 
                              "vs. Baseline (lambda = 0)" = "#e31a1c"),
                   labels = c("vs. SiMLR (Linear)", expression(paste("vs. Baseline (", lambda, " = 0)")))) +
  # Titres et axes en anglais (Publication Ready)
  labs(
    title = "LAMNr Flows Clinical Predictive Power",
    subtitle = "Uplift in Pearson correlation (95% Bootstrap CI)",
    x = expression(paste("Correlation Uplift (", Delta, "r)")),
    y = "Clinical Outcome",
    color = "Model Comparison"
  ) +
  theme_minimal(base_size = 18) +
  theme(
    # strip.background = element_rect(fill = "grey92", color = "grey80"),
    strip.text = element_text(face = "bold", size = 16, color = "black", hjust = 0),
    panel.spacing = unit(1.5, "lines"),
    legend.position = "bottom",
    axis.text.y = element_text(size = 14),
    plot.title = element_text(face = "bold", size = 24)
  )

# 5. Sauvegarde en haute résolution
ggsave("clinical_comparison_multipanel.png", plot = p, width = 12, height = 7, dpi = 300)