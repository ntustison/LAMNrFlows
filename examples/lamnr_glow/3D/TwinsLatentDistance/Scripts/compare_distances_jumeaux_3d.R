library(tidyverse)

base_dir <- "/Users/ntustison/Data/Public/OpenNeuro/ds004169"

# 1. Charger les métadonnées des participants
participants <- read_tsv(paste0(base_dir, "/manifests/participants_short.tsv"), show_col_types = FALSE)

# 2. Fonction pour lire une matrice NxN et la convertir en format long
read_pairwise_matrix <- function(file_path, condition_name) {
  cat(sprintf("[info] Loading pairwise matrix from '%s'...\n", file_path))
  
  df_wide <- read_csv(file_path, show_col_types = FALSE)
  
  df_long <- df_wide %>%
    pivot_longer(
      cols = -subject,
      names_to = "target_file",
      values_to = "Distance"
    ) %>%
    mutate(
      source_sub = str_extract(subject, "sub-\\d+"),
      target_sub = str_extract(target_file, "sub-\\d+"),
      Condition = condition_name,
      Niveau_Latent = "total_distance"
    ) %>%
    filter(source_sub != target_sub) %>%
    select(source_sub, target_sub, Distance, Condition, Niveau_Latent)
  
  return(df_long)
}

# 3. Charger et combiner les deux matrices
df_skull <- read_pairwise_matrix(paste0(base_dir, "/output_brain_3d/distances_pairwise_matrix_whole_head.csv"), "(whole head)")
df_brain <- read_pairwise_matrix(paste0(base_dir, "/output_brain_3d/distances_pairwise_matrix_brain.csv"), "(brain only)")

all_distances <- bind_rows(df_skull, df_brain)

# 4. Croiser avec les familles
all_distances <- all_distances %>%
  left_join(participants %>% select(participant_id, family_id), by = c("source_sub" = "participant_id")) %>%
  rename(source_fam = family_id) %>%
  left_join(participants %>% select(participant_id, family_id), by = c("target_sub" = "participant_id")) %>%
  rename(target_fam = family_id) %>%
  drop_na(source_fam, target_fam) %>%
  mutate(Relation = if_else(source_fam == target_fam, "Twins", "Unrelated"))

# 5. Calcul des rangs puis filtrage exclusif des jumeaux
levels_to_test <- c("total_distance")

df_twins <- all_distances %>%
  group_by(source_sub, Condition, Niveau_Latent) %>%
  # Le rang est évalué contre TOUTE la cohorte (Twins + Unrelated)
  mutate(Rang = rank(Distance, ties.method = "min")) %>%
  ungroup() %>%
  # Isolation des jumeaux pour le tracé et les statistiques
  filter(Relation == "Twins") %>%
  mutate(
    Niveau_Latent = factor(Niveau_Latent, levels = levels_to_test),
    Groupe = paste(Relation, Condition)
  )

# 6. Test de Wilcoxon Apparié (Twins: brain only vs with skull)
df_twins_wide <- df_twins %>%
  select(source_sub, target_sub, Niveau_Latent, Condition, Rang) %>%
  pivot_wider(names_from = Condition, values_from = Rang) %>%
  drop_na()

stats_pvalues <- df_twins_wide %>%
  group_by(Niveau_Latent) %>%
  summarize(
    p_value = wilcox.test(
      x = `(brain only)`, 
      y = `(whole head)`, 
      paired = TRUE,  
      alternative = "less" # Teste si l'extraction crânienne (brain only) améliore (réduit) le rang de similarité
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    x_pos = as.numeric(Niveau_Latent),
    label = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n--- P-VALUES (Paired Test: Brain Only vs Whole Head for Twins) ---\n")
print(as.data.frame(stats_pvalues))

# 7. Définition de la palette restreinte
palette_twins <- c(
  "Twins (whole head)" = "#ea801c", 
  "Twins (brain only)" = "#1a80bb"
)

# 8. Création de l'histogramme avec les médianes
# 8. Création de l'histogramme avec étiquettes de médiane
# Calcul des médianes et préparation des étiquettes
medians_twins <- df_twins %>%
  group_by(Groupe) %>%
  summarize(
    Mediane_Rang = median(Rang),
    .groups = "drop"
  ) %>%
  mutate(
    # Crée un texte formaté pour l'affichage (ex: "Médiane: 12.5")
    label_text = paste0("Median: ", round(Mediane_Rang, 1))
  )

p <- ggplot(df_twins, aes(x = Rang, fill = Groupe)) +
  geom_histogram(
    position = "identity",
    alpha = 0.6,
    binwidth = 5,
    color = "white",
    linewidth = 0.2
  ) +
  # Lignes de médiane
  geom_vline(
    data = medians_twins,
    aes(xintercept = Mediane_Rang, color = Groupe),
    linetype = "dashed",
    linewidth = 1,
    show.legend = FALSE
  ) +
  # AJOUT : Étiquettes textuelles des médianes
geom_text(
    data = medians_twins,
    aes(x = Mediane_Rang, y = Inf, label = label_text),
    color = "black",
    angle = 90,             # Rotation du texte à 90 degrés
    vjust = 1.5,            # Ajustement latéral par rapport à la ligne pointillés
    hjust = 1.1,            # Ajustement vertical pour décoller du haut du cadre
    size = 4.0,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = palette_twins) +
  scale_color_manual(values = palette_twins) +
  labs(
    title = "Twin Latent Similarity Ranking: Whole Head vs Brain Only",
    subtitle = "(Distribution of ranks. Paired Wilcoxon: *** p < 0.001)",
    x = "Twin Similarity Rank",
    y = "Frequency (Count of Twin Pairs)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, margin = margin(b = 20), color = "grey30"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    panel.grid.minor = element_blank()
  )

# Sauvegarde
ggsave("histogramme_rangs_twins_with_labels.png", plot = p, width = 10, height = 5, dpi = 300, bg = "white")
cat("\n[ok] Histogram with median labels saved: histogramme_rangs_twins_with_labels.png\n")