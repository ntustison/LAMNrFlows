library(ggplot2)
library(patchwork)

# 1. Distribution de Base (Gaussienne Standard)
base_data <- data.frame(z = seq(-4, 4, length.out = 1000))
base_data$density <- dnorm(base_data$z)

p1 <- ggplot(base_data, aes(x = z, y = density)) +
  geom_area(fill = "#3498db", alpha = 0.2) +
  geom_line(color = "#3498db", size = 1.2) +
  labs(title = "Base Distribution",
       subtitle = "Latent Space (Z) ~ N(0, 1)",
       x = "Latent Value (z)", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# 2. Distribution Complexe (Mélange de Gaussiennes)
set.seed(42)
# Simuler des données d'entrée (échantillons)
samples <- c(rnorm(150, mean = -1.5, sd = 0.6), 
             rnorm(250, mean = 1.2, sd = 0.9))
complex_samples <- data.frame(x = samples)

p2 <- ggplot(complex_samples, aes(x = x)) +
  geom_density(fill = "#e74c3c", alpha = 0.2, color = "#e74c3c", size = 1.2) +
  # Ajouter les points d'échantillonnage en bas (Rug Plot)
  geom_jitter(aes(y = -0.02), height = 0.005, alpha = 0.3, size = 0.8) +
  labs(title = "Complex Data Distribution",
       subtitle = "Observed Space (X)",
       x = "Feature Value (x)", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Combinaison des graphiques
final_plot <- p1 + p2 + 
  plot_annotation(title = "Univariate Manifold Linearization",
                  theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5)))

# Affichage
print(final_plot)

# Sauvegarde
ggsave("univariate_linearization.pdf", final_plot, width = 10, height = 4)