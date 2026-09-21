library(ggplot2)
library(patchwork)
library(MASS)
library(tikzDevice)

# 1. Configuration LaTeX
options(tikzLatex = "/Users/ntustison/Library/TinyTeX/bin/universal-darwin/pdflatex")
options(tikzLatexPackages = c(
  "\\usepackage{tikz}\n",
  "\\usepackage[active,tightpage,psfixbb]{preview}\n",
  "\\PreviewEnvironment{pgfpicture}\n",
  "\\setlength\\PreviewBorder{0pt}\n",
  "\\usepackage{amsmath}\n",
  "\\usepackage{amssymb}\n",
  "\\usepackage{amsfonts}\n"
))

# ==========================================
# PARTIE A : Univariate Manifold Linearization
# ==========================================
base_data <- data.frame(z = seq(-4, 4, length.out = 1000))
base_data$density <- dnorm(base_data$z)

subtitle_text <- "Latent Space: $\\mathcal{Z} \\sim \\mathcal{N}(0, 1)$"

p1_a <- ggplot(base_data, aes(x = z, y = density)) +
  geom_area(fill = "#3498db", alpha = 0.2) +
  geom_line(color = "#3498db", linewidth = 1.2) +
  labs(title = "Base Distribution",
       subtitle = subtitle_text,
       x = "Latent Value ($z$)", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

set.seed(42)
samples <- c(rnorm(150, mean = -1.5, sd = 0.6), 
             rnorm(250, mean = 1.2, sd = 0.9))
complex_samples <- data.frame(x = samples)

p2_a <- ggplot(complex_samples, aes(x = x)) +
  geom_density(fill = "#e74c3c", alpha = 0.2, color = "#e74c3c", linewidth = 1.2) +
  geom_jitter(aes(y = -0.01), height = 0.005, alpha = 0.3, size = 0.8) +
  labs(title = "Complex Data Distribution",
       subtitle = "Observed Space ($\\mathcal{X}$)",
       x = "Feature Value ($x$)", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

plot_a <- p1_a + p2_a + 
  plot_annotation(title = "Univariate Manifold Linearization",
                  theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)))

# ==========================================
# PARTIE B : Latent Alignment Constraint
# ==========================================
set.seed(123)
n <- 300 

rot_matrix <- function(theta) {
  matrix(c(cos(theta), sin(theta), -sin(theta), cos(theta)), 2, 2)
}

mu1 <- c(3, 2); sigma1 <- matrix(c(2, 1.5, 1.5, 2), 2)
mu2 <- c(-2, -1); sigma2 <- matrix(c(1, -0.2, -0.2, 3), 2)
mu3 <- c(1, -4); sigma3 <- matrix(c(2.5, 0, 0, 0.5), 2)

df1_unaligned <- data.frame(mvrnorm(n, mu1, sigma1), View = "View 1")
df2_unaligned <- data.frame(mvrnorm(n, mu2, sigma2), View = "View 2")
df3_unaligned <- data.frame(mvrnorm(n, mu3, sigma3), View = "View 3")
df_unaligned <- rbind(df1_unaligned, df2_unaligned, df3_unaligned)
colnames(df_unaligned) <- c("X1", "X2", "View")

eigen1 <- eigen(sigma1)$values
eigen2 <- eigen(sigma2)$values
eigen3 <- eigen(sigma3)$values

angles_jitter <- rnorm(3, mean = 0, sd = 0.15) 

sigma1_align <- rot_matrix(angles_jitter[1]) %*% diag(eigen1) %*% t(rot_matrix(angles_jitter[1]))
sigma2_align <- rot_matrix(angles_jitter[2]) %*% diag(eigen2) %*% t(rot_matrix(angles_jitter[2]))
sigma3_align <- rot_matrix(angles_jitter[3]) %*% diag(eigen3) %*% t(rot_matrix(angles_jitter[3]))

df1_aligned <- data.frame(mvrnorm(n, c(0,0), sigma1_align), View = "View 1")
df2_aligned <- data.frame(mvrnorm(n, c(0,0), sigma2_align), View = "View 2")
df3_aligned <- data.frame(mvrnorm(n, c(0,0), sigma3_align), View = "View 3")
df_aligned <- rbind(df1_aligned, df2_aligned, df3_aligned)
colnames(df_aligned) <- c("Z1", "Z2", "View")

view_colors <- c("View 1" = "#e74c3c", "View 2" = "#2ecc71", "View 3" = "#3498db")

# Panneau Gauche (Avant)
p1_b <- ggplot(df_unaligned, aes(x = X1, y = X2, color = View)) +
  geom_point(alpha = 0.4, size = 1) +
  stat_ellipse(linewidth = 1.2, level = 0.95) +
  scale_color_manual(values = view_colors) +
  labs(title = "Before Alignment", subtitle = "", x = "", y = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# Panneau Droite (Après)
p2_b <- ggplot(df_aligned, aes(x = Z1, y = Z2, color = View)) +
  geom_point(alpha = 0.4, size = 1) +
  stat_ellipse(linewidth = 1.2, level = 0.95) +
  scale_color_manual(values = view_colors) +
  labs(title = "After Alignment", subtitle = "", x = "", y = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# Formule Centrale
formula_text <- "\\Large $\\mathcal{L}_{\\text{align}} \\Bigl(\\bigl\\{\\phi^{(v)}_{\\psi}(z^{(v)}_{S,n})\\bigr\\}_{v,n}\\Bigr) \\Rightarrow$"
p_text <- ggplot() +
  annotate("text", x = 0.5, y = 0.5, label = formula_text, size = 6) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") + 
  theme_void()

combined_b_base <- p1_b + p2_b + 
  plot_layout(widths = c(1, 1), guides = "collect") + 
  plot_annotation(title = "Latent Alignment Constraint",
                  theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)))

# Insertion de la formule au premier plan
plot_b <- combined_b_base + 
  inset_element(p_text, 
                left = 0.35, bottom = 0.4, 
                right = 0.65, top = 0.6, 
                align_to = 'full', clip = FALSE, on_top = TRUE)

# ==========================================
# PARTIE C : Combinaison Finale A / B
# ==========================================
# L'utilisation de wrap_elements() garantit que chaque partie conserve son propre formatage interne
combined_final <- wrap_elements(plot_a) / wrap_elements(plot_b) + 
  plot_annotation(tag_levels = 'A') & 
  theme(plot.tag = element_text(size = 20, face = "bold"))

# Génération (Largeur et Hauteur augmentées pour correspondre à une pleine page)
tikz("combined_LAMNr_figures.tex", width = 12, height = 10, standAlone = TRUE)
print(combined_final)
dev.off()

tinytex::pdflatex("combined_LAMNr_figures.tex")