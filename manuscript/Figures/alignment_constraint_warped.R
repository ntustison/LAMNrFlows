library(ggplot2)
library(MASS)
library(tikzDevice)

# 1. Configuration LaTeX
latex_path <- Sys.which("xelatex")
if(latex_path != "") options(tikzLatex = latex_path)

options(tikzLatexPackages = c(
  "\\usepackage{tikz}\n",
  "\\usepackage[active,tightpage,psfixbb]{preview}\n",
  "\\PreviewEnvironment{pgfpicture}\n",
  "\\setlength\\PreviewBorder{0pt}\n",
  "\\usepackage{amsmath}\n",
  "\\usepackage{amssymb}\n"
))

# ---------------------------------------------------------
# Fonctions de déformation (Warping)
# ---------------------------------------------------------
warp_manifold <- function(x, y, bending = 0.1) {
  y_new <- y + bending * (x^2)
  return(data.frame(x = x, y = y_new))
}

get_warped_ellipse <- function(mu, sigma, bending = 0.1, level = 0.95, n = 100) {
  theta <- seq(0, 2*pi, length.out = n)
  circle <- cbind(cos(theta), sin(theta))
  
  ed <- eigen(sigma)
  # Chi-2 pour intervalles de confiance 2D exacts
  scale <- sqrt(ed$values) * sqrt(qchisq(level, df = 2))
  ellipse_pts <- circle %*% diag(scale) %*% t(ed$vectors)
  
  x <- ellipse_pts[,1] + mu[1]
  y <- ellipse_pts[,2] + mu[2]
  
  return(warp_manifold(x, y, bending))
}

# ---------------------------------------------------------
# 2. Simulation des données
# ---------------------------------------------------------
set.seed(123)
n <- 1000 

rot_matrix <- function(theta) {
  matrix(c(cos(theta), sin(theta), -sin(theta), cos(theta)), 2, 2)
}

color_okabe_ito <- c( "#D55E00", "#0072B2", "#009E73" )
colors_latent <- color_okabe_ito

# Paramètres initiaux
mu1 <- c(3, 2); sigma1 <- matrix(c(2, 1.5, 1.5, 2), 2)
mu2 <- c(-2, -1); sigma2 <- matrix(c(1, -0.2, -0.2, 3), 2)
mu3 <- c(1, -4); sigma3 <- matrix(c(2.5, 0, 0, 0.5), 2)

# Paramètres de courbure
bend1 <- 0.10
bend2 <- -0.10
bend3 <- -0.05

# Avant alignement (Appliquer la déformation)
pts1_raw <- mvrnorm(n, mu1, sigma1)
df1_unaligned <- warp_manifold(pts1_raw[,1], pts1_raw[,2], bending = bend1)
df1_unaligned$View <- "View 1"; df1_unaligned$State <- "Before"

pts2_raw <- mvrnorm(n, mu2, sigma2)
df2_unaligned <- warp_manifold(pts2_raw[,1], pts2_raw[,2], bending = bend2)
df2_unaligned$View <- "View 2"; df2_unaligned$State <- "Before"

pts3_raw <- mvrnorm(n, mu3, sigma3)
df3_unaligned <- warp_manifold(pts3_raw[,1], pts3_raw[,2], bending = bend3)
df3_unaligned$View <- "View 3"; df3_unaligned$State <- "Before"

# Après alignement (Espace Euclidien linéaire)
eigen1 <- c(1, 1) + 0.1 * eigen(sigma1)$values
eigen2 <- c(1, 1) + 0.2 * eigen(sigma2)$values
eigen3 <- c(1, 1) + 0.3 * eigen(sigma3)$values

angles_jitter <- rnorm(3, mean = 0, sd = 0.15) 

sigma1_align <- rot_matrix(angles_jitter[1]) %*% diag(eigen1) %*% t(rot_matrix(angles_jitter[1]))
sigma2_align <- rot_matrix(angles_jitter[2]) %*% diag(eigen2) %*% t(rot_matrix(angles_jitter[2]))
sigma3_align <- rot_matrix(angles_jitter[3]) %*% diag(eigen3) %*% t(rot_matrix(angles_jitter[3]))

# Extraction explicite (x, y) pour éviter l'erreur rbind
pts1_align <- mvrnorm(n, c(25,0), sigma1_align)
df1_aligned <- data.frame(x = pts1_align[,1], y = pts1_align[,2], View = "View 1", State = "After")

pts2_align <- mvrnorm(n, c(25,0), sigma2_align)
df2_aligned <- data.frame(x = pts2_align[,1], y = pts2_align[,2], View = "View 2", State = "After")

pts3_align <- mvrnorm(n, c(25,0), sigma3_align)
df3_aligned <- data.frame(x = pts3_align[,1], y = pts3_align[,2], View = "View 3", State = "After")

# Combinaison
df_combined <- rbind(df1_unaligned, df2_unaligned, df3_unaligned,
                     df1_aligned, df2_aligned, df3_aligned)
df_combined$State <- factor(df_combined$State, levels = c("Before", "After"))

# Génération des ellipses déformées pour "Before"
levels_to_draw <- c(0.95, 0.75, 0.5, 0.25)
warped_ellipses <- data.frame()

for(lvl in levels_to_draw) {
  e1 <- get_warped_ellipse(mu1, sigma1, bending = bend1, level = lvl)
  e1$View <- "View 1"; e1$Level <- lvl
  
  e2 <- get_warped_ellipse(mu2, sigma2, bending = bend2, level = lvl)
  e2$View <- "View 2"; e2$Level <- lvl
  
  e3 <- get_warped_ellipse(mu3, sigma3, bending = bend3, level = lvl)
  e3$View <- "View 3"; e3$Level <- lvl
  
  warped_ellipses <- rbind(warped_ellipses, e1, e2, e3)
}

# ---------------------------------------------------------
# 3. Création du Graphique Unique
# ---------------------------------------------------------
view_colors <- c("View 1" = colors_latent[1], "View 2" = colors_latent[2], "View 3" = colors_latent[3])
formula_text <- "$\\mathcal{L}_{\\text{align}} \\Bigl(\\bigl\\{\\phi^{(v)}_{\\psi}(z^{(v)}_{S,n})\\bigr\\}_{v,n}\\Bigr)\\Rightarrow$"

p_single <- ggplot(df_combined, aes(x = x, y = y, color = View, linetype = State)) +
  geom_point(aes(shape = State), size = 2, alpha = 0.25) +
  
  annotate("label", x = 12.5, y = 0.5, 
           label = formula_text, fill = 'grey10', colour = 'black', alpha = 0.25,
           size = 6, fontface = "bold", label.padding = unit(1.00, "lines"), label.r = unit(0.1, "lines")) +
  annotate("text", x = 0.5, y = 10.5, label = "Pre-alignment", size = 5, fontface = "bold") +
  annotate("text", x = 25., y = 10.5, label = "Post-alignment", size = 5, fontface = "bold") +
  annotate("text", x = 1.5, y = 5.0, label = "View 1", size = 5, fontface = "bold") +
  annotate("text", x = -4.75, y = 2.0, label = "View 2", size = 5, fontface = "bold") +
  annotate("text", x = 2.75, y = -2.15, label = "View 3", size = 5, fontface = "bold") +
  
  # --- Dessin manuel des ellipses "Before" (Warped) ---
  geom_path(data = subset(warped_ellipses, Level == 0.95), aes(x = x, y = y), linewidth = 2, alpha = 0.5, linetype = "solid") +
  geom_path(data = subset(warped_ellipses, Level == 0.75), aes(x = x, y = y), linewidth = 2, alpha = 0.35, linetype = "solid") +
  geom_path(data = subset(warped_ellipses, Level == 0.50), aes(x = x, y = y), linewidth = 2, alpha = 0.25, linetype = "solid") +
  geom_path(data = subset(warped_ellipses, Level == 0.25), aes(x = x, y = y), linewidth = 2, alpha = 0.15, linetype = "solid") +

  # --- Dessin automatique des ellipses "After" (Linéaires) ---
  stat_ellipse(data = subset(df_combined, State == "After"), linewidth = 2, level = 0.95, alpha = 0.5) +
  stat_ellipse(data = subset(df_combined, State == "After"), linewidth = 2, level = 0.75, alpha = 0.35) +
  stat_ellipse(data = subset(df_combined, State == "After"), linewidth = 2, level = 0.5, alpha = 0.25) +
  stat_ellipse(data = subset(df_combined, State == "After"), linewidth = 2, level = 0.25, alpha = 0.15) +

  scale_color_manual(values = view_colors) +
  scale_linetype_manual(values = c("Before" = "solid", "After" = "solid")) +
  
  # --- C'EST CECI QUI GARANTIT LA MÊME ÉCHELLE ---
  coord_fixed(ratio = 1) + 
  
  labs(title = "", subtitle = "", x = "", y = "") +
  theme_void() +
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA)
  )

# ---------------------------------------------------------
# 4. Génération
# ---------------------------------------------------------
tikz("latent_alignment_single.tex", width = 12, height = 6, standAlone = TRUE, bg = "transparent" )
print(p_single)
dev.off()

tinytex::pdflatex("latent_alignment_single.tex")