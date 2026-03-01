library(ggplot2)
library(MASS)
library(tikzDevice)

# 1. Configuration LaTeX
latex_path <- Sys.which("pdflatex")
if(latex_path != "") options(tikzLatex = latex_path)

options(tikzLatexPackages = c(
  "\\usepackage{tikz}\n",
  "\\usepackage[active,tightpage,psfixbb]{preview}\n",
  "\\PreviewEnvironment{pgfpicture}\n",
  "\\setlength\\PreviewBorder{0pt}\n",
  "\\usepackage{amsmath}\n",
  "\\usepackage{amssymb}\n"
))

# 2. Simulation des données
set.seed(123)
n <- 1000 

rot_matrix <- function(theta) {
  matrix(c(cos(theta), sin(theta), -sin(theta), cos(theta)), 2, 2)
}

color_okabe_ito <- c( "#D55E00", "#0072B2", "#009E73" )
color_paul_tol <- c( "#332288", "#DDCC77", "#CC6677" )

colors_latent <- color_okabe_ito
colors_input <- color_paul_tol


# Avant alignement
mu1 <- c(3, 2); sigma1 <- matrix(c(2, 1.5, 1.5, 2), 2)
mu2 <- c(-2, -1); sigma2 <- matrix(c(1, -0.2, -0.2, 3), 2)
mu3 <- c(1, -4); sigma3 <- matrix(c(2.5, 0, 0, 0.5), 2)

df1_unaligned <- data.frame(mvrnorm(n, mu1, sigma1), View = "View 1", State = "Before")
df2_unaligned <- data.frame(mvrnorm(n, mu2, sigma2), View = "View 2", State = "Before")
df3_unaligned <- data.frame(mvrnorm(n, mu3, sigma3), View = "View 3", State = "Before")

# Après alignement
eigen1 <- eigen(sigma1)$values
eigen2 <- eigen(sigma2)$values
eigen3 <- eigen(sigma3)$values

angles_jitter <- rnorm(3, mean = 0, sd = 0.15) 

sigma1_align <- rot_matrix(angles_jitter[1]) %*% diag(eigen1) %*% t(rot_matrix(angles_jitter[1]))
sigma2_align <- rot_matrix(angles_jitter[2]) %*% diag(eigen2) %*% t(rot_matrix(angles_jitter[2]))
sigma3_align <- rot_matrix(angles_jitter[3]) %*% diag(eigen3) %*% t(rot_matrix(angles_jitter[3]))

df1_aligned <- data.frame(mvrnorm(n, c(20,0), sigma1_align), View = "View 1", State = "After")
df2_aligned <- data.frame(mvrnorm(n, c(20,0), sigma2_align), View = "View 2", State = "After")
df3_aligned <- data.frame(mvrnorm(n, c(20,0), sigma3_align), View = "View 3", State = "After")

# Combinaison des données en un seul data.frame
df_combined <- rbind(df1_unaligned, df2_unaligned, df3_unaligned,
                     df1_aligned, df2_aligned, df3_aligned)
colnames(df_combined)[1:2] <- c("Dim1", "Dim2")

# Définition de l'ordre d'affichage (Before dessiné en dessous de After)
df_combined$State <- factor(df_combined$State, levels = c("Before", "After"))

# 3. Création du Graphique Unique
view_colors <- c("View 1" = colors_latent[1], "View 2" = colors_latent[2], "View 3" = colors_latent[3])
formula_text <- "$\\mathcal{L}_{\\text{align}} \\Bigl(\\bigl\\{\\phi^{(v)}_{\\psi}(z^{(v)}_{S,n})\\bigr\\}_{v,n}\\Bigr)\\Rightarrow$"

p_single <- ggplot(df_combined, aes(x = Dim1, y = Dim2, color = View, linetype = State)) +
  # Points: Before (cercles vides, très transparents) | After (cercles pleins, plus visibles)
  # annotate(
  #   'rect',
  #   xmin = -7,
  #   xmax = 27,
  #   ymin = -9,
  #   ymax = 9,
  #   alpha = 0.15, # This was put back to 0.5
  #   fill = 'blue',
  #   col = 'black',
  #   label.r = unit(0.1, "lines")) +
  geom_point(aes(shape = State), size = 2, alpha = 0.25) +
  annotate("label", x = 10., y = 0.5, 
                  label = formula_text,
                  fill = 'grey10',
                  colour = 'black',
                  alpha = 0.25,
                  size = 6, fontface = "bold", 
                  label.padding = unit(1.00, "lines"),
                  label.r = unit(0.1, "lines")) +
  annotate("text", x = 0.5, y = 6.5, label = "Pre-alignment", size = 5, fontface = "bold") +
  annotate("text", x = 20., y = 6.5, label = "Post-alignment", size = 5, fontface = "bold") +
  stat_ellipse(linewidth = 2, level = 0.95, alpha = 0.5) +
  stat_ellipse(linewidth = 2, level = 0.75, alpha = 0.35) +
  stat_ellipse(linewidth = 2, level = 0.5, alpha = 0.25) +
  stat_ellipse(linewidth = 2, level = 0.25, alpha = 0.15) +
  annotate("text", x = 3.0, y = 5.0, label = "View 1", size = 5, fontface = "bold") +
  annotate("text", x = -4.5, y = 2.0, label = "View 2", size = 5, fontface = "bold") +
  annotate("text", x = 2.75, y = -2.15, label = "View 3", size = 5, fontface = "bold") +
  scale_color_manual(values = view_colors) +
  scale_linetype_manual(values = c("Before" = "solid", "After" = "solid")) +
  labs(title = "",
       subtitle = "",
       x = "", 
       y = "") +
  theme_void() +
  theme(legend.position = "none")
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA)
  )

# 4. Génération
tikz("latent_alignment_single.tex", width = 12, height = 6, standAlone = TRUE, bg = "transparent" )
print(p_single)
dev.off()

tinytex::pdflatex("latent_alignment_single.tex")