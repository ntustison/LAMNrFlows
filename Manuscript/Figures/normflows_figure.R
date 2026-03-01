library(ggplot2)
library(patchwork)
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

color_okabe_ito <- c( "#D55E00", "#0072B2", "#009E73" )
color_paul_tol <- c( "#332288", "#DDCC77", "#CC6677" )

colors_latent <- color_okabe_ito
colors_input <- color_paul_tol


args <- commandArgs(trailingOnly = TRUE)

if( length( args ) == 0 ) {
  view <- 1
} else {
  view <- as.integer(args[1])
}  


p1a_color <- color_okabe_ito[view]
p2a_color <- color_paul_tol[view]

# ==========================================
# RANGÉE 1 : Univariate Manifold Linearization
# ==========================================
base_data <- data.frame(z = seq(-4, 4, length.out = 1000))
base_data$density <- dnorm(base_data$z)

p1_a <- ggplot(base_data, aes(x = z, y = density)) +
  geom_area(fill = p1a_color, alpha = 0.2) +
  geom_line(color = p1a_color, linewidth = 1.2) +
  labs(title = "Base Distribution",
       subtitle = "Latent Space: $\\mathcal{Z} \\sim \\mathcal{N}(0, 1)$",
       x = "Latent Value ($z$)", y = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA))

# set.seed()
samples <- c(rnorm(250, mean = -1.5, sd = 0.3), 
             rnorm(150, mean = 0.2, sd = 0.4),
             rnorm(100, mean = 1.2, sd = 0.1))
complex_samples <- data.frame(x = samples)

p2_a <- ggplot(complex_samples, aes(x = x)) +
  geom_density(fill = p2a_color, alpha = 0.2, color = p2a_color, linewidth = 1.2) +
  geom_jitter(aes(y = -0.01), height = 0.005, alpha = 0.3, size = 0.8) +
  labs(title = "Input Data Distribution",
       subtitle = "Observed Space ($\\mathcal{X}$)",
       x = "Input Feature Value ($x$)", y = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA))

row_1 <- p2_a + p1_a + 
  plot_annotation(title = paste( "Single-View", view ),
                  theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
                  plot.background = element_rect(fill = "transparent", colour = NA),
                  panel.background = element_rect(fill = "transparent", colour = NA)))

# ==========================================
# RANGÉE 2 : Network Architecture
# ==========================================
gen_trapezoid <- function(x_center, width, h_left, h_right, label_name) {
  data.frame(
    x = c(x_center - width/2, x_center - width/2, x_center + width/2, x_center + width/2),
    y = c(-h_left/2, h_left/2, h_right/2, -h_right/2),
    block = label_name
  )
}

blocks_data <- rbind(
  gen_trapezoid(2, 1.5, 2.5, 1.5, "T1"),
  gen_trapezoid(6, 1.5, 2.5, 1.5, "T2"),
  gen_trapezoid(14, 1.5, 2.5, 1.5, "Tn")
)

labels_data <- data.frame(
  x = c(2, 6, 10, 14),
  y = c(0, 0, 0, 0),
  label = c("$T_1$", "$T_2$", "$\\dots$", "$T_n$")
)

arrows_fwd <- data.frame(
  x = c(0.0, 3.0, 7.0, 11.0, 15.0),
  xend = c(1.0, 5.0, 9.0, 13.0, 16.0),
  y = c(0.4, 0.4, 0.4, 0.4, 0.4),
  yend = c(0.4, 0.4, 0.4, 0.4, 0.4)
)

arrows_bwd <- data.frame(
  x = c(1.0, 5.0, 9.0, 13.0, 16.0),
  xend = c(0.0, 3.0, 7.0, 11.0, 15.0),
  y = c(-0.4, -0.4, -0.4, -0.4, -0.4),
  yend = c(-0.4, -0.4, -0.4, -0.4, -0.4)
)

io_data <- data.frame(
  x = c(-1, 17.0),
  y = c(0, 0),
  label = c(paste0( "$x^{(", view, ")}$" ), paste0( "$z^{(", view, " )}$" ))
)

row_2 <- ggplot() +
  geom_polygon(data = blocks_data, aes(x = x, y = y, group = block), 
               fill = "#ecf0f1", color = "#2c3e50", linewidth = 1) +
  geom_text(data = labels_data, aes(x = x, y = y, label = label), size = 7) +
  geom_text(data = io_data, aes(x = x, y = y, label = label), size = 7) +
  geom_segment(data = arrows_fwd, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.25, "cm"), type = "closed"), 
               linewidth = 0.8, color = p2a_color) +
  geom_segment(data = arrows_bwd, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.25, "cm"), type = "closed"), 
               linewidth = 0.8, color = p1a_color) +
  annotate("text", x = 8, y = 2.25, label = paste0( "$f^{(", view, ")}_{\\theta} : \\mathcal{X}^{(", view, ")} \\to \\mathcal{Z}^{(", view, ")}$" ), size = 8) +
  annotate("text", x = 8, y = 4.0, label = "$\\Longrightarrow$", size = 17, color="black") +
  coord_fixed(ratio = 1, xlim = c(-1, 17), ylim = c(-2.2, 1.3), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"),
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA))

# Création d'une version réduite de la rangée 2 avec des espaces (spacers) autour
# Les largeurs c(1, 4, 1) signifient : espace(1) - graphique(4) - espace(1)
# Création d'un thème transparent pour éviter la répétition de code
theme_transp <- theme(
  plot.background = element_rect(fill = "transparent", colour = NA),
  panel.background = element_rect(fill = "transparent", colour = NA)
)

# Application de la transparence au conteneur (plot_annotation) et aux espaces (&)
row_2_resized <- (plot_spacer() + row_2 + plot_spacer() + plot_layout(widths = c(1.5, 4, 1.5))) +
  plot_annotation(theme = theme_transp) & 
  theme_transp
  
# ==========================================
# ASSEMBLAGE FINAL
# ==========================================
combined_final <- wrap_elements(row_1) / wrap_elements(row_2_resized) + 
  plot_layout(heights = c(2, 1)) &
  theme(plot.margin = margin(t = 0.25, r = 0, b = -1.25, l = 0, unit = "cm"),
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA))

# Génération PDF

tex_file <- paste0( "normflows_single_view", view, ".tex" )

tikz(tex_file, width = 10, height = 6, standAlone = TRUE, bg = "transparent" )
print(combined_final)
dev.off()

tinytex::pdflatex(tex_file)