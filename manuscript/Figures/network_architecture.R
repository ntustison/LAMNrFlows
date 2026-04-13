library(ggplot2)
library(tikzDevice)

options(tikzLatex = "/Users/ntustison/Library/TinyTeX/bin/universal-darwin/pdflatex")
options(tikzLatexPackages = c(
  "\\usepackage{tikz}\n",
  "\\usepackage[active,tightpage,psfixbb]{preview}\n",
  "\\PreviewEnvironment{pgfpicture}\n",
  "\\setlength\\PreviewBorder{0pt}\n",
  "\\usepackage{amsmath}\n",
  "\\usepackage{amssymb}\n"
))

gen_trapezoid <- function(x_center, width, h_left, h_right, label_name) {
  data.frame(
    x = c(x_center - width/2, x_center - width/2, x_center + width/2, x_center + width/2),
    y = c(-h_left/2, h_left/2, h_right/2, -h_right/2),
    block = label_name
  )
}

# Génération des blocs
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

# Remplacement de x par x^{(v)} et Z_L par z^{(v)}
io_data <- data.frame(
  x = c(-0.5, 16.5),
  y = c(0, 0),
  label = c("$x^{(v)}$", "$z^{(v)}$")
)

p_network <- ggplot() +
  geom_polygon(data = blocks_data, aes(x = x, y = y, group = block), 
               fill = "#ecf0f1", color = "#2c3e50", linewidth = 1) +
  geom_text(data = labels_data, aes(x = x, y = y, label = label), size = 7) +
  geom_text(data = io_data, aes(x = x, y = y, label = label), size = 7) +
  geom_segment(data = arrows_fwd, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.25, "cm"), type = "closed"), 
               linewidth = 0.8, color = "#2980b9") +
  geom_segment(data = arrows_bwd, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.25, "cm"), type = "closed"), 
               linewidth = 0.8, color = "#c0392b") +
  
  annotate("text", x = 8, y = -2, label = "$f^{(v)}_{\\theta} : \\mathcal{X}^{(v)} \\to \\mathcal{Z}^{(v)}$", size = 10) +
  
  coord_fixed(ratio = 1, xlim = c(-1, 17), ylim = c(-2.5, 2), clip = "off") +
  theme_void()

tikz("network_architecture.tex", width = 12, height = 3.5, standAlone = TRUE)
print(p_network)
dev.off()

tinytex::pdflatex("network_architecture.tex")