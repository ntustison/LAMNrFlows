---
title:
author: 
address:
output:
  pdf_document:
    fig_caption: true
    latex_engine: pdflatex
    keep_tex: yes
    number_sections: true
    toc: true
  word_document:
    fig_caption: true
bibliography:
  - references.bib
  - referencesLung.bib
csl: medical-image-analysis.csl
longtable: true
urlcolor: blue
header-includes:
  - \usepackage{pifont}
  - \newcommand{\cmark}{\ding{51}} 
  - \newcommand{\xmark}{\ding{55}} 
  - \newcommand{\pmark}{\(\triangle\)} 
  - \usepackage{longtable}
  - \usepackage{graphicx}
  - \usepackage{rotating}
  - \usepackage{array}
  - \usepackage{booktabs}
  - \usepackage{textcomp}
  - \usepackage{xcolor}
  - \usepackage{colortbl}
  - \usepackage{geometry}
  - \usepackage{subcaption}
  - \usepackage{lineno}
  - \usepackage{makecell}
  - \usepackage{pdflscape}
  - \usepackage[misc]{ifsym}
  - \usepackage{amsmath}
  - \usepackage{tikz}
  - \definecolor{listcomment}{rgb}{0.0,0.5,0.0}
  - \definecolor{listkeyword}{rgb}{0.0,0.0,0.5}
  - \definecolor{listnumbers}{gray}{0.65}
  - \definecolor{listlightgray}{gray}{0.955}
  - \definecolor{listwhite}{gray}{1.0}
  - \usepackage{amsmath,amssymb}
  - \usepackage{graphicx}
  - \usepackage{xcolor}
  - \usepackage{lmodern}
  - \usepackage{tikz}
  - \usetikzlibrary{arrows,arrows.meta,calc,decorations.pathreplacing,positioning,shadings}
  - \definecolor{softblue}{RGB}{221,229,239}
  - \definecolor{deepblue}{RGB}{47,96,157}
  - \definecolor{softred}{RGB}{247,218,216}
  - \definecolor{deepred}{RGB}{181,46,48}
geometry: margin=1.0in
fontsize: 11pt
linestretch: 1.5
mainfont: Georgia
---

\linenumbers