library( rmarkdown )
library( ggplot2 )

stitchedFile <- "stitched.md"

rmdFiles <- c( "format.md",
               "titlePage.md",
               "abstract.md",
               "intro.md",
               "methods.md",
               "methods_applications.md",
               "methods_implementation.md",
               "results.md",
               "results_tabular_singleview.md",
               "results_tabular_multiview.md",
               "results_glow.md",
               "results_glow_2d_visualization.md",
               "results_glow_3d_mtl.md",
               "discussion.md",
               # "future_work.md",
               "acknowledgments.md",
               "references.md"
             )

for( i in 1:length( rmdFiles ) )
  {
  cat( rmdFiles[i] )
  if( i == 1 )
    {
    cmd <- paste( "cat", rmdFiles[i], ">", stitchedFile )
    } else {
    cmd <- paste( "cat", rmdFiles[i], ">>", stitchedFile )
    }
  system( cmd )  
  }

cat( '\n Pandoc rendering', stitchedFile, '\n' )
render( stitchedFile, pdf_document( number_sections = TRUE, pandoc_args = "--variable=subparagraph" ) )
render( stitchedFile, latex_document() )

