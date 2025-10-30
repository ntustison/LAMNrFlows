library( rmarkdown )
library( ggplot2 )

stitchedFile <- "stitched.md"

rmdFiles <- c( "format.md",
               "titlePage.md",
               "abstract.md",
               "intro.md",
               "intro_previous_work.md",
               "normalizing_flows.md",
               "comparison_df_vs_cnf.md",
               "comparison_normflow_diffusion.md",
               "latent_alignment_expanded.md",
               "cgm_overview.md",
               "hcp_ya_experiments.md",
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

