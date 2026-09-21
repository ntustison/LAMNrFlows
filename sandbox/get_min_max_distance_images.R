library( ANTsR )
library( png )

n <- 3
slice_index <- 115
input_dir <- "/Users/ntustison/Desktop/lamnr_glow_dlbs/output96x128_vicreg"
output_dir <- paste0( input_dir, "/min_max_distance_images" )
dir.create( output_dir, showWarnings = FALSE )

latent_distances_file <- paste0( input_dir, "/distances_dlbs_wave2_to_antsx_template.csv" )
latent_distances <- read.csv( latent_distances_file )

latent_distances <- latent_distances[ order( latent_distances$total_distance ),]

total_number_of_images <- nrow( latent_distances )

for ( i in 1:n )
  {
  cat( "i = ", i, "\n" )

  image_file <- latent_distances$path[i]
  cat( "    Closest image_file = ", image_file, "\n" )
  image <- antsImageRead( image_file )
  slice <- as.array( extractSlice( image, slice_index, 3 ) )  
  slice <- ( slice - min( slice ) ) / ( max( slice ) - min( slice ) )
  slice <- t(slice)
  writePNG( slice, paste0( output_dir, "/latent_distance_closest_total_", i, ".png" ) )

  image_file <- latent_distances$path[total_number_of_images - i + 1]
  cat( "    Furthest image_file = ", image_file, "\n" )
  image <- antsImageRead( image_file )
  slice <- as.array( extractSlice( image, slice_index, 3 ) )  
  slice <- ( slice - min( slice ) ) / ( max( slice ) - min( slice ) )
  slice <- t(slice)
  writePNG( slice, paste0( output_dir, "/latent_distance_furthest_total_", i, ".png" ) )
  }

for( l in 1:5 )
  {
  col_index <- l + 2
  cat( "Sorting distances based on layer ", colnames(latent_distances)[col_index], "\n" )
  latent_distances <- latent_distances[ order( latent_distances[, col_index] ),]
  
  for ( i in 1:n )
    {
    cat( "i = ", i, "\n" )

    image_file <- latent_distances$path[i]
    cat( "    Closest image_file = ", image_file, "\n" )
    image <- antsImageRead( image_file )
    slice <- as.array( extractSlice( image, slice_index, 3 ) )  
    slice <- ( slice - min( slice ) ) / ( max( slice ) - min( slice ) )
    slice <- t(slice)
    writePNG( slice, paste0( output_dir, "/latent_distance_closest_layer_", l, "_", i, ".png" ) )

    image_file <- latent_distances$path[total_number_of_images - i + 1]
    cat( "    Furthest image_file = ", image_file, "\n" )
    image <- antsImageRead( image_file )
    slice <- as.array( extractSlice( image, slice_index, 3 ) )  
    slice <- ( slice - min( slice ) ) / ( max( slice ) - min( slice ) )
    slice <- t(slice)
    writePNG( slice, paste0( output_dir, "/latent_distance_furthest_layer_", l, "_", i, ".png" ) )
    }
  }




