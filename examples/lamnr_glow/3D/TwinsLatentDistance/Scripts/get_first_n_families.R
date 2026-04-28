

n <- 1203

base_directory <- "/Users/ntustison/Data/Public/OpenNeuro/ds004169"
participants <- read.table( paste0(base_directory, "/OriginalDownload/participants.tsv"), header = TRUE, sep = "\t")

participants <- participants[order( participants$family_id ),]

unique_families <- unique( participants$family_id )
single_member_families <- c() 
for( i in seq.int( length( unique_families ) ) )
  {
  family_members <- which( participants$family_id == unique_families[i] )
  if( length( family_members ) == 1 )
    {
    single_member_families <- c( single_member_families, family_members )
    }
  } 
participants <- participants[-single_member_families,]  

unique_families <- sort( unique( participants$family_id ) )[seq.int(n)]
participants <- participants[participants$family_id %in% unique_families,]

first_n_files <- c() 
pb <- txtProgressBar( min = 0, max = dim( participants )[1], style = 3 )
for( i in seq.int( dim( participants )[1] ) )
  {
  id <- paste0( participants$participant_id[i], "_ses-01_T1w.nii.gz" )
  files <- list.files( path = paste0( base_directory, "/BIDSAlignedToTemplate/" ),
                       pattern = paste0( ".*", id ),
                       recursive = TRUE,
                       full.names = TRUE,
                       include.dirs = TRUE )
  if( length( files ) == 1 )                      
    {
    first_n_files <- c( first_n_files, files )
    }
  else
    { 
    stop( paste0( "No files for participant ", id ) ) 
    }
  setTxtProgressBar( pb, i ) 
  }
cat( "\n" )  

write.table( participants, file = paste0( base_directory, "/manifests/participants_short.tsv"),
             sep="\t", row.names = FALSE, quote = FALSE ) 
write.csv( data.frame( T1 = first_n_files ), file = paste0( base_directory, "/manifests/manifest_t1_short.csv" ),
           row.names = FALSE, quote = FALSE )