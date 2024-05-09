find_dirs <- function(var_type) {
  # find all directories for 
  #  - viscosity: var_type = visc
  #  - melt rate: var_type = melt_rate 
  directories_list <- list()  # will contain the names of the direcctories
  
  # Get all items in the current directory
  items <- list.files(path = ".", full.names = FALSE)
  # print(paste("items:", toString(items)))
  
  directories_list <- items[sapply(items, function(item) {
    file.info(item)$isdir && !item %in% c("baseFiles", "images_in_grid", "branch_plots")  # ignore these directories
  })]
  
  directories_list <- sort(directories_list)

  # print(paste("Directories found:", toString(directories_list)))

    # split the names using underscores, keep the last bit (e.g. from visc_3_1e35 to 1e35)
    if (var_type == 'visc'){
        found_values <- sapply(directories_list, function(dir_name) {
        parts <- unlist(strsplit(dir_name, "_"))
        value <- parts[length(parts)]  # get the last element
        return(value)
        })
    }
    else if (var_type == 'melt_rate'){
      # get the list of melt rates, e.g. 01 from vis1e1_mR_01
        items <- list.files(path = directories_list[1], full.names = FALSE) # all files and directories inside the first parent directory
        full_paths <- list.files(path = directories_list[1], full.names = TRUE) # full paths to check if they are directories
        is_directory <- sapply(full_paths, dir.exists)

        directory_names <- items[is_directory] # keep directories only
        mR_directories <- directory_names[grepl("mR", directory_names)]  # keep only those containing 'mR'
        last_two_digits <- sub(".*_(\\d\\d)$", "\\1", mR_directories) # Extract the last two digits 

        # print(paste("last_two_digits", toString(last_two_digits)))
        return(last_two_digits)
    }
    value_only <- unname(found_values)

  return(value_only)
}
