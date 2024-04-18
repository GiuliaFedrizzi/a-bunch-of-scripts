find_dirs <- function(var_type) {
  # find all directories for viscosity
  directories_list <- list()  # will contain the names of the direcctories
  
  # Get all items in the current directory
  items <- list.files(path = ".", full.names = FALSE)
  
  directories_list <- items[sapply(items, function(item) {
    file.info(item)$isdir && !item %in% c("baseFiles", "images_in_grid", "branch_plots")  # ignore these directories
  })]
  
  directories_list <- sort(directories_list)

  print(paste("Directories found:", toString(directories_list)))

    # split the names using underscores, keep the last bit (e.g. from visc_3_1e35 to 1e35)
    if (var_type == 'visc'){
        found_values <- sapply(directories_list, function(dir_name) {
        parts <- unlist(strsplit(dir_name, "_"))
        value <- parts[length(parts)]  # get the last element
        return(value)
        })
    }
    value_only <- unname(found_values)

  return(value_only)
}
