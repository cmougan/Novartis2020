
min_which <- function(x) {
  #' @param x: a vector of type logical
  #'
  #' * NA if there's a NA
  #' * The minimum element that satisfies x = True of x
  #' * The length of x + 1 if there's not such element
  #' 
  #' @examples
  #' x <- c(0, 0, 2)
  #' x1 <- c(0, 0, 0)
  #' min(which(x != 0))
  #' min(which(x1 != 0))
  #' min_which(x != 0)
  #' min_which(x1 != 0)
  #'
  
  if(any(is.na(x))) return(NA)
  if (any(x)) return(min(which(x)))
  length(x) + 1
  
}


add_rolling_num_days_since <- function(sales_tbl, n_before, n_after) {
  
  var_name_0 <- paste0('before_', n_before, '_after_', n_after, '_0')
  var_name_40 <- paste0('before_', n_before, '_after_', n_after, '_40')
  var_name_200 <- paste0('before_', n_before, '_after_', n_after, '_200')
  # var_name_min <- paste0('month_min_before_', n_before, '_after_', n_after)
  # var_name_max <- paste0('month_max_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>% 
    group_by(brand, country) %>% 
    mutate(
      # !! var_name_max := slider::slide_dbl(volume, max, .before = n_before, .after = -n_after),
      # !! var_name_min := slider::slide_dbl(volume, min, .before = n_before, .after = -n_after),
      !! var_name_0 := slider::slide_dbl(volume, ~ min_which(. > 0), .before = n_before, .after = -n_after),
      !! var_name_40 := slider::slide_dbl(volume, ~ min_which(. > 40), .before = n_before, .after = -n_after),
      !! var_name_200 := slider::slide_dbl(volume, ~ min_which(. > 200), .before = n_before, .after = -n_after)
    )
  
}


add_rolling_stats_brand <- function(sales_tbl, n_before, n_after) {
  
  var_name_last <- paste0('brand_last_before_', n_before, '_after_', n_after)
  var_name_mean <- paste0('brand_mean_before_', n_before, '_after_', n_after)
  var_name_median <- paste0('brand_median_before_', n_before, '_after_', n_after)
  var_name_sd <- paste0('brand_month_sd_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>% 
    group_by(brand) %>% 
    mutate(
      # !! var_name_max := slider::slide_dbl(volume, max, .before = n_before, .after = -n_after),
      # !! var_name_min := slider::slide_dbl(volume, min, .before = n_before, .after = -n_after),
      !! var_name_last := slider::slide_dbl(volume, last, .before = n_before, .after = -n_after),
      !! var_name_mean := slider::slide_dbl(volume, ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
      !! var_name_median := slider::slide_dbl(volume, ~ median(., na.rm = T), .before = n_before, .after = -n_after),
      # !! var_name_sd := slider::slide_dbl(volume,  ~ sd(., na.rm = T), .before = n_before, .after = -n_after)
    )
  
}

add_rolling_stats_country <- function(sales_tbl, n_before, n_after) {
  
  var_name_last <- paste0('country_last_before_', n_before, '_after_', n_after)
  var_name_mean <- paste0('country_mean_before_', n_before, '_after_', n_after)
  var_name_median <- paste0('country_median_before_', n_before, '_after_', n_after)
  var_name_sd <- paste0('country_month_sd_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>%
    group_by(country) %>% 
    mutate(
      # !! var_name_max := slider::slide_dbl(volume, max, .before = n_before, .after = -n_after),
      # !! var_name_min := slider::slide_dbl(volume, min, .before = n_before, .after = -n_after),
      !! var_name_last := slider::slide_dbl(volume, last, .before = n_before, .after = -n_after),
      !! var_name_mean := slider::slide_dbl(volume, ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
      !! var_name_median := slider::slide_dbl(volume, ~ median(., na.rm = T), .before = n_before, .after = -n_after),
      # !! var_name_sd := slider::slide_dbl(volume,  ~ sd(., na.rm = T), .before = n_before, .after = -n_after)
    )
  
}

add_rolling_stats <- function(sales_tbl, n_before, n_after) {
  
  var_name_last <- paste0('last_before_', n_before, '_after_', n_after)
  var_name_mean <- paste0('mean_before_', n_before, '_after_', n_after)
  # var_name_min <- paste0('month_min_before_', n_before, '_after_', n_after)
  # var_name_max <- paste0('month_max_before_', n_before, '_after_', n_after)
  var_name_median <- paste0('median_before_', n_before, '_after_', n_after)
  var_name_sd <- paste0('month_sd_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>% 
    group_by(brand, country) %>% 
    mutate(
      # !! var_name_max := slider::slide_dbl(volume, max, .before = n_before, .after = -n_after),
      # !! var_name_min := slider::slide_dbl(volume, min, .before = n_before, .after = -n_after),
      !! var_name_last := slider::slide_dbl(volume, last, .before = n_before, .after = -n_after),
      !! var_name_mean := slider::slide_dbl(volume, ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
      !! var_name_median := slider::slide_dbl(volume, ~ median(., na.rm = T), .before = n_before, .after = -n_after),
      # !! var_name_sd := slider::slide_dbl(volume,  ~ sd(., na.rm = T), .before = n_before, .after = -n_after)
    )
  
}

# add_rolling_stats_month <- function(sales_tbl, n_before, n_after) {
#   
#   var_name_last <- paste0('month_last_before_', n_before, '_after_', n_after)
#   var_name_mean <- paste0('month_mean_before_', n_before, '_after_', n_after)
#   # var_name_min <- paste0('month_min_before_', n_before, '_after_', n_after)
#   # var_name_max <- paste0('month_max_before_', n_before, '_after_', n_after)
#   var_name_median <- paste0('month_median_before_', n_before, '_after_', n_after)
#   var_name_var <- paste0('month_var_before_', n_before, '_after_', n_after)
#   
#   sales_tbl <- sales_tbl %>% 
#     group_by(brand, country, month_cat) %>% 
#     mutate(
#       # !! var_name_max := slider::slide_dbl(volume, max, .before = n_before, .after = -n_after),
#       # !! var_name_min := slider::slide_dbl(volume, min, .before = n_before, .after = -n_after),
#       !! var_name_last := slider::slide_dbl(volume, last, .before = n_before, .after = -n_after),
#       !! var_name_mean := slider::slide_dbl(volume,  ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
#       !! var_name_median := slider::slide_dbl(volume,  ~ median(., na.rm = T), .before = n_before, .after = -n_after),
#       !! var_name_var := slider::slide_dbl(volume,  ~ var(., na.rm = T), .before = n_before, .after = -n_after)
#     )
#   
# }
