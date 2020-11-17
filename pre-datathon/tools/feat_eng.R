

add_rolling_stats <- function(sales_tbl, n_before, n_after) {
  
  var_name_last <- paste0('last_before_', n_before, '_after_', n_after)
  var_name_mean <- paste0('mean_before_', n_before, '_after_', n_after)
  # var_name_min <- paste0('month_min_before_', n_before, '_after_', n_after)
  # var_name_max <- paste0('month_max_before_', n_before, '_after_', n_after)
  var_name_median <- paste0('median_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>% 
    group_by(Cluster, brand_group, Country) %>% 
    mutate(
      # !! var_name_max := slider::slide_dbl(target, max, .before = n_before, .after = -n_after),
      # !! var_name_min := slider::slide_dbl(target, min, .before = n_before, .after = -n_after),
      !! var_name_last := slider::slide_dbl(target, last, .before = n_before, .after = -n_after),
      !! var_name_mean := slider::slide_dbl(target, ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
      !! var_name_median := slider::slide_dbl(target, ~ median(., na.rm = T), .before = n_before, .after = -n_after)
    )
  
}

add_rolling_stats_month <- function(sales_tbl, n_before, n_after) {
  
  var_name_last <- paste0('month_last_before_', n_before, '_after_', n_after)
  var_name_mean <- paste0('month_mean_before_', n_before, '_after_', n_after)
  # var_name_min <- paste0('month_min_before_', n_before, '_after_', n_after)
  # var_name_max <- paste0('month_max_before_', n_before, '_after_', n_after)
  var_name_median <- paste0('month_median_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>% 
    group_by(Cluster, brand_group, Country, month_cat) %>% 
    mutate(
      # !! var_name_max := slider::slide_dbl(target, max, .before = n_before, .after = -n_after),
      # !! var_name_min := slider::slide_dbl(target, min, .before = n_before, .after = -n_after),
      !! var_name_last := slider::slide_dbl(target, last, .before = n_before, .after = -n_after),
      !! var_name_mean := slider::slide_dbl(target,  ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
      !! var_name_median := slider::slide_dbl(target,  ~ median(., na.rm = T), .before = n_before, .after = -n_after)
    )
  
}
