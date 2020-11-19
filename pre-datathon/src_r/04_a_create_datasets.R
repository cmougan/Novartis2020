library(tidyverse)
library(lubridate)


excel_table <- readxl::read_xlsx('data/datathon.xlsx', skip = 3)

excel_table_clean <- excel_table[,grepl('[a-z]', names(excel_table))]

long_tbl <- excel_table_clean %>% 
  pivot_longer(contains('20'), names_to = 'cohort', values_to = 'target')

year_test <- 2017


# Add week day data ---------------------------------------------------------------------------

load('data/week_day_month.RData')

week_day_month$month_cat <- as.character(month(week_day_month$month, label = T))

# Prepare tbl -------------------------------------------------------------------------------------

sales_tbl <- long_tbl %>% 
  filter(Function == 'Sales 2') %>% 
  mutate(
    month_cat = str_sub(cohort, 1, 3),
    year = as.numeric(str_sub(cohort, 5, 8))
  ) %>% 
  select(-Function) %>% 
  rename(brand_group = `Brand Group`)


sales_tbl$month <- match(sales_tbl$month_cat, month.abb)
sales_tbl$quarter <- lubridate::quarter(sales_tbl$month)
sales_tbl$quarter_fac <- as.character(sales_tbl$quarter)

# sales_tbl <- left_join(sales_tbl, week_day_month)

target_tbl <- sales_tbl %>% filter(year == year_test)

sales_tbl[sales_tbl$year == year_test, "target"] <- NA

source('tools/feat_eng.R')

library(recipes)
library(parsnip)

sales_tbl
predictions_list <- as.list(1:12)

for(month_predict in 12:1){
  
  # Initial message
  print(paste('Month predict: ', month_predict))
  
  # Add features
  # sales_tbl_month <- add_rolling_stats(sales_tbl, 12 + month_predict, month_predict)
  # sales_tbl_month <- add_rolling_stats(sales_tbl_month, 3 + month_predict, month_predict)
  # sales_tbl_month <- add_rolling_stats(sales_tbl_month, Inf, month_predict)
  # sales_tbl_month <- add_rolling_stats_month(sales_tbl_month, Inf, 1)
  
  sales_tbl_month <- sales_tbl %>% 
    add_rolling_stats(12 + month_predict, month_predict) %>% 
    add_rolling_stats(3 + month_predict, month_predict) %>% 
    add_rolling_stats(Inf, month_predict) %>% 
    add_rolling_stats_month(Inf, 1)
  
  sales_tbl_month <- sales_tbl_month %>% 
    group_by(Cluster, brand_group, Country) %>% 
    mutate(
      cbc_mean = slider::slide_dbl(target, ~ mean(., na.rm = T), .before = Inf, .after = -month_predict)
    )
  
  # Delete rows that are basically NA
  sales_tbl_month <- sales_tbl_month %>% filter(!is.na(cbc_mean))
  
  # Split in 2017 and out of 2017
  train <- sales_tbl_month %>% 
    filter(year < year_test)
  
  test <- sales_tbl_month %>% 
    filter(year == year_test)
  
  # Prepare numeric matrices

    rec <- recipe(train) %>% 
    # step_medianimpute(all_numeric) %>% 
    step_medianimpute(contains("before")) %>%
    step_medianimpute(contains("mean")) %>%
    # step_bagimpute(contains("before"), impute_with = imp_vars(Country, brand_group)) %>%
    # step_bagimpute(contains("mean"), impute_with = imp_vars(Country, brand_group)) %>%
    step_dummy(quarter_fac, month_cat) %>% 
    prep()
  
  train_rec <- juice(rec)
  test_rec <- bake(rec, test)
  
  write.csv(
    train_rec, 
    file = glue::glue("data/feature_engineered/train_{month_predict}.csv"), 
    row.names = FALSE
    )
  
  write.csv(
    test_rec, 
    file = glue::glue("data/feature_engineered/test_{month_predict}.csv"), 
    row.names = FALSE
  )
  
  # train_x <- train_rec %>% 
  #   select(year:month_cat_Sep, -month, -quarter) %>% 
  #   as.matrix()
  # 
  # train_y <- train_rec$target
  # 
  # test_x <- test_rec %>% 
  #   select(year:month_cat_Sep, -month, -quarter) %>% 
  #   as.matrix()
  
}

# a <- 2
# glue::glue("a {a}")
