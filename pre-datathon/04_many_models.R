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
  
  train_x <- train_rec %>% 
    select(year:month_cat_Sep, -month, -quarter) %>% 
    as.matrix()
  
  train_y <- train_rec$target
  
  test_x <- test_rec %>% 
    select(year:month_cat_Sep, -month, -quarter) %>% 
    as.matrix()
  
  # Model
  
  xgb <- boost_tree(mode = 'regression', trees = 50) %>% 
    set_engine('xgboost')
  
  xgb_fit <- fit_xy(xgb, train_x, train_y)
  preds_col <- paste0("preds_", month_predict)
  
  # Evaluate
  
  target_tbl[[preds_col]] <- predict(xgb_fit, test_x)$.pred
  
  train_mae <- mean(abs(predict(xgb_fit, train_x)$.pred - train_y))
  
  print(paste('Train MAE', round(train_mae, 3)))
  print(paste('Full MAE', round(mean(abs(target_tbl[[preds_col]] - target_tbl$target)), 3)))
  
  correct_month <- target_tbl %>% 
    filter(month == month_predict)
  
  print(paste('MAE', round(mean(abs(correct_month[[preds_col]] - correct_month$target)), 3)))
  
}

target_tbl_xgb <- target_tbl
save(target_tbl_xgb, file = 'data/target_tbl_xgb.RData')


target_tbl$target_predict <- NA
for(month_predict in 1:12){
  
  preds_col <- paste0("preds_", month_predict)
  
  target_tbl[(target_tbl$month == month_predict), "target_predict"] <- 
    target_tbl[(target_tbl$month == month_predict), preds_col]
}

# MAE is 120, awesome!!
# Without months, xgb with default config
yardstick::mae(target_tbl, target, target_predict)

# Doesn't seem to improve much with min and max -> actually worse performance

# Importance plot -----------------------------------------------------------------------------

xgb_imp <- xgboost::xgb.importance(model = xgb_fit$fit)
xgboost::xgb.plot.importance(xgb_imp)
