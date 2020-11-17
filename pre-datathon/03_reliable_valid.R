library(tidyverse)
library(lubridate)

source("tools/error_metric.R")


excel_table <- readxl::read_xlsx('data/datathon.xlsx', skip = 3)

excel_table_clean <- excel_table[,grepl('[a-z]', names(excel_table))]

long_tbl <- excel_table_clean %>% 
  pivot_longer(contains('20'), names_to = 'cohort', values_to = 'target')


# sliding -------------------------------------------------------------------------------------

sales_tbl <- long_tbl %>% 
  filter(Function == 'Sales 2') %>% 
  mutate(
    month_cat = str_sub(cohort, 1, 3),
    year = as.numeric(str_sub(cohort, 5, 8))
  ) %>% 
  select(-Function) %>% 
  rename(brand_group = `Brand Group`)


target_2017 <- sales_tbl %>% filter(year == 2017)

sales_tbl[sales_tbl$year == 2017, "target"] <- NA

tictoc::tic()
sales_tbl <- sales_tbl %>% 
  group_by(Cluster, brand_group, Country) %>% 
  mutate(
    cbc_mean = slider::slide_dbl(target, ~ mean(., na.rm = T), .before = Inf, .after = -1)
  )
tictoc::toc()

add_rolling_stats <- function(sales_tbl, n_before, n_after) {
  
  var_name_last <- paste0('last_before_', n_before, '_after_', n_after)
  var_name_mean <- paste0('mean_before_', n_before, '_after_', n_after)
  var_name_median <- paste0('median_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>% 
    group_by(Cluster, brand_group, Country) %>% 
    mutate(
      !! var_name_last := slider::slide_dbl(target, last, .before = n_before, .after = -n_after),
      !! var_name_mean := slider::slide_dbl(target, ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
      !! var_name_median := slider::slide_dbl(target, ~ median(., na.rm = T), .before = n_before, .after = -n_after)
    )
  
}

add_rolling_stats_month <- function(sales_tbl, n_before, n_after) {
  
  var_name_last <- paste0('month_last_before_', n_before, '_after_', n_after)
  var_name_mean <- paste0('month_mean_before_', n_before, '_after_', n_after)
  var_name_median <- paste0('month_median_before_', n_before, '_after_', n_after)
  
  sales_tbl <- sales_tbl %>% 
    group_by(Cluster, brand_group, Country, month_cat) %>% 
    mutate(
      !! var_name_last := slider::slide_dbl(target, last, .before = n_before, .after = -n_after),
      !! var_name_mean := slider::slide_dbl(target,  ~ mean(., na.rm = T), .before = n_before, .after = -n_after),
      !! var_name_median := slider::slide_dbl(target,  ~ median(., na.rm = T), .before = n_before, .after = -n_after)
    )
  
}


sales_tbl$month <- match(sales_tbl$month_cat, month.abb)
sales_tbl$quarter <- lubridate::quarter(sales_tbl$month)

sales_tbl <- add_rolling_stats(sales_tbl, 12, 2)
sales_tbl <- add_rolling_stats_month(sales_tbl, Inf, 1)

# View(sales_tbl)

# Split train and test ------------------------------------------------------------------------

sales_tbl$quarter_fac <- as.character(sales_tbl$quarter)

train <- sales_tbl %>% 
  filter(year < 2017)

test <- sales_tbl %>% 
  filter(year == 2017)

train

library(recipes)
library(embed)

rec <- recipe(train) %>% 
  # step_medianimpute(all_numeric) %>% 
  # step_medianimpute(contains("before")) %>%
  # step_medianimpute(contains("mean")) %>%
  step_bagimpute(contains("before"), impute_with = imp_vars(Country, brand_group)) %>%
  step_bagimpute(contains("mean"), impute_with = imp_vars(Country, brand_group)) %>%
  step_dummy(quarter_fac, month_cat) %>% 
  prep()


train_rec <- juice(rec)
test_rec <- bake(rec, test)


train_x <- train_rec %>% 
  select(cbc_mean:month_cat_Sep, -month, -quarter) %>% 
  as.matrix()

train_y <- train_rec$target

test_x <- test_rec %>% 
  select(cbc_mean:month_cat_Sep, -month, -quarter) %>% 
  as.matrix()

test_y <- test_rec$target

# Model xgb -----------------------------------------------------------------------------------

library(parsnip)

xgb <- boost_tree(mode = 'regression') %>% 
  set_engine('xgboost')

rf <- rand_forest(mode = 'regression') %>%
  set_engine('ranger')

xgb_fit <- fit_xy(xgb, train_x, train_y)
# rf_fit <- fit_xy(rf, train_x, train_y)

mean(abs(predict(xgb_fit, train_x)$.pred - train_y))
mean(abs(predict(xgb_fit, test_x)$.pred - target_2017$target))
# In train it is 86
# Test 221 (without bag impute)
# Test 177 (with bag impute)



# Uncertainty -------------------------------------------------------------

xgb_test_predictions <- predict(xgb_fit, test_x)$.pred
test_y <- target_2017$target

summary(xgb_test_predictions)
summary(test_y)
test_y > xgb_test_predictions + 10
error_metric(test_y, xgb_test_predictions, xgb_test_predictions + 10, xgb_test_predictions - 10)
error_metric(test_y, xgb_test_predictions, xgb_test_predictions + 1, xgb_test_predictions - 1)
error_metric(test_y, xgb_test_predictions, xgb_test_predictions, xgb_test_predictions)
error_metric(test_y, xgb_test_predictions, xgb_test_predictions + 1e3, xgb_test_predictions - 1e3)
error_metric(test_y, xgb_test_predictions, xgb_test_predictions + 1e4, xgb_test_predictions - 1e4)
error_metric(test_y, xgb_test_predictions, xgb_test_predictions + 1e5, xgb_test_predictions - 1e5)

# mean(abs(predict(rf_fit, train_x)$.pred - train_y))
# mean(abs(predict(rf_fit, test_x)$.pred - target_2017$target))
# In train it is 56
# Test 170 (with bag impute)


# Of course this is the nice model!!

# Model glmnet --------------------------------------------------------------------------------

library(glmnet)

glm_fit <- cv.glmnet(train_x, train_y, type.measure = 'mae')

plot(glm_fit)
plot(glm_fit$glmnet.fit)

coef.cv.glmnet(glm_fit, s = 'lambda.1se')

coefplot::coefpath(glm_fit$glmnet.fit)

# Lambda min works better
mean(abs(predict(glm_fit, train_x, s = 'lambda.min') - train_y))
mean(abs(predict(glm_fit, test_x, s = 'lambda.min') - target_2017$target))
# Huge drop
# 102 - 193

mean(abs(predict(glm_fit, train_x, s = 'lambda.1se') - train_y))
mean(abs(predict(glm_fit, test_x, s = 'lambda.1se') - target_2017$target))
# Huge drop here too
     