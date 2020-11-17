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


library(fable)
library(tsibble)


sales_tbl$ym <- yearmonth((paste(sales_tbl$year, sales_tbl$month, '01', sep = '-')))

sales_tbl_mini <- sales_tbl %>% 
  filter(year < 2015) %>% 
  arrange(Cluster, brand_group, Country, ym) %>% 
  head(84*10) 


tsbl_sales <- as_tsibble(sales_tbl_mini, key = c(Cluster, brand_group, Country), index = ym)

tsbl_sales

# # No gaps
# has_gaps(tsbl_sales, .full = TRUE) %>% 
#   count(.gaps)

# TODO
# https://tsibble.tidyverts.org/articles/window.html

tsbl_sales %>% autoplot(target)

# Some ts exploration -------------------------------------------------------------------------

sales_tbl %>% 
  arrange(Cluster, brand_group, Country, ym) %>% 
  head(84*3) %>% 
  ggplot() + 
  geom_line(aes(x = ym, y = target, group = Country, color = Country))
  
sales_tbl %>% 
  arrange(Cluster, brand_group, Country, ym) %>% 
  tail(84*3) %>% 
  ggplot() + 
  geom_line(aes(x = ym, y = target, group = Country, color = Country))

