library(tidyverse)
library(lubridate)


excel_table <- readxl::read_xlsx('data/datathon.xlsx', skip = 3)

excel_table_clean <- excel_table[,grepl('[a-z]', names(excel_table))]

long_tbl <- excel_table_clean %>% 
  pivot_longer(contains('20'), names_to = 'cohort', values_to = 'target')


long_tbl %>% 
  group_by(Cluster, Function) %>% 
  summarise(
    mean(target, na.rm = T),
    n_distinct(`Brand Group`),
    n_distinct(Country),
  )
# Aim is to predict by cluster and band group

sales_tbl <- long_tbl %>% 
  filter(Function == 'Sales 2') %>% 
  mutate(
    month_cat = str_sub(cohort, 1, 3),
    year = as.numeric(str_sub(cohort, 5, 8))
    )

# Try weekday ---------------------------------------------------------------------------------

load('data/week_day_month.RData')

sales_tbl
week_day_month$month_cat <- as.character(month(week_day_month$month, label = T))

sales_tbl <- left_join(sales_tbl, week_day_month)


sales_tbl %>% 
  ggplot(aes(x = as.factor(n_week_day), y = target)) + 
  geom_boxplot()

sales_tbl %>% 
  ggplot(aes(x = target)) + 
  geom_density()


sales_tbl %>% 
  group_by(Cluster, `Brand Group`, Country) %>% 
  summarise(
    mean(target, na.rm = T)
  )




sales_tbl %>% count(year, month, Cluster, `Brand Group`, Country, sort = T)
sales_tbl %>% count(Cluster, `Brand Group`, Country, sort = T)
# 7*12


sales_tbl %>% 
  filter(
    Cluster == 'Cluster 1',
    `Brand Group` == "Brand Group 12",
    Country == 'Country 10'
  ) %>% 
  # group_by(n_real_week_day) %>% summarise(mean(target, na.rm = T))
  ggplot(aes(x = as.factor(n_hard_week_day), y = target)) + 
  geom_boxplot()
# Nope

# Different levels of aggregation -------------------------------------------------------------

sales_tbl


names(sales_tbl)[2] <- "brand_group"

sales_tbl %>% 
  summarise(
    n_distinct(Cluster),
    n_distinct(brand_group),
    n_distinct(Country)
  )

sales_tbl %>% 
  count(
    Cluster,
    brand_group,
    Country
  )
# 1079 different combinations


sales_tbl %>% 
  count(
    brand_group,
    Country
  )
# 1079 brand group country combinations

sales_tbl %>% 
  count(
    Cluster,
    Country
  )
# 50 countries clustered

sales_tbl %>% count(Country)


sales_tbl %>% count(brand_group, Cluster)
sales_tbl %>% count(brand_group)

