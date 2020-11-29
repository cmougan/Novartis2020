

library(dplyr)
library(ggplot2)

gx_merged <- read.csv("data/gx_merged_lags_months.csv")

gx_merged %>% 
  group_by(test) %>% 
  summarise(
    mean(last_before_12_after_0),
    median(last_before_12_after_0),
    mean(mean_before_12_after_0),
    median(mean_before_12_after_0),
  )


gx_merged %>% 
  group_by(test, country, brand) %>% 
  summarise(
    mean(last_before_12_after_0),
    median(last_before_12_after_0),
    mean(mean_before_12_after_0),
    median(mean_before_12_after_0),
  )# %>% View


gx_merged %>% 
  ggplot(aes(x = (brand_median_before_Inf_after_0), color = as.factor(test), fill = as.factor(test))) +
  geom_density(alpha = 0.3) + 
  geom_rug()


gx_merged %>% 
  ggplot(aes(x = log(brand_median_before_Inf_after_0), color = as.factor(test), fill = as.factor(test))) +
  geom_density(alpha = 0.3) + 
  geom_rug()


gx_merged %>% arrange(desc(brand_median_before_Inf_after_0)) %>% View
gx_merged %>% arrange(mean_before_12_after_0) %>% View
gx_merged %>% filter(is.na(volume)) %>% arrange(mean_before_12_after_0) %>% View
gx_merged %>% filter(brand == "brand_121", country == "country_1") %>% View
