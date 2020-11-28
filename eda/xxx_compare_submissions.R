


library(dplyr)
library(ggplot2)


country <- read.csv("submissions/submission_country.csv")
# baseline <- read.csv("submissions/baseline_carlos_david.csv")
# baseline <- read.csv("submissions/baseline_lgbm_offset.csv")
# baseline <- read.csv("submissions/lgbm_with_vol_feats.csv")
baseline <- read.csv("~/Downloads/NN.csv")

baseline %>% count(pred_95_low > pred_95_high)
baseline %>% count(pred_95_low > prediction)
baseline %>% count(pred_95_high < prediction, pred_95_low > prediction)

merged <- baseline %>% left_join(
  country, 
  by = c("country", "brand", "month_num"),
  suffix = c("_bs", "_ctr")
  )

merged %>% 
  ggplot(aes(y = prediction_bs, x = prediction_ctr)) + 
  geom_point()
  

merged %>% 
  arrange(desc(prediction_bs - prediction_ctr)) %>% 
  head(10)


gx_merged_raw <- read.csv("data/gx_raw.csv") %>% as_tibble()

gx_merged_raw %>% filter(country == "country_16", brand == "brand_241") %>% View
