

library(dplyr)
library(tidyr)


gx_n_generics <- read.csv("data/gx_num_generics.csv") %>% select(-X) %>% as_tibble()
gx_package <- read.csv("data/gx_package.csv") %>% select(-X) %>% as_tibble()
gx_panel <- read.csv("data/gx_panel.csv") %>% select(-X) %>% as_tibble()
gx_volume <- read.csv("data/gx_volume.csv") %>% select(-X) %>% as_tibble()
gx_ther_area <- read.csv("data/gx_therapeutic_area.csv") %>% select(-X) %>% as_tibble()
submission <- read.csv("data/submission_template.csv") %>% as_tibble()

gx_panel_wide <- gx_panel %>% 
  pivot_wider(id_cols = c("country", "brand"), names_from = "channel", values_from = "channel_rate") %>% 
  mutate_if(is.numeric, function(x) coalesce(x, 0L))

gx_volume$test <- 0
submission$test <- 1
gx_vol_full <- bind_rows(gx_volume, submission %>% select(-pred_95_low, -prediction, -pred_95_high))
gx_vol_full <- gx_vol_full %>% left_join(gx_n_generics, by = c("country", "brand"))
gx_vol_full <- gx_vol_full %>% left_join(gx_ther_area, by = c("brand"))
gx_vol_full <- gx_vol_full %>% left_join(gx_package, by = c("country", "brand"))
gx_vol_full <- gx_vol_full %>% left_join(gx_panel_wide, by = c("country", "brand"))

gx_merged <- gx_vol_full %>% filter(month_num >= 0)
gx_raw <- gx_vol_full

write.csv(gx_merged, file = "data/gx_merged.csv", row.names = F)
write.csv(gx_raw, file = "data/gx_raw.csv", row.names = F)

# gx_vol_full %>% summary()
# 
# gx_vol_full %>% filter(test == 0)
# gx_vol_full %>% filter(test == 1)
# gx_vol_full %>% filter(month_num >= 0)
# 
# gx_vol_full
# 
