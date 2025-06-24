# --------------------------------------
# Enhanced Airbnb Q3 Prediction with LightGBM
# --------------------------------------

# 0. Packages
library(dplyr)
library(lubridate)
library(tidyr)        # ✅ Needed for replace_na()
library(lightgbm)

# 1. Load data
load(url("https://drive.google.com/uc?export=download&id=1mlJAYmo9TszSJsbYSWhhOY1a3fTJB_Ko"))
# → property_info, reserve_2016Q3_train, listing_2016Q1, listing_2016Q2, PropertyID_test

# 2. Merge Q3 target
df <- property_info %>%
  left_join(reserve_2016Q3_train %>% select(PropertyID, NumReserveDays2016Q3),
            by = "PropertyID")

# 3. Drop cols >50% missing
na_pct <- colMeans(is.na(df))
df     <- df %>% select(-names(na_pct[na_pct > 0.5]))

# 4. Q1/Q2 booking aggregates
all_listings <- bind_rows(listing_2016Q1, listing_2016Q2)
q_feats <- all_listings %>%
  mutate(Booked = ifelse(!is.na(ReservationID), 1L, 0L)) %>%
  group_by(PropertyID) %>%
  summarise(
    Q1_Days    = sum(Booked[Date < "2016-04-01"], na.rm=TRUE),
    Q1_Price   = mean(Price[Date < "2016-04-01"], na.rm=TRUE),
    Q2_Days    = sum(Booked[Date >= "2016-04-01" & Date < "2016-07-01"], na.rm=TRUE),
    Q2_Price   = mean(Price[Date >= "2016-04-01" & Date < "2016-07-01"], na.rm=TRUE),
    Q1_Revenue = sum(Price[Date < "2016-04-01"] * Booked[Date < "2016-04-01"], na.rm=TRUE),
    Q2_Revenue = sum(Price[Date >= "2016-04-01" & Date < "2016-07-01"] * Booked[Date >= "2016-04-01" & Date < "2016-07-01"], na.rm=TRUE),
    .groups = "drop"
  )

df <- df %>%
  left_join(q_feats, by="PropertyID") %>%
  mutate(across(starts_with("Q"), ~replace_na(.x, 0)))  # ✅ No error now

# 5. Feature engineering
df <- df %>%
  mutate(
    logRate     = log1p(PublishedNightlyRate),
    PropAge     = as.numeric(difftime(as.Date("2016-07-01"), CreatedDate, units="days")),
    MonthListed = month(CreatedDate),
    Quarter     = quarter(CreatedDate),
    Weekday     = wday(CreatedDate),
    DaysSinceQ2 = as.numeric(difftime(as.Date("2016-06-30"), CreatedDate, units="days")),
    BedBathR    = Bedrooms / (Bathrooms + 1),
    GuestRoomR  = MaxGuests / (Bedrooms + 1)
  )

# 6. Encode categoricals via integer codes
cat_vars <- c("PropertyType","ListingType","Country","State","City",
              "Neighborhood","CancellationPolicy","InstantbookEnabled")
for(col in cat_vars){
  if(col %in% names(df)){
    df[[col]] <- as.integer(factor(df[[col]], exclude = NULL))
  }
}

# 7. Impute numeric NAs with median
num_vars <- df %>% select(where(is.numeric)) %>%
  select(-NumReserveDays2016Q3, -PropertyID) %>% names()
for(col in num_vars){
  df[[col]] <- replace_na(df[[col]], median(df[[col]], na.rm=TRUE))
}

# 8. Split train/val/test
train_all <- df %>% filter(!is.na(NumReserveDays2016Q3))
test_df   <- df %>% filter(is.na(NumReserveDays2016Q3))

set.seed(42)
idx       <- sample(nrow(train_all), 0.8 * nrow(train_all))
train_df  <- train_all[idx, ]
valid_df  <- train_all[-idx, ]

# 9. Prepare LightGBM datasets
features <- setdiff(names(train_df), c("PropertyID","NumReserveDays2016Q3"))
dtrain   <- lgb.Dataset(as.matrix(train_df[features]), label = train_df$NumReserveDays2016Q3)
dvalid   <- lgb.Dataset.create.valid(dtrain,
                                     data  = as.matrix(valid_df[features]),
                                     label = valid_df$NumReserveDays2016Q3)

# 10. Tune & train
params <- list(
  objective        = "regression",
  metric           = "rmse",
  num_leaves       = 31,
  learning_rate    = 0.03,
  feature_fraction = 0.9,
  bagging_fraction = 0.9,
  bagging_freq     = 5,
  min_data_in_leaf = 20
)

model <- lgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 5000,
  valids                = list(valid = dvalid),
  early_stopping_rounds = 100,
  verbose               = 1
)

# 11. Validate RMSE
val_pred <- predict(model, as.matrix(valid_df[features]))
val_rmse <- sqrt(mean((valid_df$NumReserveDays2016Q3 - val_pred)^2))
cat("Validation RMSE:", round(val_rmse, 3), "\n")

# 12. Final predict & save
pred <- predict(model, as.matrix(test_df[features]))
save(pred, file = "BPK.rdata")
