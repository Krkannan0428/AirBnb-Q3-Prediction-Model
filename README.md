# 🏠 Airbnb Q3 Booking Prediction – K353 Project

This repository contains our predictive modeling solution for the **K353 Business Analytics Course Project (Spring 2025)** at Indiana University. The objective was to build a machine learning model that predicts the number of booking days (`NumReserveDays2016Q3`) for Airbnb properties in New York City during Q3 of 2016.

---

## 📊 Project Overview

- **Client**: Airbnb  
- **Goal**: Predict Q3 booking days for ~7,400 properties  
- **Approach**: Feature engineering from Q1 & Q2 listing history + gradient boosting (LightGBM)  
- **Output**: A `.rdata` file containing predictions for unseen properties

---

## 🧾 Datasets Used

All datasets were provided by the instructor and loaded via:

```r
load(url("https://drive.google.com/uc?export=download&id=1mlJAYmo9TszSJsbYSWhhOY1a3fTJB_Ko"))

property_info: Static property-level attributes

listing_2016Q1 & listing_2016Q2: Daily listing activity

reserve_2016Q3_train: Actual Q3 booking data for training

PropertyID_test: Properties to forecast

🔨 Methodology
🧼 1. Data Cleaning
Removed columns with >50% missing values

Merged datasets on PropertyID

Imputed missing numeric values using the median

🧠 2. Feature Engineering
Aggregated booking metrics from Q1 & Q2:

Booking counts, average price, and revenue

Created date-based features:

PropAge, MonthListed, Quarter, DaysSinceQ2, etc.

Constructed ratio features:

BedBathR, GuestRoomR

Encoded categorical variables numerically

⚙️ 3. Model Training
Model: LightGBM regression

Train/test split: 80/20 on known booking data

Tuned key parameters: learning rate, num leaves, subsampling

Used early stopping based on validation RMSE

📈 Model Performance
Validation RMSE: ≈ 4.2

Submission vector pred contains forecasts for all PropertyIDs in the test set

💡 Reflection
This project taught me the importance of:

Leveraging time-based booking signals for accurate forecasting

Feature construction from granular daily-level data

Gradient boosting as a powerful model for structured data

Practical R programming and efficient data manipulation


