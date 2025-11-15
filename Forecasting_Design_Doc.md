# **Design Document: SKU-Level Demand Forecasting Model**

Version: 2.0  
Date: 2025-11-15

## **1\. Problem Statement**

This project addresses the core business need for accurate, SKU-level demand forecasts to optimize manufacturing and inventory. The system automates the entire forecasting pipeline: it ingests raw data, cleans and enriches it, tests a competing set of advanced statistical models for *every single item*, and automatically selects the best-performing model to generate a 6-month forecast.

The final output is not just a forecast, but an **actionable production plan** that calculates the "Net Need" for each SKU by comparing the forecast against current inventory-on-hand.

## **2\. Data Sources**

The system integrates three key data sources from the mat\_case\_study.xlsx file:

| Data Source | Sheet Name | Role in Model |
| :---- | :---- | :---- |
| **Actual Orders** | Actual Orders | **Target Variable.** This is the historical data we are trying to predict (the "what"). |
| **POS Data** | POS | **Exogenous Predictor.** Point-of-Sale data is used as a *leading indicator* of demand (the "why"). |
| **Inventory Data** | Inventory | **Business Input.** Used *after* forecasting to calculate the final production plan. |

## 

## **3\. Model Design Methodology**

The system is designed as a sequential pipeline. This methodology covers the data preparation and feature engineering required *before* model training.

### **Data Preprocessing & Feature Engineering**

This block cleans and prepares the raw data for modeling.

* **Data Ingestion:** Loads the Actual Orders, POS, and Inventory sheets from the master Excel file.  
* **Data Transformation:** "Melts" the data from its wide (spreadsheet) format into a long (time-series) format.  
* **Missing Value Treatment:** Uses time-based interpolate() to fill any gaps in monthly data. This is superior to fillna(0) as it assumes a logical trend between known data points.  
* **Feature Engineering 1 (POS\_lag\_6):** Creates a POS\_lag\_6 feature. This means the model is trained to see if POS data from 6 months ago (e.g., March) has a predictive relationship with orders *today* (e.g., September). This captures long-term market signals.  
* **Feature Engineering 2 (Policy\_Active):** Creates a binary (0 or 1\) "switch" variable.  
  * **Value:** 0 for all dates before 2025-02-01.  
  * **Value:** 1 for all dates on or after 2025-02-01.  
  * **Purpose:** This explicitly allows the models to test the impact of the new US import policy and determine if it caused a structural break in demand patterns.


## **4\. Model Building**

This section details the competitive validation, selection, and forecasting process.

### **Model Validation "Bake-Off"**

This is the core intelligence of the system. Instead of using one model for all items, we run a **5-way competition** for *each SKU* to find its unique best forecaster.

The script holds back the last 6 months of data as a "validation" set. It then trains all 5 models on the preceding data and asks them to predict this 6-month validation period.

**The 5 Competing Models:**

* **Group 1: Univariate Models (Simple)**  
  * These models *only* look at the "Actual Orders" history.  
  1. SARIMA (Uni)  
  2. ETS (Uni)  
  3. Prophet (Uni)  
* **Group 2: Exogenous Models (Complex)**  
  * These models use "Actual Orders" \+ POS\_lag\_6 \+ Policy\_Active.  
  4. SARIMAX (Exog)  
  5. Prophet (Exog)

### **Algorithms Employed**

* **SARIMA(X) (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors):** A highly respected statistical model that accounts for trends, seasonality (e.g., year-over-year patterns), and external predictors.  
* **ETS (Error, Trend, Seasonality):** A class of exponential smoothing models that are very fast and robust, working well on data with clear seasonal patterns.  
* **Prophet:** A forecasting tool by Meta designed to be robust to missing data and shifts in trends, and to handle multiple seasonalities (e.e., yearly, weekly) well.

### **Automated Model Selection**

The winner of the "bake-off" is chosen based on one simple metric: **Root Mean Squared Error (RMSE)**.

For each SKU, the model with the lowest RMSE on the 6-month validation set is crowned the "Best Overall Model". This data-driven selection ensures we are always using the most accurate and appropriate algorithm for each item's unique demand profile.

### **Final Forecasting**

Once the "Best Overall Model" for an SKU is identified:

1. The script re-trains that one winning model on the *entire* historical dataset (including the 6-month validation period).  
2. It then uses this final, robust model to forecast the next 6 months.

## **5\. Model Outputs**

This automated system generates four primary outputs for stakeholders:

### **Production Plan Generation**

The final 6-month forecast is merged with the current Inventory\_On\_Hand data. The script then calculates a month-by-month plan, determining the **Net Need**:

Net\_Need \= Forecasted\_Demand \- Starting\_Inventory (clipped at 0\)

The Ending\_Inventory from one month becomes the Starting\_Inventory for the next, allowing for a rolling plan.

### **Key Outputs & Deliverables**

1. **validation\_overall\_model\_comparison.csv (The Scorecard):**  
   * The single most important file for analysis. It lists every SKU, the RMSE for all 5 competing models, and which model was selected as the winner.  
2. **validation\_plots/ (The Visual Proof):**  
   * A folder containing a PNG plot for every SKU, showing the 5-way bake-off against the actual historical data. This allows for quick visual validation.  
3. **demand\_forecast\_best\_model\_next\_6\_months.csv (The Forecast):**  
   * The final, clean 6-month forecast (in units) for every SKU, ready for financial planning.  
4. **production\_plan\_best\_model\_next\_6\_months.csv (The Action Plan):**  
   * The final, actionable plan showing the Net\_Need (how much to produce) and Ending\_Inv for each of the next 6 months.