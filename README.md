# Capstone-Project
Online Retail Sales Analysis and Daily Revenue Prediction

Project Title

Online Retail Sales Analysis and Daily Revenue Prediction Using Machine Learning

Name: Gautham Rajesh

Project Type: Regression / Time-Series Forecasting

Objective

The objective of this project is to analyse historical online retail transactions and develop a machine learning model to predict daily sales revenue. The project involves data cleaning, exploratory data analysis, feature engineering, and the comparison of multiple regression algorithms to identify the model with the best predictive performance.
Dataset Used
Dataset Name: Online Retail II
Dataset File: Dataset.csv
Dataset Period: 2009–2010
The dataset initially contains 525,461 transaction records and 8 columns. It includes information about invoices, products, quantities sold, transaction dates, prices, customer identifiers, and countries.
The target variable is Revenue, calculated by multiplying the transaction quantity by the unit price. Transaction-level records are aggregated into daily revenue values for forecasting.
Main Features
- Invoice – Invoice or transaction identifier
- StockCode – Product identification code
- Description – Product description
- Quantity – Number of units recorded in a transaction
- InvoiceDate – Date and time of the transaction
- Price – Unit price of the product
- Customer ID – Customer identifier
- Country – Country associated with the transaction
- Revenue – Revenue calculated as Quantity × Price
For modelling, the project uses calendar features and historical revenue features derived from the daily sales data.
Data Preprocessing
The dataset was examined for missing values, duplicate records, invalid values, and cancelled transactions before modelling.
The preprocessing steps included:
1. Removing records missing essential transaction information.
2. Removing duplicate records.
3. Excluding cancelled invoices identified by invoice numbers beginning with C.
4. Removing records with non-positive quantities or prices.
5. Converting transaction dates and numerical columns to appropriate data types.
6. Calculating revenue for each transaction line.
7. Aggregating transaction records into daily sales totals.
After cleaning, the dataset contained 504,730 rows and 11 columns. Missing customer identifiers were retained because customer-level identification was not required for the daily revenue prediction task.
The modelling dataset was then created by aggregating sales by date and generating historical features. After feature engineering, 360 daily observations remained for modelling.
Note: The model focuses on positive sales revenue. Since cancelled invoices and non-positive quantities were excluded, the target does not represent net revenue after returns or refunds.
Exploratory Data Analysis
The project used several visualisations to understand sales patterns and revenue distribution, including:
- Distribution of transaction quantities
- Distribution of product prices
- Top 10 products by revenue
- Top 10 countries by revenue
- Monthly revenue trend
- Correlation between quantity, price, and revenue
- Daily revenue over time
The analysis showed that transaction quantities and product prices were right-skewed, with most transaction lines involving relatively small quantities and lower unit prices.
The United Kingdom contributed substantially more revenue than other countries in the dataset. The product revenue analysis identified Manual, REGENCY CAKESTAND 3 TIER, and WHITE HANGING HEART T-LIGHT HOLDER among the highest-revenue entries.
Monthly revenue varied across the observation period, with a substantial increase during autumn 2010. The daily revenue plot also showed recurring patterns and occasional large revenue spikes.
The correlation analysis indicated a positive relationship between unit price and revenue, while the relationship between quantity and revenue was weaker. These correlations describe transaction-level relationships and do not establish causation.
Feature Engineering
Feature engineering was performed to create variables that could help the model identify calendar patterns and historical sales behaviour.
The following features were used:
- DayOfWeek – Day of the week, from Monday (0) to Sunday (6)
- DayOfMonth – Day of the month
- Month – Calendar month
- IsWeekend – Indicates whether the day falls on a weekend
- Lag_1 – Revenue from the previous day
- Lag_7 – Revenue from seven days earlier
- Lag_14 – Revenue from fourteen days earlier
- RollingMean_7 – Average revenue over the previous seven days
- RollingMean_14 – Average revenue over the previous fourteen days
Rolling averages were calculated using historical revenue only, helping prevent the current day's target from leaking into its own features.
Models Used
1. Linear Regression
Linear Regression was used as a baseline regression model. It estimates relationships between the input features and daily revenue using a linear function.
2. Decision Tree Regressor
Decision Tree Regression was selected because it can capture nonlinear relationships between calendar features, historical sales, and revenue. It divides observations into groups based on feature values to generate predictions.
3. Random Forest Regressor
Random Forest was used as an ensemble learning algorithm that combines multiple decision trees. It can model nonlinear patterns and interactions between features and often provides more robust predictions than a single decision tree.
Model Evaluation
The dataset was split chronologically, with the first 80% of observations used for training and the remaining 20% used for testing. This preserves the time order of the data and better reflects the forecasting task.
The models were evaluated using the following regression metrics:
- Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted revenue.
- Root Mean Squared Error (RMSE): Measures prediction error while penalising larger errors more heavily.
- R² Score: Measures how much variation in the target is explained by the model relative to a mean-prediction baseline.
Model Performance
The following results were recorded in the project notebook.
Model	MAE	RMSE	R² Score
Random Forest	13,216.64	18,790.06	0.4238
Decision Tree	16,396.53	22,971.02	0.1389
Linear Regression	17,343.97	23,300.29	0.1140
Note: Revenue and error values are expressed in the monetary units represented by the dataset.
Key Results
The Random Forest Regressor performed best overall among the three tested models. It achieved the lowest MAE of approximately 13,216.64, the lowest RMSE of approximately 18,790.06, and the highest R² score of approximately 0.4238.
The Decision Tree and Linear Regression models produced higher prediction errors and lower R² scores.
The actual-versus-predicted plot showed that Random Forest captured some recurring weekly sales patterns but did not fully capture every large revenue spike. This indicates that historical sales and calendar features provide useful predictive information, while unusual high-revenue days remain difficult to estimate accurately.
The feature-importance analysis identified DayOfWeek and IsWeekend as the most influential features in the recorded Random Forest model, followed by Lag_1 and RollingMean_7.
Therefore, Random Forest was selected as the final model because it provided the strongest overall performance on the chronological test set.
Final Model
The selected Random Forest model was saved using joblib so that it can be loaded and reused without retraining.
The saved model bundle contains the trained estimator, feature-column names, model metadata, and information about the training and test periods.
The model predicts daily revenue as a numerical value.
- Input: Calendar features and historical revenue values
- Output: Estimated daily revenue for the target date
The saved model was subsequently loaded and tested on a held-out observation. For the displayed example, the actual revenue was 38,634.22, while the predicted revenue was 56,709.38. This confirms that the saved model can generate predictions, although individual predictions may contain substantial errors.
Model Deployment
The trained model was prepared for local deployment using Python and joblib.
A prediction script, predict_sales.py, loads the saved model, prepares a sample input DataFrame containing the required features, and generates a daily revenue prediction.
The prediction script uses the following saved model file:
artifacts/online_retail_sales_model.pkl


The example input values are illustrative and should be replaced with realistic historical sales values when generating predictions.
A Flask-based interface can be used as a further deployment step to allow users to provide inputs through a web form. Public deployment and automated retrieval of historical sales features would require additional implementation and testing.
Limitations
- The dataset covers 2009–2010 and may not represent current retail behaviour.
- The daily modelling dataset contains only 360 observations after feature engineering.
- The model does not fully capture every unusually large revenue spike.
- Cancelled invoices and non-positive quantities were excluded, so the target represents positive sales rather than net revenue after refunds.
- December 2010 may be incomplete, which affects interpretation of the final monthly revenue value.
- The current model relies on accurate lag and rolling-average inputs.
- Forecasting multiple future days requires updating historical features for each forecast step.
- Only three regression models were compared in the recorded experiment.
- A simple forecasting baseline and additional chronological validation periods would provide a stronger assessment of predictive performance.
- The notebook showed a scikit-learn version compatibility warning when loading the saved model, which should be resolved for reproducible deployment.
Future Work
- Compare the model against baseline forecasts, such as previous-day revenue and revenue from the same weekday in the previous week.
- Perform additional chronological validation to assess consistency across different time periods.
- Explore additional time-series forecasting algorithms.
- Investigate feature engineering for holidays, seasonal effects, and other retail patterns where suitable data is available.
- Add prediction intervals or uncertainty estimates.
- Build a Flask application that automatically calculates historical features from stored sales data.
- Improve deployment reliability by pinning dependency versions and testing the saved model in a clean environment.
- If net revenue is the business objective, explicitly incorporate returns and refunds into the target definition.
Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Flask (deployment extension)
- Jupyter Notebook
- Visual Studio Code
Conclusion
This project demonstrates the application of machine learning to online retail sales analysis and daily revenue prediction. The workflow involved cleaning transaction data, performing exploratory data analysis, aggregating daily sales, engineering historical and calendar features, and comparing multiple regression algorithms.
Among the models tested, Random Forest achieved the strongest overall performance, with an MAE of 13,216.64, an RMSE of 18,790.06, and an R² score of 0.4238 on the chronological test set.
The trained model was saved and successfully reloaded for local prediction. The results suggest that calendar patterns and historical revenue contain useful information for forecasting daily sales. However, the limited dataset length and difficulty predicting large revenue spikes indicate that further validation and model improvement are needed.
Overall, the project provides an end-to-end baseline for retail revenue forecasting and a foundation for future improvements in time-series modelling and deployment.
