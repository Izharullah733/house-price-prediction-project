# **House Price Prediction Project**

**Overview**

This project focuses on predicting house prices using a linear regression model based on the California Housing dataset. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and hypothesis testing to assess the significance of features. The dataset is analyzed to identify patterns, handle outliers, and transform features for improved model performance.

The repository contains two main Python scripts:


main.py: Performs data preprocessing, EDA, linear regression modeling, and visualization of results.


hypothesis_testing.py: Conducts hypothesis testing to evaluate the statistical significance of each feature in the linear regression model.



**Dataset**

The project uses the California Housing dataset (housing.csv), which includes features such as housing median age, total rooms, total bedrooms, population, households, median income, and median house value. The dataset also includes a categorical variable, ocean_proximity, which is encoded numerically for analysis.

**Note: **The dataset is not included in the repository due to size constraints. You can download it from Kaggle or another reliable source and place it in the project directory.

**Prerequisites**

To run this project, you need the following:

Python: Version 3.11.9 (developed using VS Code)

**Libraries:**


pandas

numpy



seaborn

matplotlib

scikit-learn

scipy

**Dataset:** housing.csv (place it in the project directory or update the file path in the scripts)

You can install the required libraries using pip:

pip install pandas numpy seaborn matplotlib scikit-learn scipy

**Project Structure**





main.py: The main script for data preprocessing, exploratory data analysis, linear regression modeling, and visualization.





Loads and preprocesses the dataset (handles missing values, encodes categorical variables, drops irrelevant columns).



Performs EDA with correlation matrices, histograms, boxplots, and scatter plots.



Applies log transformation to handle skewed data.



Trains a linear regression model, evaluates its performance (MSE, R² score), and handles outliers.



Visualizes actual vs. predicted values and regression plots for each feature.



Includes a basic hypothesis testing table for the overall model.



hypothesis_testing.py: Focuses on hypothesis testing for each feature in the linear regression model.





Calculates t-statistics and p-values for each feature.



Determines whether each feature has a statistically significant effect on the target variable (median_house_value).



README.md: This file, providing an overview and instructions for the project.
