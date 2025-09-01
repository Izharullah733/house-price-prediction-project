#model  for each feature with accurate p value and apply hypothesis
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the data
df = pd.read_csv("D:\msds\msds\stat\project\housing.csv")

# Drop null values
df.dropna(inplace=True)

# Encode categorical variable
df['ocean_proximity'] = df['ocean_proximity'].astype('category').cat.codes

# Drop irrelevant columns
df = df.drop(['longitude', 'latitude', 'ocean_proximity'], axis=1)

# Transform the data
transformed_df = np.log(df)

# Split the data into training and testing sets
X = transformed_df.drop('median_house_value', axis=1)
y = transformed_df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate the residuals
residuals = y_test - predictions

# Calculate the standard deviation of the residuals
std_residuals = residuals.std()

# Calculate the t-statistic for each feature
t_statistics = []
for i, col in enumerate(X_test.columns):
    x = X_test[col]
    n = len(x)
    mean_x = x.mean()
    b1 = model.coef_[i]
    t_value = b1 * (n - 1) / (std_residuals * np.sqrt(n * np.sum((x - mean_x)**2) - (np.sum(x - mean_x))**2))
    t_statistics.append(t_value)

# Calculate the p-values
p_values = []
for t_value in t_statistics:
    p_value = 2 * (1 - t.cdf(abs(t_value), len(X_test) - len(X_test.columns) - 1))
    p_values.append(p_value)

# Create a table of results
results_df = pd.DataFrame({
    'Feature': X_test.columns,
    't-statistic': t_statistics,
    'p-value': p_values
})

# Print the table of results
print(results_df.to_string())

# Apply hypothesis testing
alpha = 0.05
for i, row in results_df.iterrows():
    if row['p-value'] < alpha:
        print(f"Reject H0: There is significant evidence that the feature '{row['Feature']}' has a non-zero effect on the target variable.")
    else:
        print(f"Fail to reject H0: There is not enough evidence to conclude that the feature '{row['Feature']}' has a non-zero effect on the target variable.")
