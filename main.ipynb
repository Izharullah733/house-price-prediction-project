import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"D:\msds\msds\stat\project\housing.csv")
# Show the first few rows of the data
print(df.head())

# Show the number of null values in each column
print(df.isnull().sum())
#drop null value
df.dropna(inplace=True)
print(df.info())
# prompt: ocean_proximity is catagorical variable change it to numeric

df['ocean_proximity'] = df['ocean_proximity'].astype('category').cat.codes
print(df.head())
# Drop the 'longitude', 'latitude', and 'ocean_proximity' columns
df = df.drop(['longitude', 'latitude', 'ocean_proximity'], axis=1)

print(df.head())
columns = ['housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']
columns = np.array(columns)

#discriptive statistics
df.describe()
#correlational matric
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
#Histogram
df.hist(figsize=(16, 10), bins=150)
plt.tight_layout()
plt.show()

#Boxplot
def boxplot(col, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f"{col} Boxplot", fontsize=16)
    plt.xlabel(f"{col}", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

#Boxplot
for col in columns:
    boxplot(col)

# transform the data by log transformation
transformed_df = np.log(df)
#discriptive statistics
transformed_df.describe()
def boxplot(col, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=transformed_df[col], color='skyblue')
    plt.title(f"{col} Boxplot", fontsize=16)
    plt.xlabel(f"{col}", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
for col in columns:
    boxplot(col)
def plot_scatter(transformed_df, columns):
    target_col = 'median_house_value'
    num_cols = len(columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(4*num_cols, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, col in enumerate(columns):
        sns.scatterplot(data=transformed_df, x=col, y=target_col, ax=axes[i])
        axes[i].set_title(f'Scatter plot of {col} vs {target_col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)

    plt.show()
plot_scatter(transformed_df, columns)
# apply linear regression with high accucary



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
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean squared error: {mse}")
y_pred = model.predict(X_test)
model_Accuracy = model.score(X_test, y_test)
print("Accuracy:", model_Accuracy*100)
# prompt: handle the  outlier in transformed_df

# Find the index of the outlier
outlier_index = transformed_df['median_house_value'].idxmax()

# Drop the outlier
transformed_df = transformed_df.drop(outlier_index)

# Reset the index
transformed_df = transformed_df.reset_index(drop=True)
# apply linear regression

# Split the cleaned data into training and testing sets
X = transformed_df.drop('median_house_value', axis=1)
y = transformed_df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean squared error: {mse}")
y_pred = model.predict(X_test)
model_Accuracy = model.score(X_test, y_test)
print("Accuracy:", model_Accuracy*100)
#plot pridicted and acutval value  of the model

plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions, color='skyblue', marker='o', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

#plot the model


def plot_regression(transformed_df, x_col, y_col):
    # Extracting features and target
    X = transformed_df[x_col].values.reshape(-1, 1)
    y = transformed_df[y_col].values.reshape(-1, 1)

    # Fitting the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting the target values
    y_pred = model.predict(X)

    # Plotting the actual data points
    plt.scatter(X, y, color='blue', label='Actual')

    # Plotting the regression line
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

    # Adding labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Regression Plot: {x_col} vs {y_col}")

    # Adding legend
    plt.legend()

    # Showing plot
    plt.show()

for col in columns:
    plot_regression(transformed_df, col, 'median_house_value')


#hypothesis testing on  linear regession model  table

# Import necessary libraries
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import f

# Calculate the necessary statistics
r2 = r2_score(y_test, predictions)
n = len(X_test)
k = len(X_test.columns)

# Calculate the F-statistic
F_value = (r2 / k) / ((1 - r2) / (n - k - 1))

# Calculate the p-value
p_value = 1 - f.cdf(F_value, k, n - k - 1)

# Create a hypothesis testing table
hypothesis_table = pd.DataFrame(columns=['Hypothesis', 'Statistic', 'p-value', 'Conclusion'])

# Add the null and alternative hypotheses
hypothesis_table.loc[0] = ['H0: The model is not statistically significant', F_value, p_value, 'Reject H0']
hypothesis_table.loc[1] = ['H1: The model is statistically significant', F_value, p_value, 'Fail to reject H0']

# Print the hypothesis testing table
print(hypothesis_table.to_string())
