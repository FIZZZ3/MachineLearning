
import pandas as pd

# Load the dataset directly from the URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
df.head()



df.info()

missing_values = df.isnull().sum()
print(missing_values)



df.describe()





import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot for total_bill vs. tip
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='total_bill', y='tip', hue='sex', style='time', s=100)
plt.title('Scatter Plot of Total Bill vs. Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend(title='Sex and Time')
plt.grid(True)
plt.show()

# Box plot for sex vs. tip
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='sex', y='tip', hue='time')
plt.title('Box Plot of Tip by Sex and Time')
plt.ylabel('Tip')
plt.xlabel('Sex')
plt.legend(title='Time')
plt.show()

# Calculate average tips by time
average_tips = df.groupby('time')['tip'].mean().reset_index()

# Bar plot for average tips by time
plt.figure(figsize=(8, 5))
sns.barplot(data=average_tips, x='time', y='tip', palette='viridis')
plt.title('Average Tip by Time')
plt.ylabel('Average Tip')
plt.xlabel('Time')
plt.show()

# Pair plot for total_bill, tip and sex
sns.pairplot(df, hue='sex', vars=['total_bill', 'tip'])
plt.title('Pair Plot of Total Bill and Tip by Sex')
plt.show()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)

df_cleaned = df.dropna()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['total_bill', 'tip']])
plt.title('Box Plot for Total Bill and Tip')
plt.show()

# Calculate IQR for 'tip'
Q1 = df['tip'].quantile(0.25)
Q3 = df['tip'].quantile(0.75)
IQR = Q3 - Q1

# Determine outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df_no_outliers = df[(df['tip'] >= lower_bound) & (df['tip'] <= upper_bound)]

# Verify the cleaned dataset
print("Shape of the cleaned dataset:", df_no_outliers.shape)
print("Missing Values in Each Column after cleaning:")
print(df_no_outliers.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot for total_bill vs tip
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='total_bill', y='tip', hue='sex', style='time', alpha=0.7)
plt.title('Scatter Plot of Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend(title='Sex & Time')
plt.show()

# Box plot for tips based on sex and time
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='time', y='tip', hue='sex')
plt.title('Box Plot of Tips by Time and Sex')
plt.xlabel('Time')
plt.ylabel('Tip')
plt.legend(title='Sex')
plt.show()

# Pair plot to visualize relationships
sns.pairplot(df, vars=['total_bill', 'tip'], hue='sex')
plt.suptitle('Pair Plot of Total Bill and Tip', y=1.02)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder

# Assuming df has the categorical columns 'sex' and 'time'
label_encoder = LabelEncoder()

# Convert 'sex' column to numerical values
df['sex'] = label_encoder.fit_transform(df['sex'])

# Convert 'time' column to numerical values
df['time'] = label_encoder.fit_transform(df['time'])

# Display the first few rows to verify the encoding
print(df.head())

# Define independent variables (features) and dependent variable (target)
X = df[['total_bill', 'sex', 'time']]
y = df['tip']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Predict the target variable 'tip' on the testing set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R-squared (R²) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")

# Compare predicted values with actual values by displaying a few examples
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

