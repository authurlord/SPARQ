import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataframe
df = pd.read_csv('table.csv')

# Extract and clean the 'Year of election' and '% of popular vote' columns
df['Year of election'] = pd.to_numeric(df['Year of election'])
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Prepare data for regression
X = df['Year of election'].values.reshape(-1, 1)  # Independent variable: year
y = df['% of popular vote'].values  # Dependent variable: popular vote percentage

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next election year (2015)
next_year = 2015
predicted_vote = model.predict([[next_year]])[0]

print(f"Final Answer: {predicted_vote:.2f}")