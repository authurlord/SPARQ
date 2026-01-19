import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract year and viewers
df['broadcast date'] = pd.to_numeric(df['broadcast date'])
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'])

# Prepare data for linear regression
X = df[['broadcast date']].values  # Year as input
y = df['viewers (millions)'].values  # Viewers as output

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict viewership for 2013
predicted_2013 = model.predict([[2013]])[0]

print(f"Final Answer: {predicted_2013:.2f}")