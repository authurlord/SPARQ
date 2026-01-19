import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Use the index as the independent variable (year)
X = np.arange(len(df)).reshape(-1, 1)  # Year index
y = df['total support and revenue'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the next year (index = len(df))
next_year_index = len(df)
predicted_value = model.predict([[next_year_index]])[0]

print(f"Final Answer: {int(predicted_value)}")