import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Filter out rows with "n.a." in % of popular vote
df_clean = df[df['% of popular vote'] != 'n.a.']

# Convert percentage column to float
df_clean['% of popular vote'] = df_clean['% of popular vote'].str.replace('%', '').astype(float)

# Extract years and vote percentages
years = pd.to_numeric(df_clean['Year of election'], errors='coerce')
votes = df_clean['% of popular vote']

# Remove any NaN values
years = years.dropna()
votes = votes.dropna()

# Fit linear regression model
X = years.values.reshape(-1, 1)
y = votes.values
model = LinearRegression().fit(X, y)
next_year = 2012
forecasted_vote = model.predict([[next_year]])[0]

print(f"Final Answer: {forecasted_vote:.2f}%")