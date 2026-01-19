import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert 'Year of election' to numeric
df['Year of election'] = pd.to_numeric(df['Year of election'], errors='coerce')

# Drop rows with missing % of popular vote
df = df.dropna(subset=['% of popular vote'])

# Extract year and popular vote percentage
X = df[['Year of election']].values.reshape(-1, 1)
y = df['% of popular vote'].astype(float).values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next year (2012)
next_year = 2012
predicted_vote = model.predict([[next_year]])[0]

print(f"Final Answer: {predicted_vote:.2f}")