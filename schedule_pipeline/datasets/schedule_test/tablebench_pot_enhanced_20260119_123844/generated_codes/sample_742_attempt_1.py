import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert 'Year of election' to integer
df['Year of election'] = pd.to_numeric(df['Year of election'])

# Convert '% of popular vote' to float by removing '%' and dividing by 100
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float) / 100

# Prepare features (X) and target (y)
X = df['Year of election'].values.reshape(-1, 1)
y = df['% of popular vote'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next election year (2015)
next_year = 2015
predicted_vote = model.predict([[next_year]])[0]

print(f"Final Answer: {predicted_vote * 100:.2f}")