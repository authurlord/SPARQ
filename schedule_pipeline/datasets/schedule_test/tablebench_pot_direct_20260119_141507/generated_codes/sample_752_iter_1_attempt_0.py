import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Clean the columns
df['total votes'] = pd.to_numeric(df['total votes'], errors='coerce')
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Prepare data for regression
years = np.array([int(row[0]) for row in df.values])  # election years
total_votes = df['total votes'].values
pop_vote_pct = df['% of popular vote'].values

# Fit linear regression models for both variables
model_votes = LinearRegression()
model_votes.fit(years.reshape(-1, 1), total_votes)

model_pct = LinearRegression()
model_pct.fit(years.reshape(-1, 1), pop_vote_pct)

# Predict for next election (2017)
next_year = 2017
predicted_votes = model_votes.predict([[next_year]])[0]
predicted_pct = model_pct.predict([[next_year]])[0]

print(f"Final Answer: {predicted_votes:.0f}, {predicted_pct:.2f}%")