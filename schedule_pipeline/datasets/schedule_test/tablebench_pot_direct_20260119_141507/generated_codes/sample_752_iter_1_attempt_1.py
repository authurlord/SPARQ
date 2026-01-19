import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Clean the '% of popular vote' column by removing '%' and converting to float
df['% of popular vote'] = df['% of popular vote'].str.rstrip('%').astype(float)

# Prepare data: years as index (convert election year to numeric)
years = pd.to_numeric(df['election'], errors='coerce')
# Remove NaNs due to parsing issues
years = years.dropna()

# Extract the features and target variables
X = years.values.reshape(-1, 1)  # years as feature
y_votes = df['total votes'].values
y_vote_pct = df['% of popular vote'].values

# Fit linear regression for total votes
model_votes = LinearRegression()
model_votes.fit(X, y_votes)

# Fit linear regression for % of popular vote
model_pct = LinearRegression()
model_pct.fit(X, y_vote_pct)

# Predict for the next election cycle (after 2013, assume 2017)
next_year = 2017
next_votes_pred = model_votes.predict([[next_year]])[0]
next_pct_pred = model_pct.predict([[next_year]])[0]

print(f"Final Answer: {next_votes_pred:.0f}, {next_pct_pred:.2f}%")