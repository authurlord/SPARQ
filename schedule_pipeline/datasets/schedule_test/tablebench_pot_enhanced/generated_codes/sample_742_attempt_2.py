import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Clean the '% of popular vote' column
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Prepare the data for linear regression
X = df['Year of election'].values.reshape(-1, 1)
y = df['% of popular vote'].values

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next election year (2015)
next_year = np.array([[2015]])
predicted_vote = model.predict(next_year)[0]

print(f"Final Answer: {predicted_vote:.2f}")