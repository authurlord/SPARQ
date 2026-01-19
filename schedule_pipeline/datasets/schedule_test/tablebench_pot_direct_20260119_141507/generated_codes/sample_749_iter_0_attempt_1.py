import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract years and viewership
years = []
viewership = []

for index, row in df.iterrows():
    year = int(row['broadcast date'])
    if 2005 <= year <= 2011:
        years.append(year)
        viewership.append(float(row['viewers (millions)']))

# Convert to numpy arrays
X = np.array(years).reshape(-1, 1)
y = np.array(viewership)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict viewership for 2013
predicted_2013 = model.predict([[2013]])[0]

print(f"Final Answer: {predicted_2013:.2f}")