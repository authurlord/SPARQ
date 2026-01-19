import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Extract years and viewership
years = [int(row['broadcast date']) for row in df.to_dict('records') if row['broadcast date'].isdigit()]
viewership = [float(row['viewers (millions)']) for row in df.to_dict('records') if row['broadcast date'].isdigit()]

# Create a DataFrame for regression
X = np.array(years).reshape(-1, 1)
y = np.array(viewership)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict viewership for 2013
forecast_2013 = model.predict([[2013]])[0]
print(f"Final Answer: {forecast_2013:.2f}")