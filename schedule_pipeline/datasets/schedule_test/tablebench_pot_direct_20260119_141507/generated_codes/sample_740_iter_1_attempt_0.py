import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert Period to numeric time (midpoint of each period)
df['year'] = df['Period'].str.extract(r'(\d{4})-(\d{4})').apply(lambda x: int(x[0]) + 0.5 if x[0] and x[1] else np.nan)
df['year'] = df['year'].fillna(method='ffill')  # fill forward if missing

# Extract the relevant columns
cbri = df['CBR1'].astype(float)
cdr_i = df['CDR1'].astype(float)
imr_i = df['IMR1'].astype(float)

# Create a time array (year)
time = df['year'].values

# Reshape for regression
X = time.reshape(-1, 1)
y_cbr = cbri.values
y_cdr = cdr_i.values
y_imr = imr_i.values

# Fit linear regression models
model_cbr = LinearRegression().fit(X, y_cbr)
model_cdr = LinearRegression().fit(X, y_cdr)
model_imr = LinearRegression().fit(X, y_imr)

# Predict for 2010–2015 (midpoint = 2012.5)
future_time = np.array([[2012.5]])

# Project values
projected_cbr = model_cbr.predict(future_time)[0]
projected_cdr = model_cdr.predict(future_time)[0]
projected_imr = model_imr.predict(future_time)[0]

print(f"Final Answer: {projected_cbr:.2f}, {projected_cdr:.2f}, {projected_imr:.2f}")