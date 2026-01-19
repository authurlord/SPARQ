import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert Period to numeric year (midpoint of range)
df['year'] = df['Period'].str.extract(r'(\d{4})-(\d{4})').apply(lambda x: int(x[0]) + 0.5 if x[0] and x[1] else np.nan)
df['year'] = df['year'].fillna(method='ffill')  # fill with previous value if missing

# Select relevant columns
df_clean = df[['year', 'CBR1', 'CDR1', 'IMR1']].dropna()

# Define features and target
X = df_clean[['year']]
y_cbr = df_clean['CBR1']
y_cdr = df_clean['CDR1']
y_imr = df_clean['IMR1']

# Fit linear models
model_cbr = LinearRegression().fit(X, y_cbr)
model_cdr = LinearRegression().fit(X, y_cdr)
model_imr = LinearRegression().fit(X, y_imr)

# Project to 2012.5 (midpoint of 2010–2015)
future_year = np.array([[2012.5]])

# Predict values
cbr_proj = model_cbr.predict(future_year)[0]
cdr_proj = model_cdr.predict(future_year)[0]
imr_proj = model_imr.predict(future_year)[0]

print(f"Final Answer: {cbr_proj:.2f}, {cdr_proj:.2f}, {imr_proj:.2f}")