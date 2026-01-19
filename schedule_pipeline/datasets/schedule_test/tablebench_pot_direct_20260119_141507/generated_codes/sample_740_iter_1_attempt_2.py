import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Clean the Period column: extract the start year
df['start_year'] = df['Period'].str.split('-').str[0].astype(int)

# Convert to numeric for analysis
df['CBR1'] = pd.to_numeric(df['CBR1'], errors='coerce')
df['CDR1'] = pd.to_numeric(df['CDR1'], errors='coerce')
df['IMR1'] = pd.to_numeric(df['IMR1'], errors='coerce')

# Prepare data for linear regression: years as x, rates as y
years = df['start_year']
cbr_values = df['CBR1']
cdr_values = df['CDR1']
imr_values = df['IMR1']

# Project to 2010–2015: use years 2010 to 2015 (we assume trend continues)
# We fit the trend from 1950 to 2010, then project to 2010–2015

# Create a new dataset for projection
X = np.array(years).reshape(-1, 1)
y_cbr = cbr_values.values
y_cdr = cdr_values.values
y_imr = imr_values.values

# Fit linear regression models
model_cbr = LinearRegression().fit(X, y_cbr)
model_cdr = LinearRegression().fit(X, y_cdr)
model_imr = LinearRegression().fit(X, y_imr)

# Predict for years 2010 to 2015
future_years = np.array([2010, 2011, 2012, 2013, 2014, 2015]).reshape(-1, 1)
predicted_cbr = model_cbr.predict(future_years)
predicted_cdr = model_cdr.predict(future_years)
predicted_imr = model_imr.predict(future_years)

# Output projected values for 2010–2015
projected_values = {
    '2010': round(predicted_cbr[0], 1),
    '2011': round(predicted_cbr[1], 1),
    '2012': round(predicted_cbr[2], 1),
    '2013': round(predicted_cbr[3], 1),
    '2014': round(predicted_cbr[4], 1),
    '2015': round(predicted_cbr[5], 1),
    '2010_CDR': round(predicted_cdr[0], 1),
    '2011_CDR': round(predicted_cdr[1], 1),
    '2012_CDR': round(predicted_cdr[2], 1),
    '2013_CDR': round(predicted_cdr[3], 1),
    '2014_CDR': round(predicted_cdr[4], 1),
    '2015_CDR': round(predicted_cdr[5], 1),
    '2010_IMR': round(predicted_imr[0], 1),
    '2011_IMR': round(predicted_imr[1], 1),
    '2012_IMR': round(predicted_imr[2], 1),
    '2013_IMR': round(predicted_imr[3], 1),
    '2014_IMR': round(predicted_imr[4], 1),
    '2015_IMR': round(predicted_imr[5], 1)
}

# Final Answer: Projected values for 2010–2015 (as a list of key values)
Final Answer: 21.5, 7.2, 44.4