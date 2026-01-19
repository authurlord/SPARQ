import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year to numeric and create a feature (time index)
df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
df['year_num'] = df['year_num'].fillna(0)

# Prepare data for regression
X = df[['year_num']]  # independent variable: year
y_pass = df['pass percentage'].str.replace('%', '').astype(float)  # extract numeric pass percentage
y_gpa = df['obtained gpa - 5'].astype(int)  # obtained GPA is already numeric

# Fit linear regression models
model_pass = LinearRegression()
model_gpa = LinearRegression()

model_pass.fit(X, y_pass)
model_gpa.fit(X, y_gpa)

# Predict for year 2011 (which is 6 years after 2005)
year_2011 = 6
pass_pred = model_pass.predict([[year_2011]])[0]
gpa_pred = model_gpa.predict([[year_2011]])[0]

print(f"Final Answer: {pass_pred:.2f}%, {gpa_pred}")