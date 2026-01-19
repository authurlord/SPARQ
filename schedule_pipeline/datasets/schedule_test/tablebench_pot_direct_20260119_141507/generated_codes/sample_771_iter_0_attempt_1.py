import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Prepare data: years from 2005 to 2010
years = np.array([2005, 2006, 2007, 2008, 2009, 2010]).reshape(-1, 1)
pass_percentage = np.array([67.75, 72.37, 68.62, 75.54, 78.35, 79.68])
gpa = np.array([31, 54, 63, 79, 83, 85])

# Fit linear regression for pass percentage
model_pass = LinearRegression()
model_pass.fit(years, pass_percentage)
predicted_pass_2011 = model_pass.predict([[2011]])[0]

# Fit linear regression for GPA
model_gpa = LinearRegression()
model_gpa.fit(years, gpa)
predicted_gpa_2011 = model_gpa.predict([[2011]])[0]

print(f"Final Answer: {predicted_pass_2011:.2f}%, {predicted_gpa_2011:.0f}")