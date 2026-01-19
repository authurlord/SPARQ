import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract relevant columns: Year_2 and -_2 (values for each decade)
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Create a linear regression model
X = years.values.reshape(-1, 1)
y = values.values

# Fit linear model
model = np.polyfit(X.flatten(), y, 1)
predicted_2020 = np.polyval(model, 2020)

print(f"Final Answer: {int(predicted_2020)}")