import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant columns: Year_2 and its corresponding value (-_2)
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Fit a linear model (y = mx + b)
m, b = np.polyfit(years, values, 1)

# Predict for year 2020
predicted_value = m * 2020 + b

print(f"Final Answer: {int(predicted_value)}")