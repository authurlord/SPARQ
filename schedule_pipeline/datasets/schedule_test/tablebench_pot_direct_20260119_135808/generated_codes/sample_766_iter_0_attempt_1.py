import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])

# Extract year and total steel production
years = df['year'].astype(int)
production = df['total s ton']

# Fit a linear trend (simple linear regression)
slope, intercept = np.polyfit(years, production, 1)

# Predict for 2007
predicted_2007 = slope * 2007 + intercept

print(f"Final Answer: {int(predicted_2007)}")