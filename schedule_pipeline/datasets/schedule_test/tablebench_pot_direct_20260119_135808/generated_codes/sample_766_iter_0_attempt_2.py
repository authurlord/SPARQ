import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])

# Extract year and total steel production
years = df['year'].astype(int)
production = df['total s ton']

# Fit a linear regression model
coefficients = np.polyfit(years, production, 1)
polynomial = np.poly1d(coefficients)

# Predict for 2007
forecast_2007 = polynomial(2007)

print(f"Final Answer: {int(forecast_2007)}")