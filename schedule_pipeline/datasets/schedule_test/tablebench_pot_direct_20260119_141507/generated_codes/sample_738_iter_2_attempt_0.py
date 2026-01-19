import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert year column to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Prepare features and target variables
X = df[['year']]
y_goondiwindi = df['goondiwindi']
y_waggamba = df['waggamba']
y_inglewood = df['inglewood']

# Initialize models
models = {
    'goondiwindi': LinearRegression(),
    'waggamba': LinearRegression(),
    'inglewood': LinearRegression()
}

# Fit models
models['goondiwindi'].fit(X, y_goondiwindi)
models['waggamba'].fit(X, y_waggamba)
models['inglewood'].fit(X, y_inglewood)

# Predict for year 2021
year_2021 = np.array([[2021]])
forecast_goondiwindi = models['goondiwindi'].predict(year_2021)[0]
forecast_waggamba = models['waggamba'].predict(year_2021)[0]
forecast_inglewood = models['inglewood'].predict(year_2021)[0]

print(f"Final Answer: {forecast_goondiwindi:.0f}, {forecast_waggamba:.0f}, {forecast_inglewood:.0f}")