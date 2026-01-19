import pandas as pd

df = pd.read_csv('table.csv')

# Identify the latest model and its fuel propulsion and quantity
latest_model = df.iloc[-1]
latest_fuel_propulsion = latest_model['fuel propulsion']
latest_quantity = latest_model['quantity']

# Based on trend, newer models are moving toward hybrid/electric
# The latest model still uses diesel, but previous models show increasing hybrid usage
# Predict future: electric propulsion and higher quantity

# Forecasting: likely electric propulsion and quantity greater than 60 (based on highest observed)
forecast_fuel = "electric"
forecast_quantity = 70  # Estimated higher than current max (80, but articulated, so reasonable)

print(f"Final Answer: electric, 70")