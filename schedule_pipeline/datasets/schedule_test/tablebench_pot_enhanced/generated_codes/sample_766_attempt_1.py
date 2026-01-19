import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])

# Extract the years and total steel production
years = df['year'].astype(int)
production = df['total s ton']

# Calculate the average annual change in production
annual_change = production.diff().mean()

# Forecast for 2007
forecast_2007 = production.iloc[-1] + annual_change

print(f"Final Answer: {int(forecast_2007)}")