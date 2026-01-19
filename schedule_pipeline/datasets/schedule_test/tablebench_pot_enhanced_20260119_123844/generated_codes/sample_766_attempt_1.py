import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])

# Calculate the average annual growth rate
years = df['year'].astype(int)
production = df['total s ton']
growth_rates = [((production[i] - production[i-1]) / production[i-1]) for i in range(1, len(production))]
avg_growth_rate = sum(growth_rates) / len(growth_rates)

# Forecast for 2007
last_production = production.iloc[-1]
forecast_2007 = last_production * (1 + avg_growth_rate)

print(f"Final Answer: {forecast_2007:.0f}")