import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])

# Extract the total steel production values
production = df['total s ton']

# Calculate the average annual growth rate
growth_rate = (production.iloc[-1] - production.iloc[0]) / (len(production) - 1)

# Forecast 2007
forecast_2007 = production.iloc[-1] + growth_rate

print(f"Final Answer: {int(forecast_2007)}")