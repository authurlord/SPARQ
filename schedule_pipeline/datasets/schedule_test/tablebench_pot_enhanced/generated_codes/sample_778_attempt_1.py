import pandas as pd

df = pd.read_csv('table.csv')
# Extract values for 2005 and 2010
value_2005 = int(df[df['Year_2'] == '2005']['-_2'].iloc[0].replace(',', ''))
value_2010 = int(df[df['Year_2'] == '2010']['-_2'].iloc[0].replace(',', ''))

# Calculate annual increase
annual_increase = (value_2010 - value_2005) / 5

# Forecast for 2020
forecast_2020 = value_2010 + (annual_increase * 10)

print(f"Final Answer: {forecast_2020:.0f}")