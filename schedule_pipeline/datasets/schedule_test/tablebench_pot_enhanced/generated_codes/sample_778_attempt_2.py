import pandas as pd

df = pd.read_csv('table.csv')
# Extract the data for 'Year_2' and '-_2'
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Calculate average annual growth rate from 1950 to 2010
start_year = 1950
end_year = 2010
n_years = end_year - start_year
avg_increase = (values.iloc[-1] - values.iloc[0]) / n_years

# Forecast for 2020
forecast_2020 = values.iloc[-1] + avg_increase * (2020 - end_year)

print(f"Final Answer: {int(forecast_2020)}")