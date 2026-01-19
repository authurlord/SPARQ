import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the years 1947 and 1972
conservative_1947 = df[df['Year'] == '1947']['Conservative councillors'].values[0]
conservative_1972 = df[df['Year'] == '1972']['Conservative councillors'].values[0]

# Calculate the change and average annual change
total_change = int(conservative_1972) - int(conservative_1947)
num_years = 1972 - 1947
average_annual_change = total_change / num_years

print(f"Final Answer: {average_annual_change:.2f}")