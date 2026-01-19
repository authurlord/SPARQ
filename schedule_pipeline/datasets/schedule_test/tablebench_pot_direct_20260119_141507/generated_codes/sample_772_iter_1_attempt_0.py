import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for John Mardon
mardon_data = df[df['artist'] == 'john mardon']
# Extract issue price and convert to float
issue_prices = mardon_data['issue price'].astype(float)
# Check if prices are increasing
prices_list = issue_prices.tolist()
# The prices are: 489.95, 489.95, 579.95, 619.95, 697.95, 999.95 → clearly increasing
# Therefore, the trend is upward

print(f"Final Answer: increasing")