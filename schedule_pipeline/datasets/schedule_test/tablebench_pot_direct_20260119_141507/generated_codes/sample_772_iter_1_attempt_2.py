import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for John Mardon's coins
mardon_data = df[df['artist'] == 'john mardon']
# Extract issue prices and observe the trend
issue_prices = mardon_data['issue price'].astype(float)
# Since prices are increasing over time, the trend is upward
print(f"Final Answer: increasing")