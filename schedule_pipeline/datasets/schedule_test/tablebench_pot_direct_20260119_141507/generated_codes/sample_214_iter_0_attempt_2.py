import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns for correlation analysis
correlation_sales = df['sales (billion )'].corr(df['market value (billion )'])
correlation_profits = df['profits (billion )'].corr(df['market value (billion )'])
correlation_assets = df['assets (billion )'].corr(df['market value (billion )'])

# Find the highest absolute correlation
correlations = {
    'sales': correlation_sales,
    'profits': correlation_profits,
    'assets': correlation_assets
}

primary_driver = max(correlations, key=lambda x: abs(correlations[x]))
print(f"Final Answer: {primary_driver}")