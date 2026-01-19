import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer type
df['year'] = pd.to_numeric(df['year'], errors='coerce')
# Filter years between 2000 and 2005 inclusive
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2005)]
# Sum the mintage
total_coins = filtered_df['mintage'].sum()
print(f"Final Answer: {total_coins}")