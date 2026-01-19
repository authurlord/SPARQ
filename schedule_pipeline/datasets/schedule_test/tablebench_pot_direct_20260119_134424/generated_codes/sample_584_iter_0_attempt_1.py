import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years between 2002 and 2005
filtered_df = df[(df['year'].astype(int) >= 2002) & (df['year'].astype(int) <= 2005)]
# Convert 'issue price' to float and calculate mean
average_price = filtered_df['issue price'].astype(float).mean()
print(f"Final Answer: {average_price:.2f}")