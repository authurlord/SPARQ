import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years between 2002 and 2005 inclusive
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2005)]
# Calculate the mean of 'issue price' (convert to float)
average_price = filtered_df['issue price'].astype(float).mean()
print(f"Final Answer: {average_price:.2f}")