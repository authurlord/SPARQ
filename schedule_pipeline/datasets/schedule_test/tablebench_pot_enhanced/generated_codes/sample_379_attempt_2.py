import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to appropriate types (Natural change and Deaths are strings with spaces)
df['Natural change'] = df['Natural change'].str.replace(' ', '').astype(int)
df['Deaths'] = df['Deaths'].astype(int)

# Filter conditions
filtered_df = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]

# Count the number of years
count_years = len(filtered_df)
print(f"Final Answer: {count_years}")