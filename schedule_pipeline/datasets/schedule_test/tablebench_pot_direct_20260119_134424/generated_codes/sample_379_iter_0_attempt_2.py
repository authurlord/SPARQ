import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to appropriate data types
df['Natural change'] = df['Natural change'].astype(int)
df['Deaths'] = df['Deaths'].astype(int)

# Filter rows based on conditions
filtered_df = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]

# Count the number of years satisfying the conditions
count_years = len(filtered_df)
print(f"Final Answer: {count_years}")