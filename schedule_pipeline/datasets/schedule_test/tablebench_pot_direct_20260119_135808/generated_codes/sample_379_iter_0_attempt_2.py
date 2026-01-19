import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Natural change' and 'Deaths' columns to integers for comparison
df['Natural change'] = df['Natural change'].str.replace(' ', '').astype(int)
df['Deaths'] = df['Deaths'].astype(int)

# Filter rows where natural change > 150 and deaths < 350
filtered_rows = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]

# Count the number of such years
count_years = len(filtered_rows)

print(f"Final Answer: {count_years}")