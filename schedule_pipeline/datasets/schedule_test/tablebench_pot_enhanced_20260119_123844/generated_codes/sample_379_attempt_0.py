import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Natural change' and 'Deaths' columns to numeric (some values have spaces, e.g., '1 104')
df['Natural change'] = df['Natural change'].astype(str).str.replace(' ', '').astype(int)
df['Deaths'] = df['Deaths'].astype(str).str.replace(' ', '').astype(int)

# Apply the conditions
filtered_df = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]

# Count the number of years
count_years = len(filtered_df)
print(f"Final Answer: {count_years}")