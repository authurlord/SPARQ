import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Deaths' and 'Natural change' to numeric, coercing errors to NaN if needed
df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
df['Natural change'] = pd.to_numeric(df['Natural change'], errors='coerce')

# Filter rows where natural change > 150 and deaths < 350
filtered_rows = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]

# Count the number of such years
count = len(filtered_rows)
print(f"Final Answer: {count}")