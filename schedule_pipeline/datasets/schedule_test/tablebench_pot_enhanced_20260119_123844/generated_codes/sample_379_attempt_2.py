import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Natural change' and 'Deaths' columns to numeric (handling any formatting issues like spaces)
df['Natural change'] = pd.to_numeric(df['Natural change'], errors='coerce')
df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')

# Apply the conditions
filtered_df = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]

# Count the number of years satisfying the conditions
count_years = len(filtered_df)

print(f"Final Answer: {count_years}")