import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for years 2000 to 2004 (first 5 years of the 2000s)
filtered_df = df[(df['year'].astype(str).str.startswith('200')) & (df['year'] <= 2004)]

# Convert 'mintage (proof)' to numeric, handling missing or non-numeric values
filtered_df['mintage (proof)'] = pd.to_numeric(filtered_df['mintage (proof)'], errors='coerce')

# Drop rows where mintage (proof) is NaN (i.e., missing or invalid)
filtered_df = filtered_df.dropna(subset=['mintage (proof)'])

# Calculate the average mintage (proof)
average_mintage = filtered_df['mintage (proof)'].mean()

print(f"Final Answer: {average_mintage:.1f}")