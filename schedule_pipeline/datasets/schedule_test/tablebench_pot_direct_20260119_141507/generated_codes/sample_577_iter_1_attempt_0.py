import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for proper comparison
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter rows where year is between 2000 and 2004 inclusive
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2004)]

# Calculate average mintage (proof) for the filtered rows
average_mintage_proof = filtered_df['mintage (proof)'].mean()

print(f"Final Answer: {average_mintage_proof:.1f}")