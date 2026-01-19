import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer type for filtering
df['year'] = df['year'].astype(int)
# Filter for years from 2000 to 2004 (inclusive)
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2004)]
# Calculate average mintage (proof) for the filtered rows
avg_mintage_proof = filtered_df['mintage (proof)'].mean()
print(f"Final Answer: {avg_mintage_proof:.2f}")