import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2000 to 2004
filtered_df = df[df['year'].between(2000, 2004)]
# Extract mintage (proof) and compute mean, ignoring NaN
mean_mintage_proof = filtered_df['mintage (proof)'].dropna().mean()
print(f"Final Answer: {mean_mintage_proof:.1f}")