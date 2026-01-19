import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2000 to 2004
filtered_df = df[df['year'].between(2000, 2004)]
# Calculate the mean of 'mintage (proof)'
average_mintage_proof = filtered_df['mintage (proof)'].mean()
print(f"Final Answer: {average_mintage_proof:.1f}")