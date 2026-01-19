import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric for calculation
km_from_kingston = pd.to_numeric(df['km from kingston'], errors='coerce')
length_meters = pd.to_numeric(df['length meters'], errors='coerce')

# Calculate correlation coefficient
correlation = km_from_kingston.corr(length_meters)
print(f"Final Answer: {correlation:.3f}")