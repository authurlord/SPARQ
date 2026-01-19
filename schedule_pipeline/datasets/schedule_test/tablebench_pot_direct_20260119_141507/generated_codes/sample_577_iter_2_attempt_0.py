import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter years from 2000 to 2004 (first 5 years of 2000s)
filtered_df = df[df['year'].between(2000, 2004)]
# Convert 'mintage (proof)' to numeric, ignoring non-numeric values
mintage_proof_numeric = pd.to_numeric(filtered_df['mintage (proof)'], errors='coerce')
# Calculate the mean of valid values
average_mintage = mintage_proof_numeric.mean()
print(f"Final Answer: {average_mintage:.1f}")