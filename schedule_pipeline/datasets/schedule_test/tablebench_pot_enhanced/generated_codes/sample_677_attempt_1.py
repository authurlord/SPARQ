import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric, removing non-numeric characters like '%' or commas
df['total renewable generation'] = pd.to_numeric(df['total renewable generation'], errors='coerce')
df['total electricity demand'] = pd.to_numeric(df['total electricity demand'], errors='coerce')

# Calculate correlation coefficient
correlation = df['total renewable generation'].corr(df['total electricity demand'])

print(f"Final Answer: {correlation:.3f}")