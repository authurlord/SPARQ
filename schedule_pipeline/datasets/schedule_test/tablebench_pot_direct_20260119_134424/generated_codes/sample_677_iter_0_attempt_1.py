import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total renewable generation' and 'total electricity demand' to numeric
df['total renewable generation'] = pd.to_numeric(df['total renewable generation'], errors='coerce')
df['total electricity demand'] = pd.to_numeric(df['total electricity demand'], errors='coerce')

# Calculate the correlation coefficient
correlation = df['total renewable generation'].corr(df['total electricity demand'])
print(f"Final Answer: {correlation:.3f}")