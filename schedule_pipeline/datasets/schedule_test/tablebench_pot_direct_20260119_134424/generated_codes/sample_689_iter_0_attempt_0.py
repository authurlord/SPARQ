import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'years of kindergarten legally required' to numeric, handling non-numeric values
df['years of kindergarten legally required'] = pd.to_numeric(df['years of kindergarten legally required'], errors='coerce')
# Find the canton with the maximum value
max_canton = df.loc[df['years of kindergarten legally required'].idxmax(), 'canton']
print(f"Final Answer: {max_canton}")