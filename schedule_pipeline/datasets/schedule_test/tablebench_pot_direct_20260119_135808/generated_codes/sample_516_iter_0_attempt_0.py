import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'entered service' and 'withdrawn' to integers
df['entered service'] = pd.to_numeric(df['entered service'])
df['withdrawn'] = pd.to_numeric(df['withdrawn'])
# Calculate service life
df['service life'] = df['withdrawn'] - df['entered service']
# Find the maximum service life
max_service_life = df['service life'].max()
print(f"Final Answer: {max_service_life}")