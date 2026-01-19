import pandas as pd

df = pd.read_csv('table.csv')
# Calculate service life as withdrawn - entered service
df['service_life'] = df['withdrawn'] - df['entered service']
# Find the maximum service life
max_service_life = df['service_life'].max()
print(f"Final Answer: {max_service_life}")