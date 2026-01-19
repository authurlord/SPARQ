import pandas as pd

df = pd.read_csv('table.csv')
# Filter provinces where PCI > 12
filtered_df = df[df['PCI'] > 12]

# Convert DC and PSI to numeric for correlation
df_numeric = filtered_df[['DC', 'PSI']].astype(float)

# Compute correlation between DC and PSI
correlation = df_numeric['DC'].corr(df_numeric['PSI'])
print(f"Final Answer: {correlation:.2f}")