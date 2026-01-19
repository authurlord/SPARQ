import pandas as pd

df = pd.read_csv('table.csv')
# Filter provinces with PCI > 12
filtered_df = df[df['PCI'].astype(float) > 12]
# Convert DC and PSI to float
filtered_df['DC'] = filtered_df['DC'].astype(float)
filtered_df['PSI'] = filtered_df['PSI'].astype(float)
# Sort by DC to see the trend
sorted_df = filtered_df.sort_values(by='DC')
# Display the result
print(sorted_df[['Province', 'DC', 'PSI']])