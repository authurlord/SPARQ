import pandas as pd

df = pd.read_csv('table.csv')
# Filter provinces with PCI > 12
filtered_df = df[df['PCI'].astype(float) > 12]
# Display the DC and PSI values for these provinces
print(f"Final Answer: {filtered_df[['Province', 'DC', 'PSI']].to_string(index=False)}")