import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'CSAR No.' is not missing (not '-')
df_filtered = df[df['CSAR No.'] != '-']
# Convert 'CSAR No.' to numeric (removes non-numeric entries)
csar_numeric = pd.to_numeric(df_filtered['CSAR No.'], errors='coerce')
# Drop any remaining NaN values (due to invalid entries)
csar_numeric = csar_numeric.dropna()
# Calculate median
median_csar = csar_numeric.median()
print(f"Final Answer: {median_csar}")