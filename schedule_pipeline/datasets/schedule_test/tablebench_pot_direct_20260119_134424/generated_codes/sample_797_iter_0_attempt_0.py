import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'CSAR No.' column to numeric, invalid parsing will be set as NaN
csar_no = pd.to_numeric(df['CSAR No.'], errors='coerce')
# Drop missing values
csar_no_clean = csar_no.dropna()
# Calculate median
median_csar = csar_no_clean.median()
print(f"Final Answer: {median_csar}")