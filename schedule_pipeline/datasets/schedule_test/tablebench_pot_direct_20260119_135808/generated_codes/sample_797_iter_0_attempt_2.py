import pandas as pd

df = pd.read_csv('table.csv')
# Drop rows where 'CSAR No.' is empty or missing
csar_no = df['CSAR No.'].dropna()
# Convert to numeric to ensure proper calculation
csar_no = pd.to_numeric(csar_no, errors='coerce').dropna()
# Calculate median
median_csar = csar_no.median()
print(f"Final Answer: {median_csar}")