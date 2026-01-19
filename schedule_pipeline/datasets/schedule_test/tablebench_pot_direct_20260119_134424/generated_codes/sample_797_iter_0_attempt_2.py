import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where 'CSAR No.' is missing (i.e., '-')
csar_no = df['CSAR No.'].replace('-', pd.NA)
# Convert to numeric and drop NA values
csar_no = pd.to_numeric(csar_no, errors='coerce').dropna()
# Calculate median
median_csar = csar_no.median()
print(f"Final Answer: {median_csar}")