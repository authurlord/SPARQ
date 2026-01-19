import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where 'CSAR No.' is missing (i.e., '-')
csar_no = df['CSAR No.'].replace('-', pd.NA).dropna()
# Convert to numeric and calculate median
median_csar = csar_no.astype(int).median()
print(f"Final Answer: {median_csar}")