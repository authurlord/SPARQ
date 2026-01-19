import pandas as pd

df = pd.read_csv('table.csv')
# Filter CSAR No. column, remove missing values ('-'), convert to numeric, and compute median
csar_values = df['CSAR No.'].dropna().astype(str).str.replace('-', '').str.strip()
# Convert to numeric, handling invalid entries by ignoring them
csar_numeric = pd.to_numeric(csar_values, errors='coerce')
# Drop NaNs from invalid conversions
csar_numeric = csar_numeric.dropna()
median_csar = csar_numeric.median()
print(f"Final Answer: {median_csar}")