import pandas as pd

df = pd.read_csv('table.csv')
# Filter CSAR No. column, remove missing values ('-'), convert to numeric, and compute median
csar_values = df['CSAR No.'].dropna().astype(str).str.replace('-', '').str.replace(' ', '').astype(float)
median_csar = csar_values.median()
print(f"Final Answer: {median_csar:.0f}")