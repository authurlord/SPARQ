import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'CSAR No.' column to numeric, coercing errors (like '-') to NaN
csar_no = pd.to_numeric(df['CSAR No.'], errors='coerce')
# Drop NaN values and compute median
median_csar = csar_no.dropna().median()
print(f"Final Answer: {median_csar}")