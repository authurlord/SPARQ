import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to float and filter jurisdictions with more than 70% in favor
df['percent for'] = df['percent for'].str.replace('%', '').astype(float)
jurisdictions_above_70 = df[df['percent for'] > 70]['jurisdiction'].count()
print(f"Final Answer: {jurisdictions_above_70}")