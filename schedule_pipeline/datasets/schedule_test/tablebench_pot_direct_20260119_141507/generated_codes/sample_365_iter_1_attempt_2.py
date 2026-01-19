import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to float and count jurisdictions with more than 70% in favor
df['percent for'] = pd.to_numeric(df['percent for'], errors='coerce')
count_above_70 = df[df['percent for'] > 70]['percent for'].count()
print(f"Final Answer: {count_above_70}")