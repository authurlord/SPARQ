import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to float and filter jurisdictions with more than 70% in favor
count = df[df['percent for'].astype(float) > 70].shape[0]
print(f"Final Answer: {count}")