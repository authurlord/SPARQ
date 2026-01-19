import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Natural change > 150 and Deaths < 350
filtered_rows = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]
count = len(filtered_rows)
print(f"Final Answer: {count}")