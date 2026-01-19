import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where natural change > 150 and deaths < 350
filtered_rows = df[(df['Natural change'] > 150) & (df['Deaths'] < 350)]
count = len(filtered_rows)
print(f"Final Answer: {count}")