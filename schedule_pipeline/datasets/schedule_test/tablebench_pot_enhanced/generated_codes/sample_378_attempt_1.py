import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where seats >= 75
count_elections = df[df['seats'] >= 75].shape[0]
print(f"Final Answer: {count_elections}")