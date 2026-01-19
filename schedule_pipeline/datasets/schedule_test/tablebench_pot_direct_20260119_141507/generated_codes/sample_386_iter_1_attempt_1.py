import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'seats won' to integer type
df['seats won'] = pd.to_numeric(df['seats won'], errors='coerce')
# Count parties where seats won is 10 or fewer
count_parties = df[df['seats won'] <= 10].shape[0]
print(f"Final Answer: {count_parties}")