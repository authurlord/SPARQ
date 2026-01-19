import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'seats won' column to integer for numerical comparison
df['seats won'] = pd.to_numeric(df['seats won'], errors='coerce')
# Count parties with seats won <= 10
count_parties = df[df['seats won'] <= 10].shape[0]
print(f"Final Answer: {count_parties}")