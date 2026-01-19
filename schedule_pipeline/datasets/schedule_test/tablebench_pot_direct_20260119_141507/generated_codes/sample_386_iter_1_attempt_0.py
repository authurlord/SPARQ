import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'seats won' to numeric, coercing errors to NaN if any
df['seats won'] = pd.to_numeric(df['seats won'], errors='coerce')

# Filter parties where seats won is 10 or fewer
filtered_parties = df[df['seats won'] <= 10]

# Count the number of such parties
count = len(filtered_parties)
print(f"Final Answer: {count}")