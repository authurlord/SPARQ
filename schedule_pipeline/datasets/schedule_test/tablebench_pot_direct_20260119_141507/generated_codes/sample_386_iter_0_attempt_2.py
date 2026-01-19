import pandas as pd

df = pd.read_csv('table.csv')
# Count parties where 'seats won' is 10 or fewer
count_parties = df[df['seats won'] <= 10].shape[0]
print(f"Final Answer: {count_parties}")