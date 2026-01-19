import pandas as pd

df = pd.read_csv('table.csv')
# Count parties with seats won <= 10
count_parties = df[df['seats won'] <= 10].shape[0]
print(f"Final Answer: {count_parties}")