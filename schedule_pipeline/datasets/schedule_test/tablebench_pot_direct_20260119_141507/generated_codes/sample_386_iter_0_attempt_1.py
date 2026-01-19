import pandas as pd

df = pd.read_csv('table.csv')
# Count parties with 10 or fewer seats won
count = df[df['seats won'] <= 10].shape[0]
print(f"Final Answer: {count}")