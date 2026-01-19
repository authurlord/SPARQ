import pandas as pd

df = pd.read_csv('table.csv')
# Filter engines with specific impulse > 1000 seconds
count = df[df['specific impulse (s)'] > 1000].shape[0]
print(f"Final Answer: {count}")