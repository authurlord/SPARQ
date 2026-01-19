import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers and calculate duration
df['duration'] = df['End'].astype(int) - df['Start'].astype(int)
# Find the person with the maximum duration
longest_serving = df.loc[df['duration'].idxmax(), 'Name']
print(f"Final Answer: {longest_serving}")