import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers and calculate the duration for each
df['duration'] = df['End'].astype(int) - df['Start'].astype(int)
# Sum the durations
total_years = df['duration'].sum()
print(f"Final Answer: {total_years}")