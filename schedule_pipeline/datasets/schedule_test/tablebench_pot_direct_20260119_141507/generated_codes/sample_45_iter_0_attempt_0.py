import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers and calculate the duration in years
df['duration'] = df['End'].astype(int) - df['Start'].astype(int)
# Sum all durations
total_years = df['duration'].sum()
print(f"Final Answer: {total_years}")