import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
df['End'] = pd.to_numeric(df['End'], errors='coerce')
# Calculate the duration for each entry
duration = df['End'] - df['Start']
# Sum all durations
total_years = duration.sum()
print(f"Final Answer: {total_years}")