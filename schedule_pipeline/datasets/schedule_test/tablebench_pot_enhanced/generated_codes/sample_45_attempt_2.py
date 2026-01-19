import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Start' and 'End' columns to integers
df['Start'] = pd.to_numeric(df['Start'])
df['End'] = pd.to_numeric(df['End'])

# Calculate the duration of service for each ambassador
df['Duration'] = df['End'] - df['Start']

# Sum the durations to get total years served
total_years = df['Duration'].sum()
print(f"Final Answer: {total_years}")