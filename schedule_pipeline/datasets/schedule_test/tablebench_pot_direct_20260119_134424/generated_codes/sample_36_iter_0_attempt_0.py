import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tumbling' column to numeric
df['tumbling'] = pd.to_numeric(df['tumbling'])
# Calculate the average
average_tumbling = df['tumbling'].mean()
print(f"Final Answer: {average_tumbling:.1f}")