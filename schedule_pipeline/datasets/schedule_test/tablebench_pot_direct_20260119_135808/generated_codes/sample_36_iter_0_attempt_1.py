import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tumbling' column to numeric
df['tumbling'] = pd.to_numeric(df['tumbling'])
# Calculate the average score in the 'tumbling' category
avg_tumbling = df['tumbling'].mean()
print(f"Final Answer: {avg_tumbling:.1f}")