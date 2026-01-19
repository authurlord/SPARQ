import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tumbling' column to numeric, handling any non-numeric values like '(13)'
df['tumbling'] = pd.to_numeric(df['tumbling'], errors='coerce')
# Calculate the average of the 'tumbling' column
average_tumbling = df['tumbling'].mean()
print(f"Final Answer: {average_tumbling:.1f}")