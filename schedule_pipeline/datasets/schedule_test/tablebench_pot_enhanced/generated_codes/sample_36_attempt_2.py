import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tumbling' column to numeric, handling any non-numeric values
df['tumbling'] = pd.to_numeric(df['tumbling'], errors='coerce')
# Calculate the average score in the 'tumbling' category
average_tumbling = df['tumbling'].mean()
print(f"Final Answer: {average_tumbling:.1f}")