import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of the 'tumbling' column
average_tumbling = df['tumbling'].mean()
print(f"Final Answer: {average_tumbling:.1f}")