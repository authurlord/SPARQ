import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of the 'tumbling' column
mean_tumbling = df['tumbling'].mean()
print(f"Final Answer: {mean_tumbling:.1f}")