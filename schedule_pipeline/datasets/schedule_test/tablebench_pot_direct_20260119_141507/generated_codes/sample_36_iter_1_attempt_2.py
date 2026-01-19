import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'tumbling' column and compute its mean
mean_tumbling = df['tumbling'].mean()
print(f"Final Answer: {mean_tumbling:.1f}")