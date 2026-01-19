import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (millions)' to numeric, handling any potential parsing issues
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'], errors='coerce')

# Select the first 5 episodes and calculate the average viewers
first_five_viewers = df.iloc[:5]['viewers (millions)']
average_viewers = first_five_viewers.mean()

print(f"Final Answer: {average_viewers:.2f}")