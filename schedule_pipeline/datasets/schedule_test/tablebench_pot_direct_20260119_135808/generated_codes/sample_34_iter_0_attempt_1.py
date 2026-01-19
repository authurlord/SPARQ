import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'time' column to float for accurate calculation
df['time'] = pd.to_numeric(df['time'])
# Calculate the average time
average_time = df['time'].mean()
print(f"Final Answer: {average_time:.2f}")