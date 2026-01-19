import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average time from the 'time' column
average_time = df['time'].mean()
print(f"Final Answer: {average_time:.2f}")