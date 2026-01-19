import pandas as pd

df = pd.read_csv('table.csv')
# Convert time column to float and calculate the mean
average_time = df['time'].astype(float).mean()
print(f"Final Answer: {average_time:.2f}")