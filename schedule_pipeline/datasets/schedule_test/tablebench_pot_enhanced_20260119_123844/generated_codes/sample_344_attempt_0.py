import pandas as pd

df = pd.read_csv('table.csv')
# Filter engines with specific impulse greater than 1000 seconds
filtered_engines = df[df['specific impulse (s)'] > 1000]
count = len(filtered_engines)
print(f"Final Answer: {count}")