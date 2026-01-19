import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'US Chart position' column by extracting only numeric values
df['US Chart position'] = df['US Chart position'].astype(str).str.extract('(\d+)').astype(float)
# Calculate the average
average_position = df['US Chart position'].mean()
print(f"Final Answer: {average_position:.1f}")