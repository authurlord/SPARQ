import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'US Chart position' column by extracting only numeric values
df['US Chart position'] = df['US Chart position'].str.replace(r'\(.*\)', '', regex=True).astype(float)
# Calculate the average
avg_chart_position = df['US Chart position'].mean()
print(f"Final Answer: {avg_chart_position:.1f}")