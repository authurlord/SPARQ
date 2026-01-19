import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'US Chart position' column by removing text in parentheses and converting to numeric
df['US Chart position'] = df['US Chart position'].astype(str).str.replace(r'\s*\(.*\)', '', regex=True)
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')
# Calculate the average, ignoring NaN values
average_chart_position = df['US Chart position'].mean()
print(f"Final Answer: {average_chart_position:.1f}")