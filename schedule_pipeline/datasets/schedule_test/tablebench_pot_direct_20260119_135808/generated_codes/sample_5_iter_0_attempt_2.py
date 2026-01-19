import pandas as pd

df = pd.read_csv('table.csv')
# Clean the '2001 general' column: remove text in parentheses and convert to float
df['2001 general'] = df['2001 general'].str.replace(r'\(.*\)', '', regex=True).astype(float)
# Calculate the average
average_2001_general = df['2001 general'].mean()
print(f"Final Answer: {average_2001_general:.1f}")