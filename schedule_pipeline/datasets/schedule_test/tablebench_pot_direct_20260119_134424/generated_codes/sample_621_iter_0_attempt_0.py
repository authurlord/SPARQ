import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric and calculate mean
average_age = df['Age'].astype(int).mean()
print(f"Final Answer: {average_age:.1f}")