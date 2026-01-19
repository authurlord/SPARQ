import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'males (%)' column to numeric
df['males (%)'] = pd.to_numeric(df['males (%)'])
# Calculate the average percentage of males
average_males = df['males (%)'].mean()
print(f"Final Answer: {average_males:.1f}")