import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'males (%)' column to numeric, handling any parsing issues
df['males (%)'] = pd.to_numeric(df['males (%)'], errors='coerce')
# Calculate the average of males percentage
average_males = df['males (%)'].mean()
print(f"Final Answer: {average_males:.1f}")