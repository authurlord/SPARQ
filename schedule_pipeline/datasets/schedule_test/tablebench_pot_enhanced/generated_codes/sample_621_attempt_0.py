import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric, handling any potential non-numeric values
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Calculate the average age
average_age = df['Age'].mean()
print(f"Final Answer: {average_age:.1f}")