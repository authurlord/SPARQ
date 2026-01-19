import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric, in case it's stored as string
df['Age'] = pd.to_numeric(df['Age'])
# Calculate average age
average_age = df['Age'].mean()
print(f"Final Answer: {average_age:.1f}")