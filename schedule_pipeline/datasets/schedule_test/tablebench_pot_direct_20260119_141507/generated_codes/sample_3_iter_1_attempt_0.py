import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric, ensuring proper parsing
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Drop any rows with invalid age (though none expected here)
df = df.dropna(subset=['Age'])
# Calculate the average age
average_age = df['Age'].mean()
print(f"Final Answer: {average_age:.1f}")