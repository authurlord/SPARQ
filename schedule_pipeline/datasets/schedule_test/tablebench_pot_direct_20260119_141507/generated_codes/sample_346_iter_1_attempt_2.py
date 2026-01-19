import pandas as pd

df = pd.read_csv('table.csv')
# Convert the 'spanish' column to integer (and ensure proper type)
df['spanish'] = pd.to_numeric(df['spanish'], errors='coerce')

# Count municipalities where Spanish-speaking population is >= 40,000
municipalities = df[df['spanish'] >= 40000]
count = len(municipalities)
print(f"Final Answer: {count}")