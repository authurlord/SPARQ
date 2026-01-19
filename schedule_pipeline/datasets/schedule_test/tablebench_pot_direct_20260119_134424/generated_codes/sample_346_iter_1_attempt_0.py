import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'spanish' column to numeric
df['spanish'] = pd.to_numeric(df['spanish'], errors='coerce')
# Filter municipalities with Spanish speakers >= 40,000
high_spanish_municipalities = df[df['spanish'] >= 40000]
# Count the number of such municipalities
count = len(high_spanish_municipalities)
print(f"Final Answer: {count}")