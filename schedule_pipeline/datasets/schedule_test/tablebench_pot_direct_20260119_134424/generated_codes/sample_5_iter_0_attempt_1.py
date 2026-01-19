import pandas as pd

df = pd.read_csv('table.csv')
# Clean the '2001 general' column by extracting numeric values
df['2001 general'] = df['2001 general'].astype(str).str.replace(r'\s*\([^)]+\)', '', regex=True)
df['2001 general'] = pd.to_numeric(df['2001 general'], errors='coerce')

# Calculate the average, ignoring NaN values
mean_2001_general = df['2001 general'].mean()
print(f"Final Answer: {mean_2001_general:.1f}")