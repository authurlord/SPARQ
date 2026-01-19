import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, handling any non-numeric entries
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['typhus', 'typhoid fever'], inplace=True)

# Calculate correlation
correlation = df['typhus'].corr(df['typhoid fever'])

# Since the question asks whether an increase in typhus leads to an increase in typhoid fever,
# we interpret positive correlation as "yes" and negative or near-zero as "no"
# But since we must output only the correlation value, we return it as is
print(f"Final Answer: {correlation:.2f}")