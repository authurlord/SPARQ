import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Correct column name for resistance (fixing the symbol)
df.columns = ['frequency (hz)', 'r (Ω / km)', 'l (mh / km)', 'g (μs / km)', 'c (nf / km)']

# Convert frequency and resistance to numeric
df['frequency (hz)'] = pd.to_numeric(df['frequency (hz)'], errors='coerce')
df['r (Ω / km)'] = pd.to_numeric(df['r (Ω / km)'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Check if there's a positive trend (increasing frequency → increasing resistance)
correlation = df['frequency (hz)'].corr(df['r (Ω / km)'])

# Determine trend based on correlation
if correlation > 0.7:
    trend = "positive"
elif correlation < -0.7:
    trend = "negative"
else:
    trend = "no clear trend"

print(f"Final Answer: {trend}")