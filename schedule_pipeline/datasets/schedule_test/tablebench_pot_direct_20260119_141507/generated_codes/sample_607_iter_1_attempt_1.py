import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'DC' column to float
df['DC'] = pd.to_numeric(df['DC'], errors='coerce')

# Get DC values for Verona and Venice
verona_dc = df[df['Province'] == 'Verona']['DC'].values[0]
venice_dc = df[df['Province'] == 'Venice']['DC'].values[0]

# Calculate the difference
difference = verona_dc - venice_dc
print(f"Final Answer: {difference:.1f}")