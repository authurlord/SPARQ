import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Verona and Venice
verona_dc = df[df['Province'] == 'Verona']['DC'].values[0]
venice_dc = df[df['Province'] == 'Venice']['DC'].values[0]
# Calculate difference
difference = float(verona_dc) - float(venice_dc)
print(f"Final Answer: {difference:.1f}")