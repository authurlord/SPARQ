import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Verona and Venice
verona_dc = float(df[df['Province'] == 'Verona']['DC'].iloc[0])
venice_dc = float(df[df['Province'] == 'Venice']['DC'].iloc[0])
# Calculate the difference
difference = verona_dc - venice_dc
print(f"Final Answer: {difference:.1f}")