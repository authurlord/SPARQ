import pandas as pd

df = pd.read_csv('table.csv')
# Get DC values for Verona and Venice
verona_dc = df[df['Province'] == 'Verona']['DC'].values[0]
venice_dc = df[df['Province'] == 'Venice']['DC'].values[0]
difference = verona_dc - venice_dc
print(f"Final Answer: {difference}")