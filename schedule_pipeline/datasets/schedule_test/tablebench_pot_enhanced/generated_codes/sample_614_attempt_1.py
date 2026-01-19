import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Harold Snepsts and get his reg gp value
harold_gp = df[df['player'] == 'harold snepsts']['reg gp'].values[0]
# Calculate how many more games he played than the average
difference = int(harold_gp) - 300
print(f"Final Answer: {difference}")