import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'spanish' row and sum across municipalities
spanish_speakers = df[df['language'] == 'spanish'].iloc[0, 1:]
# Find the municipality with the highest number of Spanish speakers
max_municipality = spanish_speakers.idxmax()
print(f"Final Answer: {max_municipality}")