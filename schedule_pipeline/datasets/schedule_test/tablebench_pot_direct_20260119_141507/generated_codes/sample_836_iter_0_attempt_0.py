import pandas as pd

df = pd.read_csv('table.csv')
# Find the municipality with the highest Spanish speakers
spanish_speakers = df['Spanish']
max_population = spanish_speakers.max()
# Get the corresponding municipality name
municipality_with_max_spanish = df.columns[spanish_speakers.idxmax()]
print(f"Final Answer: {municipality_with_max_population}")