import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant language columns for Spanish speakers
spanish_columns = ['spanish', 'only spanish', 'native and spanish']

# Sum the Spanish-related values for each municipality
df['total_spanish_speakers'] = df[spanish_columns].sum(axis=1)

# Find the municipality with the highest total
municipality_with_most_spanish = df.loc[df['total_spanish_speakers'].idxmax(), 'language']

print(f"Final Answer: {municipality_with_most_spanish}")