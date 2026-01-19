import pandas as pd

df = pd.read_csv('table.csv')
# Extract the row for Spanish speakers
spanish_row = df[df['language'] == 'spanish']
# Get the values for each municipality
municipalities = df.columns[1:]  # Exclude the 'language' column
spanish_values = spanish_row[municipalities].values.flatten()
# Find the municipality with the highest Spanish speaker population
max_municipality = municipalities[spanish_values.argmax()]
print(f"Final Answer: {max_municipality}")