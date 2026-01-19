import pandas as pd

df = pd.read_csv('table.csv')
# Extract the row for 'spanish' and find the municipality with the highest value
spanish_row = df[df['language'] == 'spanish']
municipalities = df.columns[1:]  # Exclude the 'language' column
max_population = spanish_row[municipalities].values[0].max()
max_municipality = municipalities[spanish_row[municipalities].values[0].argmax()]
print(f"Final Answer: {max_municipality}")