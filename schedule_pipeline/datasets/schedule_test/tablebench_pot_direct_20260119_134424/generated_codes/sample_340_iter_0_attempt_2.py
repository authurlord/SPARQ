import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'Other artist(s)' column: handle missing or empty values
df['Other artist(s)'] = df['Other artist(s)'].astype(str).str.strip()
df['Other artist(s)'] = df['Other artist(s)'].replace('-', '')

# Create a dictionary to store composer -> set of unique singers
composer_singers = {}

for _, row in df.iterrows():
    composer = row['Composer']
    singers = row['Other artist(s)'].split(',')
    singers = [s.strip() for s in singers if s.strip()]  # Remove empty strings
    if composer not in composer_singers:
        composer_singers[composer] = set()
    composer_singers[composer].update(singers)

# Find the composer with the maximum number of unique singers
max_singers = 0
most_composer = None
for composer, singers in composer_singers.items():
    if len(singers) > max_singers:
        max_singers = len(singers)
        most_composer = composer

print(f"Final Answer: {most_composer}")