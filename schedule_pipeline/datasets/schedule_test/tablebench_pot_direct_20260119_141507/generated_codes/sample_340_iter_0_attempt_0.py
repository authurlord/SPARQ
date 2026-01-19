import pandas as pd

df = pd.read_csv('table.csv')

# Extract the "Other artist(s)" column and clean it
singers_list = []
for value in df['Other artist(s)']:
    if pd.isna(value) or value == '-':
        continue
    # Split by comma and strip whitespace
    singers = [s.strip() for s in value.split(',')]
    singers_list.extend(singers)

# Group by composer and count unique singers per composer
composer_singer_count = {}
for index, row in df.iterrows():
    composer = row['Composer']
    singers = row['Other artist(s)']
    if pd.isna(singers) or singers == '-':
        continue
    singer_list = [s.strip() for s in singers.split(',')]
    for singer in singer_list:
        if composer not in composer_singer_count:
            composer_singer_count[composer] = set()
        composer_singer_count[composer].add(singer)

# Find the composer with the maximum number of unique singers
max_singers = 0
result_composer = None
for composer, singers in composer_singer_count.items():
    if len(singers) > max_singers:
        max_singers = len(singers)
        result_composer = composer

print(f"Final Answer: {result_composer}")