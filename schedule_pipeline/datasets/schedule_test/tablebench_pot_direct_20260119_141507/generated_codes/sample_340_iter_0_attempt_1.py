import pandas as pd

df = pd.read_csv('table.csv')

# Clean and split the "Other artist(s)" column to extract individual singers
singers_list = []
for value in df['Other artist(s)']:
    if pd.isna(value) or value == '-':
        continue
    singers = [s.strip() for s in value.split(',')]
    singers_list.extend(singers)

# Count unique singers per composer
composer_singer_count = {}
for idx, row in df.iterrows():
    composer = row['Composer']
    singers = row['Other artist(s)']
    if pd.isna(singers) or singers == '-':
        continue
    singers = [s.strip() for s in singers.split(',')]
    for singer in singers:
        if composer not in composer_singer_count:
            composer_singer_count[composer] = set()
        composer_singer_count[composer].add(singer)

# Find the composer with the most unique singers
max_singers = 0
result_composer = ""
for composer, singer_set in composer_singer_count.items():
    if len(singer_set) > max_singers:
        max_singers = len(singer_set)
        result_composer = composer

print(f"Final Answer: {result_composer}")