import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean and split 'Other artist(s)' to extract individual singers
def extract_singers(singer_str):
    if pd.isna(singer_str) or singer_str == '-':
        return []
    # Remove quotes and split by comma
    singers = [s.strip() for s in singer_str.replace('"', '').split(',')]
    return singers

# Apply the function to get list of singers per row
singers_list = df['Other artist(s)'].apply(extract_singers)

# Group by composer and count unique singers
composer_singer_counts = df.groupby('Composer')['Other artist(s)'].apply(
    lambda x: len(set([s.strip() for singer_list in x.apply(extract_singers) for s in singer_list if s.strip()]))
).reset_index()

# Rename and ensure we have the correct count per composer
composer_singer_counts.columns = ['Composer', 'Unique_Singers']

# Find the composer with the maximum unique singers
max_singer_composer = composer_singer_counts.loc[composer_singer_counts['Unique_Singers'].idxmax(), 'Composer']
print(f"Final Answer: {max_singer_composer}")