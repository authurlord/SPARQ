import pandas as pd

df = pd.read_csv('table.csv')
# Filter films directed by Mostofa Sarwar Farooki and check if notes mention Academy Awards submission
filtered_films = df[(df['Director'] == 'Mostofa Sarwar Farooki') & 
                    (df['Notes'].str.contains('Academy Awards', case=False, na=False))]

count_academy_submissions = len(filtered_films)
print(f"Final Answer: {count_academy_submissions}")