import pandas as pd

df = pd.read_csv('table.csv')
# Filter films directed by Mostofa Sarwar Farooki
farooki_films = df[df['Director'] == 'Mostofa Sarwar Farooki']
# Check for submissions to Academy Awards in Notes
academy_submissions = farooki_films[farooki_films['Notes'].str.contains('submission to Academy Awards', case=False, na=False)]
count = len(academy_submissions)
print(f"Final Answer: {count}")