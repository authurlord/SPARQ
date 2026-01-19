import pandas as pd

df = pd.read_csv('table.csv')
# Filter films directed by Mostofa Sarwar Farooki
farooki_films = df[df['Director'] == 'Mostofa Sarwar Farooki']

# Check for submissions to Academy Awards
academy_submissions = farooki_films[farooki_films['Notes'].str.contains('Academy Awards|submission to Academy Awards', na=False)]

# Count the number of such films
count = len(academy_submissions)
print(f"Final Answer: {count}")