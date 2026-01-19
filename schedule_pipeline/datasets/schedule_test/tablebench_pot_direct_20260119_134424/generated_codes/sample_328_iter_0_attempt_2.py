import pandas as pd

df = pd.read_csv('table.csv')
# Filter films directed by Mostofa Sarwar Farooki
farooki_films = df[df['Director'] == 'Mostofa Sarwar Farooki']
# Check for Academy Award submissions in the Notes column
academy_submissions = farooki_films[farooki_films['Notes'].str.contains('submission to Academy Awards', case=False, na=False)]
# Count the number of such films
count = len(academy_submissions)
print(f"Final Answer: {count}")