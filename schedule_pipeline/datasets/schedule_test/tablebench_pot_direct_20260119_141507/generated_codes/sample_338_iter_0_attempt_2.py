import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter for Telugu movies in 1955
telugu_1955 = df[(df['Year'] == '1955') & (df['Language'] == 'Telugu')]

# Get the director of the 1955 Telugu movie
director_1955_telugu = telugu_1955['Director'].values[0]

# Check if this director also directed a Tamil movie in 1956
tamil_1956 = df[(df['Year'] == '1956') & (df['Language'] == 'Tamil')]
director_1956_tamil = tamil_1956['Director'].values[0]

# If the directors match, the answer is the title of the 1955 Telugu movie
if director_1955_telugu == director_1956_tamil:
    final_answer = telugu_1955['Title'].values[0]
else:
    final_answer = None

print(f"Final Answer: {final_answer}")