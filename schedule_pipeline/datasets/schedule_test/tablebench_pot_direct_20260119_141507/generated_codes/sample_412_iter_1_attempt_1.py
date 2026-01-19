import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Display the column names and a brief description of each
columns = df.columns.tolist()
description = {
    'year': 'The year in which the disease data was recorded.',
    'typhus': 'Number of reported cases of typhus.',
    'typhoid fever': 'Number of reported cases of typhoid fever.',
    'relapsing fever': 'Number of reported cases of relapsing fever.',
    'smallpox': 'Number of reported cases of smallpox.',
    'malaria': 'Number of reported cases of malaria.'
}

# Print the main features
print("Main features of the table:")
for col in columns:
    print(f"- {col}: {description[col]}")

Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria