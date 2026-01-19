import pandas as pd

df = pd.read_csv('table.csv')
# Description of the main features of the table
columns = df.columns.tolist()
description = (
    f"Key columns: {', '.join(columns)}\n"
    f"Description: The table records historical disease case counts by year. "
    f"Columns include 'year', 'typhus', 'typhoid fever', 'relapsing fever', 'smallpox', and 'malaria'. "
    f'Malaria has an average value noted in one row.'
)
print(description)
Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria