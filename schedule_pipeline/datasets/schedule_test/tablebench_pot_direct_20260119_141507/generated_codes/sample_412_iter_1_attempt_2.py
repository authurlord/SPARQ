import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Describe the main features of the table
columns = df.columns.tolist()
description = (
    f"The table has the following columns: {', '.join(columns)}. "
    f"Each row represents data for a specific year, and the columns indicate the number of reported cases "
    f"of various diseases (typhus, typhoid fever, relapsing fever, smallpox, malaria) during that year. "
    f"The 'malaria' column contains a notable outlier in 1934 with a high value (9477), and the 191822 row appears to be an exception due to its different format (with an average value in parentheses)."
)

print(description)