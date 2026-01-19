import pandas as pd

df = pd.read_csv('table.csv')
# Display the column names and a brief description
print("Key columns in the table:", df.columns.tolist())
print("The table contains annual disease incidence data from 1913 to 1935.")
print("It includes counts for typhus, typhoid fever, relapsing fever, smallpox, and malaria.")
print("Some entries contain averages (e.g., '2940 (avg)') indicating estimated values.")