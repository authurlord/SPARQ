import pandas as pd

df = pd.read_csv('table.csv')

# Display the column names and their descriptions
print("Key Columns and Their Features:")
print("year: Represents the year or range of years for the data.")
print("typhus: Number of typhus cases reported.")
print("typhoid fever: Number of typhoid fever cases reported.")
print("relapsing fever: Number of relapsing fever cases reported.")
print("smallpox: Number of smallpox cases reported.")
print("malaria: Number of malaria cases reported, with some entries having averages (e.g., '2940 (avg)')")

# Show basic info about the dataframe
print("\nData Summary:")
print(df.info())
print("\nFirst few rows of the table:")
print(df.head())