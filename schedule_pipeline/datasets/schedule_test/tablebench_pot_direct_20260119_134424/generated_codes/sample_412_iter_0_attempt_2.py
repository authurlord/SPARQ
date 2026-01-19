import pandas as pd

df = pd.read_csv('table.csv')
# Describe the main features of the table
print("The table contains health-related disease data over several years.")
print("Key columns include:")
print("- 'year': The year or range of years the data corresponds to.")
print("- 'typhus': Number of typhus cases.")
print("- 'typhoid fever': Number of typhoid fever cases.")
print("- 'relapsing fever': Number of relapsing fever cases.")
print("- 'smallpox': Number of smallpox cases.")
print("- 'malaria': Number of malaria cases, with some entries showing averages (e.g., '2940 (avg)').")
print(f"Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria")