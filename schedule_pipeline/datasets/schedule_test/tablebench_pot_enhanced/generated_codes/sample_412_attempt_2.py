import pandas as pd

df = pd.read_csv('table.csv')
# Describe the key columns and their features
print("The table contains data on disease cases over several years. Key columns include:")
print("- 'year': The year or range of years for which data is recorded.")
print("- 'typhus': Number of typhus cases.")
print("- 'typhoid fever': Number of typhoid fever cases.")
print("- 'relapsing fever': Number of relapsing fever cases.")
print("- 'smallpox': Number of smallpox cases.")
print("- 'malaria': Number of malaria cases, with some entries indicating an average value.")
print(f"Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria")