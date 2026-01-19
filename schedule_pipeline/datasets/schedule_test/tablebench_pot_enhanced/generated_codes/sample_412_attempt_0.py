import pandas as pd

df = pd.read_csv('table.csv')
# Display the column names and their descriptions
print("Key columns in the table:")
print("- year: Represents the year or range of years for which data is recorded.")
print("- typhus: Number of typhus cases reported.")
print("- typhoid fever: Number of typhoid fever cases reported.")
print("- relapsing fever: Number of relapsing fever cases reported.")
print("- smallpox: Number of smallpox cases reported.")
print("- malaria: Number of malaria cases reported, with some entries showing averages.")

# Summary of the table
print("\nMain features of the table:")
print("- The table contains historical disease case data from 1913 to 1935.")
print("- It includes multiple infectious disease counts over time.")
print("- Some values are averages (e.g., '2940 (avg)') indicating aggregated data.")
print("- The data shows trends in disease prevalence over time, with peaks in certain years.")

# Final answer summarizing the key features
print("Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria")