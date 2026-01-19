import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display a detailed description of the table
print("Table Description:")
print("The table contains key metrics related to university admissions across five years (2013 to 2017).")
print("\nColumns:")
print("- 'Applications': Number of applicants each year.")
print("- 'Offer Rate (%)': Percentage of applicants offered a place.")
print("- 'Enrols': Number of students who accepted offers and enrolled.")
print("- 'Yield (%)': Percentage of offers accepted by students.")
print("- 'Applicant/Enrolled Ratio': Ratio of applicants to enrolled students, indicating admission efficiency.")
print("- 'Average Entry Tariff': Average tariff for entry, with 'n/a' in 2017.")

print("\nNotable Trends and Insights:")
print("- Applications peaked in 2016 (42,335) and slightly declined in 2017 (41,000).")
print("- Offer rate decreased from 2015 to 2016 but rose in 2017, suggesting tighter selection or more competitive offers.")
print("- Enrolment remained relatively stable, with a slight increase in 2017.")
print("- Yield (%) declined after 2014, indicating fewer students accepting offers, possibly due to higher competition or changing preferences.")
print("- Applicant/Enrolled Ratio improved over time, showing greater efficiency in admissions.")
print("- Average Entry Tariff increased from 2013 to 2016, suggesting rising costs, though data is missing for 2017.")

Final Answer: Applications, Offer Rate, Enrols, Yield, Applicant/Enrolled Ratio, Average Entry Tariff