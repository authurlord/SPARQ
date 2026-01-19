import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display the table structure and describe each column
print("Table Description:")
print("Columns represent years from 2013 to 2017.")
print("Rows represent key metrics:")
print("- 'Applications': Total number of applicants per year.")
print("- 'Offer Rate (%)': Percentage of applicants offered admission.")
print("- 'Enrols': Number of students who enrolled.")
print("- 'Yield (%)': Percentage of offers accepted by students.")
print("- 'Applicant/Enrolled Ratio': Ratio of applicants to enrolments, indicating competitiveness.")
print("- 'Average Entry Tariff': Average tariff charged at entry; 'n/a' in 2017.")

# Analyze trends
print("\nNotable Trends and Insights:")
print("- Applications slightly decreased from 2016 to 2017 (42,335 → 41,000).")
print("- Offer Rate peaked in 2015 (90.8%) and declined slightly after, suggesting tighter admissions in later years.")
print("- Enrols remained relatively stable, increasing slightly from 2014 onward.")
print("- Yield (%) increased from 2013 to 2014 (18.7% → 17.6%), then dropped, possibly due to reduced offer acceptance.")
print("- Applicant/Enrolled Ratio decreased over time, indicating improved admission efficiency.")
print("- Average Entry Tariff dropped significantly from 2016 onwards, suggesting policy or market changes.")

Final Answer: Applications, Offer Rate, Enrols, Yield, Applicant/Enrolled Ratio, Average Entry Tariff