# Exploratory Data Analysis - Customer Loans in Finance

## Project description
In this scenario, there is a large financial institution, where managing loans is a critical component of business operations.
To ensure informed decisions are made about loan approvals and risk is efficiently managed, the task is to gain a comprehensive understanding of the loan portfolio data.
The task is to perform exploratory data analysis on the loan portfolio, using various statistical and data visualisation techniques to uncover patterns, relationships, and anomalies in the loan data.
This information will enable the business to make more informed decisions about loan approvals, pricing, and risk management.
By conducting exploratory data analysis on the loan data, the aim is to gain a deeper understanding of the risk and return associated with the business' loans.
Ultimately, the goal is to improve the performance and profitability of the loan portfolio.  
  
Currently the code will create a dataframe and .csv file from an online source containing loan payments.  
From this .csv file, the columns are updated to more appropriate data types and then NaNs are removed by either dropping entire columns (if the percentage of NaNs is too high) or 
imputing the NaN values with appropriate values. A small number of rows are also removed where the NaNs cannot be imputed (e.g. times). The number and percentage of NaNs before and 
after is then printed alongside a matrix plot before and after the NaNs are removed.  
The skew of the data is then adjusted, with any columns that have a skew above 2 or below -2 transformed to reduce the skew as much as possible. A histogram of one column before 
and after changing the skew is shown for comparison.  
Outliers are then removed using the interquartile range (IQR). A histogram and box plot of a column before and after the outliers is removed to show a comparison.  
Finally, the correlation is compared using a heatmap and any columns that have a correlation above 0.9 are removed, with some columns remaining to make the data more understandable
for later analysis (e.g. id is removed, while member_id is kept).  
  
The data is then analysed, looking at both charged off loans and late loans and hwo much of a loss they are or could be to the company. An investigation is also made to see if any
of the other columns have a link between these types of loans, and if people with certain information are more likely to defualt on their loans.  

## Installation and usage
The python file db_utils.py should be run first to download the database as a csv file to use within the EDA.ipynb file. The EDA.ipynb file works best when running from top to bottom but any of the later analysis
can be run in any order.
The modules used the run these files are PyYAML (shown as yaml in the code), SQLAlchemy, psycopg2-binary (shown as psycopg2 in the code), Pandas, numpy, missingno, statsmodels, matplotlib, seaborn, scipy and textwrap. 
These can all be installed using pip.

## File structure
db_utils.py currently contains a class that creates an engine using a yaml file (not included for security), using one method in the class. A .csv file is then created using this engine.  
EDA. ipynb then takes this file and converts it to a dataframe to use during analysis, which is also contained within this file.  
  
loan_payments.csv contains the newly created database in a csv format. The table contains information about loan payments, with the following headers:  
- **id**: unique id of the loan.
- **member_id**: id of the member to took out the loan.
- **loan_amount**: amount of loan the applicant received.
- **funded_amount**: The total amount committed to the loan at the point in time.
- **funded_amount_inv**: The total amount committed by investors for that loan at that point in time. 
- **term**: The number of monthly payments for the loan.
- **int_rate**: Interest rate on the loan.
- **instalment**: The monthly payment owned by the borrower.
- **grade**: LC assigned loan grade.
- **sub_grade**: LC assigned loan sub grade.
- **employment_length**: Employment length in years.
- **home_ownership**: The home ownership status provided by the borrower.
- **annual_inc**: The annual income of the borrower.
- **verification_status**: Indicates whether the borrowers income was verified by the LC or the income source was verified.
- **issue_date:** Issue date of the loan.
- **loan_status**: Current status of the loan.
- **payment_plan**: Indicates if a payment plan is in place for the loan. Indication borrower is struggling to pay.
- **purpose**: A category provided by the borrower for the loan request.
- **dti**: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowerâ€™s self-reported monthly income.
- **delinq_2yr**: The number of 30+ days past-due payment in the borrower's credit file for the past 2 years.
- **earliest_credit_line**: The month the borrower's earliest reported credit line was opened.
- **inq_last_6mths**: The number of inquiries in past 6 months (excluding auto and mortgage inquiries).
- **mths_since_last_record**: The number of months since the last public record.
- **open_accounts**: The number of open credit lines in the borrower's credit file.
- **total_accounts**: The total number of credit lines currently in the borrower's credit file.
- **out_prncp**: Remaining outstanding principal for total amount funded.
- **out_prncp_inv**: Remaining outstanding principal for portion of total amount funded by investors.
- **total_payment**: Payments received to date for total amount funded.
- **total_rec_int**: Interest received to date.
- **total_rec_late_fee**: Late fees received to date.
- **recoveries**: Post charge off gross recovery.
- **collection_recovery_fee**: Post charge off collection fee.
- **last_payment_date**: Last month payment was received.
- **last_payment_amount**: Last total payment amount received.
- **next_payment_date**: Next scheduled payment date.
- **last_credit_pull_date**: The most recent month LC pulled credit for this loan.
- **collections_12_mths_ex_med**: Number of collections in 12 months excluding medical collections.
- **mths_since_last_major_derog**: Months since most recent 90-day or worse rating.
- **policy_code**: If publicly available, code=1 and if new products not publicly available, code=2.
- **application_type**: Indicates whether the loan is an individual application or a joint application with two co-borrowers.

## License information
Copyright (c) <2023>, <mfmealing>  
All rights reserved.  
  
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
