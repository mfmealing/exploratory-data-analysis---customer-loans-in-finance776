import yaml
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
import missingno as msno
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import seaborn as sns
from scipy import stats
import plotly.express as px


def load_yaml():
    with open('credentials.yaml', 'r') as f:
        credentials = yaml.safe_load(f)
    return credentials

class RDSDatabaseConnector:
    def __init__(self, creds):
        self.creds = creds

    def engine_init(self):
        engine = create_engine(f"postgresql+psycopg2://{self.creds['RDS_USER']}:{self.creds['RDS_PASSWORD']}@{self.creds['RDS_HOST']}:{self.creds['RDS_PORT']}/{self.creds['RDS_DATABASE']}")
        return engine
    
    def create_df(self, engine):
        dataframe = pd.read_sql_table('loan_payments', engine)
        return dataframe
    
    def df_to_csv(self, dataframe):
        dataframe.to_csv('loan_payments.csv')

def csv_to_df():
    loans_df = pd.read_csv('loan_payments.csv', index_col=0)
    return loans_df

class DataTransform:
    def __init__(self, df):
        self.df = df
    
    def map_columns(self):
        loan_status_mapping = {'Does not meet the credit policy. Status:Charged Off': 'Charged Off',
                               'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid'}
        self.df['loan_status'] = self.df['loan_status'].replace(loan_status_mapping)

    def update_cols(self):
        self.df = self.df.astype({"loan_amount":'float64', 
                                  "issue_date":'datetime64[ns]', 
                                  "earliest_credit_line":'datetime64[ns]',
                                  "last_payment_date":'datetime64[ns]',
                                  "next_payment_date":'datetime64[ns]',
                                  "last_credit_pull_date":'datetime64[ns]',
                                  "loan_status":'category', 
                                  "grade":'category', 
                                  "sub_grade":'category', 
                                  "home_ownership":'category',
                                  "verification_status":'category', 
                                  "payment_plan":'category', 
                                  "purpose":'category',
                                  "policy_code":'category',
                                  "application_type":'category'})
        return self.df

class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def data_types(self):
        print(self.df.dtypes)
    
    def stat_values(self):
        print(self.df.describe())
    
    def unique_count(self):
        print(self.df.nunique())

    def shape(self):
        print(self.df.shape)

    def null_count(self):
        null_df = self.df.isnull()
        print("Null count:")
        print(null_df.sum(), "\n")
        print("Null percentage count:")
        print((null_df.sum()/len(self.df))*100)

class DataFrameTransform:
    def __init__(self, df):
        self.df = df
    
    def drop_columns(self):
        new_df_cols = self.df.drop(labels=["mths_since_last_delinq","mths_since_last_record","next_payment_date","mths_since_last_major_derog"], axis=1)
        return new_df_cols

    def drop_rows(self, new_df):
        new_df.dropna(axis=0, subset=["last_payment_date", "last_credit_pull_date"], how="any", inplace=True)
        return new_df

    def impute_columns(self, new_df):
        new_df['funded_amount'].fillna((new_df['funded_amount'].mean()+loans_df['funded_amount'].median())/2, inplace=True)
        new_df['term'].fillna("36 months", inplace=True)
        new_df['int_rate'].fillna(method="ffill", inplace=True)
        new_df['employment_length'].fillna(method="ffill", inplace=True)
        new_df['collections_12_mths_ex_med'].fillna(0, inplace=True)
        return new_df
    
    def log_transform(self, col):
        log_col = col.map(lambda i: np.log(i) if i > 0 else 0)
        return log_col
    
    def box_cox(self, col):
        transform_data, lambda_value = stats.boxcox(col)
        return transform_data  
    
    def yeo_johnson(self, col):
        transform_data, lambda_value = stats.yeojohnson(col)
        return transform_data
    
    def remove_outliers(self, cols):
        Q1 = self.df[cols].quantile(0.25)
        Q3 = self.df[cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered = self.df[~((self.df[cols] < lower_bound) | (self.df[cols] > upper_bound)).any(axis=1)]
        return filtered

class Plotter:
    def __init__(self, df):
        self.df = df
    
    def matrix_plot(self):
        msno.matrix(self.df)

    def hist_plot(self, col):
        sns.histplot(col,label="Skewness: %.2f"%(col.skew()), bins=20)
        pyplot.legend()
        pyplot.show()

    def qq_plot(self, col):
        qq_plot = qqplot(col , scale=1 ,line='q', fit=True)
        pyplot.show()

    def box_plot(self,col):
        pyplot.figure()
        sns.boxplot(col, showfliers=True)
        pyplot.show()


if __name__ == "__main__":
    # Loads dataframe and converts to csv which is then saved
    loans = RDSDatabaseConnector(load_yaml())
    loans_engine = loans.engine_init()
    loans_dataframe = loans.create_df(loans_engine)
    loans.df_to_csv(loans_dataframe)

    # Opens csv file and converts columns to correct type
    pd.set_option('display.max_columns', None)
    old_loans_df = csv_to_df()
    loans_df = DataTransform(old_loans_df)
    loans_df.map_columns()
    loans_df = loans_df.update_cols()

    # Drops any columns that have a percentage of NaN's higher than 50% and imputes all other columns 
    # that have NaN's with either a combination of the mean and median, using the ffill function to fill 
    # in NaN's with the next closest value in the dataset or dropping the row entirely if the column has
    # a small number of NaN's that can't be imputed (e.g. a date)
    new_loans = DataFrameTransform(loans_df)
    dropped_cols_loans = new_loans.drop_columns()
    dropped_rows_loans = new_loans.drop_rows(dropped_cols_loans)
    updated_loans = new_loans.impute_columns(dropped_rows_loans)

    # Shows the null count and percentage null count before and after applying the above code
    #loans_info = DataFrameInfo(loans_df)
    #loans_info.null_count()
    #updated_loans_info = DataFrameInfo(updated_loans)
    #updated_loans_info.null_count()

    # Creates a matrix plot to show the distibution of nulls before and after the dataframetransform is applied
    # Also shows a histogram plot of one of the columns, showing the skew of the data in the top right
    #loans_plot = Plotter(loans_df)
    #loans_plot.matrix_plot()
    updated_loans_plot = Plotter(updated_loans)
    #updated_loans_plot.matrix_plot()
    #updated_loans_plot.hist_plot(updated_loans['collections_12_mths_ex_med'])

    # Changes skew on any columns that have a skew value above 2 or below -2
    # Uses either a log, box cox or yeo johnson transform, depending on the data
    # The transform used is simply the one that give the best reduction (closest to 0) in skew
    # Also shows another histogram of the same column as above but with the skew improved, again showing the changed skew
    data_transform = DataFrameTransform(updated_loans)
    updated_loans['annual_inc'] = data_transform.box_cox(updated_loans['annual_inc'])
    updated_loans['delinq_2yrs'] = data_transform.yeo_johnson(updated_loans['delinq_2yrs'])
    updated_loans['inq_last_6mths'] = data_transform.yeo_johnson(updated_loans['inq_last_6mths'])
    updated_loans['out_prncp'] = data_transform.yeo_johnson(updated_loans['out_prncp'])
    updated_loans['out_prncp_inv'] = data_transform.yeo_johnson(updated_loans['out_prncp_inv'])
    updated_loans['total_rec_int'] = data_transform.box_cox(updated_loans['total_rec_int'])
    updated_loans['total_rec_late_fee'] = data_transform.yeo_johnson(updated_loans['total_rec_late_fee'])
    updated_loans['recoveries'] = data_transform.yeo_johnson(updated_loans['recoveries'])
    updated_loans['collection_recovery_fee'] = data_transform.yeo_johnson(updated_loans['collection_recovery_fee'])
    updated_loans['collections_12_mths_ex_med'] = data_transform.yeo_johnson(updated_loans['collections_12_mths_ex_med'])
    #updated_loans_plot.hist_plot(updated_loans['collections_12_mths_ex_med'])    

    #updated_loans_plot.hist_plot(updated_loans['annual_inc'])
    #updated_loans_plot.box_plot(updated_loans['annual_inc'])
    updated_loans_new = data_transform.remove_outliers(['int_rate','instalment','annual_inc', 'open_accounts',
                                                     'total_accounts','total_payment', 'total_payment_inv',
                                                     'total_rec_prncp', 'total_rec_int', 'last_payment_amount'])
    #updated_loans_plot.hist_plot(updated_loans1['annual_inc'])
    #updated_loans_plot.box_plot(updated_loans1['annual_inc'])

    for column in updated_loans_new.columns:
        if updated_loans_new[column].dtype == 'object' or updated_loans_new[column].dtype == 'category':
            updated_loans_new.loc[:, column] = updated_loans_new[column].astype('category').cat.codes
        elif updated_loans_new[column].dtype == 'datetime64[ns]':
            updated_loans_new.loc[:,column] = pd.to_datetime(updated_loans_new[column])

    # Correlation threshold set to 0.9
    correlation = updated_loans_new.corr()
    pyplot.figure(figsize = (20,20))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    pyplot.show()
    updated_loans_new.drop(labels=["id", "funded_amount_inv", "instalment", "grade", "out_prncp_inv", 
                                   "total_payment_inv", "total_rec_prncp", "collection_recovery_fee"], axis=1, inplace=True)
    correlation_new = updated_loans_new.corr()
    pyplot.figure(figsize = (20,20))
    sns.heatmap(correlation_new, annot=True, fmt=".2f", cmap="coolwarm")
    pyplot.show()