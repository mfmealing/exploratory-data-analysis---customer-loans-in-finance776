import yaml
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
from IPython.display import display
import numpy as np
import missingno as msno
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot


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
        df_rows = new_df.dropna(axis=0, subset="last_payment_date")
        new_df_rows = df_rows.dropna(axis=0, subset="last_credit_pull_date")
        return new_df_rows


    def impute_columns(self, new_df):
        new_df['funded_amount'].fillna((new_df['funded_amount'].mean()+loans_df['funded_amount'].median())/2, inplace=True)
        new_df['term'].fillna("36 months", inplace=True)
        new_df['int_rate'].fillna(method="ffill", inplace=True)
        new_df['employment_length'].fillna(method="ffill", inplace=True)
        new_df['collections_12_mths_ex_med'].fillna(0, inplace=True)
        return new_df

class Plotter:
    def __init__(self, df):
        self.df = df
    
    def matrix_plot(self):
        msno.matrix(self.df)


if __name__ == "__main__":
    loans = RDSDatabaseConnector(load_yaml())
    loans_engine = loans.engine_init()
    loans_dataframe = loans.create_df(loans_engine)
    loans.df_to_csv(loans_dataframe)

    pd.set_option('display.max_columns', None)
    old_loans_df = csv_to_df()
    loans_df = DataTransform(old_loans_df)
    loans_df.map_columns()
    loans_df = loans_df.update_cols()

    new_loans = DataFrameTransform(loans_df)
    dropped_cols_loans = new_loans.drop_columns()
    dropped_rows_loans = new_loans.drop_rows(dropped_cols_loans)
    no_nan_loans = new_loans.impute_columns(dropped_rows_loans)

    loans_info = DataFrameInfo(loans_df)
    loans_info.null_count()
    no_nan_info = DataFrameInfo(no_nan_loans)
    no_nan_info.null_count()

    loans_plot = Plotter(loans_df)
    loans_plot.matrix_plot()
    no_nan_plot = Plotter(no_nan_loans)
    no_nan_plot.matrix_plot()