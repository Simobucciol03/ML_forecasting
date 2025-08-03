"""
MySQL Database Connection Script

This script establishes a connection to a MySQL database using the `mysql-connector-python` library. 

Functions:
- `create_server_connection(host_name, user_name, user_password, db_name)`: 
    Establishes and returns a connection to a MySQL server using the specified parameters.
    - Parameters:
        - `host_name` (str): The hostname or IP address of the MySQL server.
        - `user_name` (str): The username for authenticating with the MySQL server.
        - `user_password` (str): The password for the specified username.
        - `db_name` (str): The name of the database to connect to.

    - Returns:
        - A connection object if the connection is successful.
        - `None` if an error occurs during the connection attempt.

Usage:
1. Modify the `database_name`, `host_name`, `user_name`, and `user_password` variables to match your MySQL server and database credentials.
2. Call the `create_server_connection` function with these parameters to establish a connection.

Dependencies:
- `mysql-connector-python`: Ensure this library is installed in your Python environment. You can install it using pip:
    `pip install mysql-connector-python`
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import yfinance as yf
import sqlalchemy
import os


class DataProvider:
    def create_server_connection(self, host_name, user_name, user_password, db_name):
        connection = None
        try:
            connection = mysql.connector.connect(
                host=host_name,
                port=3306,
                user=user_name,
                password=user_password,
                database=db_name
            )
            print("MySQL Database connection successful, now you can create your magic ideas!")
        except Error as err:
            print(f"Error: '{err}'")

        return connection

    def getdata(self, tickers):
        """
        Fetch stock data for given tickers with capital letter column names preserved.
        :param tickers: List of ticker symbols
        :return: Dictionary of DataFrames with stock data, keyed by ticker symbol
        """
        data_dict = {}
        for ticker in tickers:
            ticker_encoded = ticker.replace('^', '%5E').replace('=', '%3D')
            try:
                # Fetch stock data
                data = yf.download(ticker_encoded, interval="1wk")
                if not data.empty:
                    # Reset index but keep Date as a column
                    data = data.reset_index(level=0)  # This keeps 'Date' as a column
                    
                    # Set Date column as the index again
                    data = data.set_index('Date')  # This restores DatetimeIndex
                    
                    # Flatten the column structure
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
                    
                    # Ensure first letter is capital
                    data.columns = [col.capitalize() for col in data.columns]
                    data_dict[ticker] = data
                    print(f"Successfully downloaded data for {ticker}")
                else:
                    print(f"No data found for ticker: {ticker}")
                
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        return data_dict 


    # Function to create MySQL engine
    def create_engine_mysql(self, user, password, host, port, database):
        connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        engine = sqlalchemy.create_engine(connection_string)
        return engine

    def TOSQL(self, data_dict, engine):
        """
        Import stock data into MySQL database with capital letter column names.

        :param data_dict: Dictionary of DataFrames with stock data, keyed by ticker symbol
        :param engine: SQLAlchemy engine object for MySQL database
        """
        try:
            # Write each dataframe to MySQL database
            for ticker, df in data_dict.items():
                safe_ticker = ticker.replace('^', '_index_').replace('=', '_futures_').replace('.', '_')
                # Store with capitalized column names
                df.to_sql(safe_ticker, engine, if_exists='replace', index=True)
                print(f"Data for {ticker} imported successfully to table '{safe_ticker}'.")
        except Exception as e:
            print(f"Error importing data: {e}")

    def read_and_convert_to_lowercase(self, engine, table_name):
        """
        Read data from MySQL and convert column names to lowercase.
        
        :param engine: SQLAlchemy engine object
        :param table_name: Name of the table to read from
        :return: DataFrame with lowercase column names
        """
        try:
            # Read data from the database
            query = f"SELECT * FROM `{table_name}`"
            df = pd.read_sql(query, engine)
            
            # Convert column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            return df
        except Exception as e:
            print(f"Error reading data: {e}")
            return None

    def main(self):
        # Database credentials
        database_name = "xxxx"  # Replace with your database name
        user_name = "xxxx"  # Replace with your MySQL username
        password = "xxxxxxx"  # Replace with your MySQL password
        
        # MySQL server details
        host_name = "xxxxx"  # Replace with your MySQL server host
        # Default MySQL port
        port = xxxx  # Replace with your MySQL server port
        
        # Create the database connection
        connection = self.create_server_connection(host_name, user_name, password, database_name)
        
        if connection:
            try:
                mysql_engine = self.create_engine_mysql(user_name, password, host_name, port, database_name)
                
                # Ask user whether they want to import data to the database
                user_input = input("Do you want download and import data to the database? (y/n): ")
                
                if user_input.lower() == "y":
                    tickers_input = input("Enter the tickers separated by space: ")
                    tickers = tickers_input.strip().split()
                    
                    if not tickers:
                        print("No tickers entered. Exiting.")
                        return

                    stock_data_dict = self.getdata(tickers)
                    
                    if not stock_data_dict:
                        print("No data retrieved for any ticker. Exiting.")
                        return
                    
                    self.TOSQL(stock_data_dict, mysql_engine)
                
                else:
                    print("Data import skipped.")

                query_data = input("Do you want import data from the database? (y/n): ")
                
                if query_data.lower() == "y":
                    inspector = sqlalchemy.inspect(mysql_engine)
                    available_tables = inspector.get_table_names()
                    
                    if not available_tables:
                        print("No tables found in the database.")
                        return
                    
                    print("Available tables:", ", ".join(available_tables))
                    
                    data_tables = input("Enter the table names you want to see (separated by space): ").split()
                    
                    for table in data_tables:
                        if table in available_tables:
                            try:
                                # Fetch all data from the selected table
                                query = f"SELECT * FROM `{table}`"
                                df_original = pd.read_sql(query, mysql_engine)
                                
                                print(f"All columns in table {table}: {df_original.columns.tolist()}")
                                
                                # Identify stock-related columns (case insensitive)
                                stock_columns = {'close': None, 'high': None, 'low': None, 
                                                'open': None, 'volume': None}
                                
                                for col in df_original.columns:
                                    col_lower = col.lower()
                                    if col_lower in stock_columns:
                                        stock_columns[col_lower] = col
                                
                                # Look for date column
                                date_column = None
                                date_columns = [col for col in df_original.columns if 
                                                'date' in col.lower() or 'time' in col.lower()]
                                
                                if date_columns:
                                    date_column = date_columns[0]
                                    print(f"Detected date column: {date_column}")
                                else:
                                    for col in df_original.columns:
                                        if col in stock_columns.values():
                                            continue  
                                        try:
                                            pd.to_datetime(df_original[col])
                                            date_column = col
                                            print(f"Auto-detected date column: {col}")
                                            break
                                        except:
                                            continue
                                
                                # Convert columns to lowercase
                                df_lowercase = df_original.copy()
                                df_lowercase.columns = [col.lower() for col in df_lowercase.columns]
                                
                                expected_columns = ['close', 'high', 'low', 'open', 'volume']
                                for expected_col in expected_columns:
                                    if expected_col not in df_lowercase.columns:
                                        orig_col = stock_columns.get(expected_col)
                                        if orig_col and orig_col in df_original.columns:
                                            df_lowercase[expected_col] = df_original[orig_col]
                                            print(f"Added missing '{expected_col}' column")
                                
                                print(f"\n🔹 Final dataframe columns: {df_lowercase.columns.tolist()}")
                                print(f"📅 Final dataframe index type: {type(df_lowercase.index)}")
                                print(f"📊 Number of rows: {len(df_lowercase)}")
                                
                            except Exception as e:
                                print(f"Error querying table '{table}': {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"Table '{table}' not found in database.")
                else:
                    print("Data query skipped.")
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if connection.is_connected():
                    connection.close()
                    print("MySQL connection is closed")
        
        return df_lowercase, tickers 

if __name__ == "__main__":
    provider = DataProvider()
    df_lowercase, tickers = provider.main()

    if df_lowercase is not None and not df_lowercase.empty:
        print("Data processing completed successfully.")
        print(f"Processed tickers: {tickers}")

        # Trova il desktop path in modo dinamico (funziona su Windows, macOS, Linux)
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

        # Imposta nome file CSV
        if tickers and len(tickers) > 0:
            filename = f"{tickers[0]}_data.csv"
        else:
            filename = "stock_data.csv"

        csv_file_path = os.path.join(desktop_path, filename)

        # Salva DataFrame in CSV
        try:
            df_lowercase.to_csv(csv_file_path, index=False)
            print(f"✅ Data saved to {csv_file_path}")
        except Exception as e:
            print(f"❌ Failed to save data to CSV: {e}")
    else:
        print("⚠️ No data processed or data is empty.")
        print("Please check the logs for any errors.")
else:
    print("This script is intended to be run as a standalone program.")
    print("Please run it directly to establish a MySQL connection and process stock data.")
    exit(1)
