import pandas as pd
import streamlit as st

class Pandas_Utilities():
    def load_data(self):
        """Loads data from a CSV or Excel file, handling various formats and potential errors."""
        try:
            if self.filepath.endswith('.csv'):
                df = pd.read_csv(self.filepath, encoding='utf-8', on_bad_lines='skip', engine='python')
            elif self.filepath.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(self.filepath, engine='openpyxl')  # Requires openpyxl package
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return pd.DataFrame()  # Return an empty DataFrame if file not found
        except pd.errors.EmptyDataError:
            print(f"Error: The file at {self.filepath} is empty.")
            return pd.DataFrame()
        except pd.errors.ParserError:
            print(f"Error: Could not parse the file at {self.filepath}. Check the file format.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()

    def clean_data(self):
        """Performs basic data cleaning operations."""
        if self.df.empty:
            print("Warning: DataFrame is empty. Skipping data cleaning.")
            return

        # Example cleaning operations (customize as needed)
        self.df = self.df.dropna(how='all')  # Remove rows with all NaN values
        self.df = self.df.replace({r'[^\x00-\x7F]+': ''}, regex=True) # remove non-ascii characters
        self.df = self.df.rename(columns=lambda x: x.strip()) # remove leading/trailing whitespace from column names

    def analyze_data(self):
        """Performs data analysis and returns key insights."""
        if self.df.empty:
            print("Warning: DataFrame is empty. Skipping data analysis.")

    def streamlit_file_picker(self):

        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip', engine='python')
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    self.df = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    st.error("Unsupported file format. Please provide a CSV or Excel file.")
                    return
                st.write(self.df)
                self.clean_data()
                self.analyze_data()
                return self.df

            except Exception as e:
                st.error(f"An error occurred: {e}")
                return
