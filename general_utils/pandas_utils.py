import glob
import os.path
import threading
import time
from typing import Set, Any, List, Union
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Constants
NEW_CSV_FILE_EXTRA_STR = "_no_dups"
LP_COL_NAME = "PLATES"
COLOR_COL = "COLOR"
COLOR_TYPE_COL = "COLOR_TYPE"
DESC1_COL = "DESC1"
DESC2_COL = "DESC2"
COL_TO_GET = [LP_COL_NAME, COLOR_COL, COLOR_TYPE_COL, DESC1_COL, DESC2_COL]


class PandasUtils:
    @staticmethod
    def read_csv_with_progress_bar(csv_path: str, read_cols: List[str], chunk_size=1000) -> pd.DataFrame:
        chunks = []
        print(f"Start loading csv file: {csv_path}")
        start_time = time.time()

        # Use tqdm to display a progress bar
        for chunk in tqdm(pd.read_csv(csv_path, usecols=read_cols, chunksize=chunk_size)):
            # Add each chunk to the list
            chunks.append(chunk)
        # Concatenate all the chunks into a single dataframe
        df = pd.concat(chunks)

        load_time = time.time() - start_time
        print(f"Finished loading csv file: {csv_path}. Load time: {load_time}")
        return df

    @staticmethod
    def load_csv(csv_path: str, read_cols: List[str]) -> pd.DataFrame:
        print(f"Start loading csv file: {csv_path}")
        start_time = time.time()
        new_csv = pd.read_csv(csv_path, usecols=read_cols)
        load_time = time.time() - start_time
        print(f"Finished loading csv file: {csv_path}. Load time: {load_time}")
        return new_csv

    @staticmethod
    def load_csv_remove_dups_and_save(csv_path: str, read_cols: List[str],
                                      remove_dup_by_col: Union[str, List[str]]) -> None:
        df = pd.read_csv(csv_path, usecols=read_cols)
        df = PandasUtils.remove_duplicates(df, remove_dup_by_col=remove_dup_by_col)
        new_csv_file_path = os.path.splitext(csv_path)[0] + NEW_CSV_FILE_EXTRA_STR + ".csv"
        df.to_csv(new_csv_file_path, index=False)

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, remove_dup_by_col: Union[str, List[str]]) -> pd.DataFrame:
        """
        Removes lines which have the same value in col "col_name"
        """
        if not isinstance(remove_dup_by_col, list):
            remove_dup_by_col = [remove_dup_by_col]
        return df.drop_duplicates(subset=remove_dup_by_col)

    @staticmethod
    def _load_csv_to_list(df_list: List[pd.DataFrame], csv_path: str, read_cols: List[str]) -> None:
        new_csv = PandasUtils.load_csv(csv_path, read_cols)
        df_list.append(new_csv)


    @staticmethod
    def load_csv_dir(dir_path: str = '.', read_cols: List[str] = None) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
        print(f"Starting to load csv dir: {dir_path} with {len(csv_files)} csv files")
        start_time = time.time()
        df_list = []
        threads = []
        for csv_path in csv_files:
            t = threading.Thread(target=PandasUtils._load_csv_to_list, args=(df_list, csv_path, read_cols))
            threads.append(t)

        # Start the threads
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Concatenate the dataframes into a single dataframe
        df = pd.concat(df_list)
        load_time = time.time() - start_time
        print(f"Finished loading all csv files. total time: {load_time}")
        return df

    @staticmethod
    def find_line_info(df: pd.DataFrame, col_name: str, query: Any) -> pd.DataFrame:
        col = df.loc[df[col_name] == query]
        return col

    @staticmethod
    def get_col_val_as_set(df: pd.DataFrame, col_name: str) -> Set:
        return set(df[col_name].unique())

    @staticmethod
    def plot_col_statistics(df: pd.DataFrame, col_name: str, plot_type="pie") -> None:
        unique_val_count = df[col_name].value_counts()
        plt.figure()
        unique_val_count.plot(kind=plot_type, title=f"{plot_type} chart of {col_name} column")
        plt.show(block=False)
        plt.pause(0.1)


def panda_utils_demo() -> None:
    df = PandasUtils.load_csv_dir(".", read_cols=COL_TO_GET)
    print(f"Total num of rows: {df.shape[0]}")
    df = PandasUtils.remove_duplicates(df, LP_COL_NAME)
    print(f"Totoal num of rows after removing {LP_COL_NAME} duplicates: {df.shape[0]}")
    lp = "A0983"
    print(f"lp: {lp}, info: \n{PandasUtils.find_line_info(df, col_name=LP_COL_NAME, query=lp)}")
    # cols_to_analyze = [COLOR_COL, COLOR_TYPE_COL, DESC1_COL, DESC2_COL]
    cols_to_analyze = []
    for col_name in cols_to_analyze:
        set_of_possible_values_in_col = PandasUtils.get_col_val_as_set(df, col_name)
        print(f"Col {col_name} possible #values: {len(set_of_possible_values_in_col)}. "
              f"found values: {set_of_possible_values_in_col}")
        PandasUtils.plot_col_statistics(df, col_name, plot_type="pie")
        PandasUtils.plot_col_statistics(df, col_name, plot_type="bar")
    # print(df)


if __name__ == '__main__':
    # PandasUtils.read_csv_with_progress_bar("data.csv", read_cols=COL_TO_GET, chunk_size=2)
    panda_utils_demo()
    # input("asfasf")
