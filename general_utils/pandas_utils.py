from typing import Set
import matplotlib.pyplot as plt
import pandas as pd

# Constants
LP_COL_NAME = "PLATES"
COL_TO_GET = ["PLATES", "COLOR", "COLOR_TYPE", "DESC1", "DESC2"]
COLOR_COL = "COLOR"


def find_lp_info(df: pd.DataFrame, lp: str) -> pd.DataFrame:
    col = df.loc[df[LP_COL_NAME] == lp]
    return col


def get_col_val_as_set(df: pd.DataFrame, col_name: str, plot_type="pie") -> Set:
    return set(df[col_name].unique())


def plot_col_statistics(df: pd.DataFrame, col_name: str, plot_type="pie") -> None:
    unique_val_count = df[col_name].value_counts()
    plt.figure()
    unique_val_count.plot(kind=plot_type, title=f"{plot_type} chart of {col_name} column")
    plt.show(block=False)
    plt.pause(0.01)


if __name__ == '__main__':
    df = pd.read_csv("data.csv", usecols=COL_TO_GET)
    print(f"Total num of rows: {df.shape[0]}")
    lp = "A0983"
    print(f"lp: {lp}, info: {find_lp_info(df, lp)}")
    print(f"Possible colors: {get_col_val_as_set(df, COLOR_COL)}")
    plot_col_statistics(df, COLOR_COL, plot_type="pie")
    plot_col_statistics(df, COLOR_COL, plot_type="bar")
    print(df)
    input("asfasf")
