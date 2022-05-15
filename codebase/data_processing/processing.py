# %%

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_columns = 50

# %% Functions used to extract data

print(f"Processing from: {sys.path[0]}")


# %%
if 'jli' in sys.path[0]:
    PATH_TO_DATA = '/Users/jli/Repos/m5-forecasting-accuracy/'
    sys.path.insert(0, '/Users/jli/Repos/m5-forecasting-accuracy/BayesianSales/data_processing')
    sys.path.insert(0, '/Users/jli/Repos/m5-forecasting-accuracy/BayesianSales/main_code')
else:
    PATH_TO_DATA = '/Users/ryznerf/Documents/0_MIT/Spring_2022/0_Projects/1_Bayesian_Modelling_project/BayesianSales/data_processing/data/'
# %%

class M5Data():

    queries = {
        "1item1store": "item_id == 'FOODS_3_090'",
        "1item1state": "item_id == 'HOBBIES_1_348' and state_id == 'CA'",

    }

    def __init__(self, path_to_data=PATH_TO_DATA, day_start=790, day_end=1520, verbose=False):
        """
        Reads in data from file path specified
        day_start and day_end specified
        Save attributes sales, prices, calendar etc
        extracts number if needed

        staticmedthod reduce_mem_usage

        getSales(query specified)
        use .reshape, .merge_calendar, .merge_prices
        """
        self.input_dir = path_to_data
        self.day_start = day_start
        self.day_end = day_end
        self.verbose = verbose

        (
            sales,
            prices,
            calendar
        ) = self.read_data()

        self.sales = sales
        self.prices = prices
        self.calendar = calendar
        self.calendar["d"] = self.extract_num(calendar["d"])

        self.item_ids = self.sales.item_id.unique()

        self.query = None
        self.subset = None

    @staticmethod
    def reduce_mem_usage(df, verbose=False):
        start_mem = df.memory_usage().sum() / 1024 ** 2
        int_columns = df.select_dtypes(include=["int"]).columns
        float_columns = df.select_dtypes(include=["float"]).columns

        for col in int_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in float_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

    def read_data(self):

        if self.verbose:
            print("Reading files...")

        calendar = pd.read_csv(
            f"{self.input_dir}/calendar.csv").pipe(self.reduce_mem_usage)
        prices = pd.read_csv(
            f"{self.input_dir}/sell_prices.csv").pipe(self.reduce_mem_usage)

        cat_cols = ['id', 'item_id', 'dept_id',
                    'store_id', 'cat_id', 'state_id']
        numcols = [f"d_{day}" for day in range(self.day_start, self.day_end+1)]
        dtype = {numcol: "float32" for numcol in numcols}
        dtype.update({col: "category" for col in cat_cols if col != "id"})

        sales = pd.read_csv(f"{self.input_dir}/sales_train_evaluation.csv",
                            usecols=cat_cols + numcols, dtype=dtype
                            ).pipe(
            self.reduce_mem_usage
        )

        # calendar shape: (1969, 14)
        # sell_prices shape: (6841121, 4)
        # sales_train_val shape: (30490, 1919)
        # submission shape: (60980, 29)

        return sales, prices, calendar

    @staticmethod
    def extract_num(ser):
        return ser.str.extract(r"(\d+)").astype(np.int16)

    def reshape_sales(self, sales_sub):

        # melt sales data
        id_columns = ["id", "item_id", "dept_id",
                      "cat_id", "store_id", "state_id"]

        sales_sub = sales_sub.melt(
            id_vars=id_columns, var_name="d", value_name="demand",).pipe(self.reduce_mem_usage)
        sales_sub["d"] = self.extract_num(sales_sub["d"])
        sales_sub = sales_sub[(sales_sub["d"] >= self.day_start) &
                              (sales_sub["d"] <= self.day_end)]
        sales_sub = sales_sub.drop(columns=['id'])

        return sales_sub

    @staticmethod
    def merge_calendar(data, calendar):
        calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
        return data.merge(calendar, how="left", on="d")

    @staticmethod
    def merge_prices(data, prices):
        return data.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])

    def make_subset(self, query=None):
        if query is None:
            query = self.queries['1item1store']
        elif query in self.queries.keys():
            query = self.queries[query]

        # Check if requested query is the same as current
        if query == self.query:
            print("Same query, subset not changed")
            return None

        print(f"Using query: {query}")
        sales_sub = self.sales.query(query)

        sales_sub = self.reshape_sales(sales_sub)

        # merge with calendar and prices
        data = self.merge_calendar(sales_sub, self.calendar)
        data = self.merge_prices(data, self.prices).pipe(self.reduce_mem_usage)

        data.index = pd.to_datetime(data['date'])

        self.subset = data
        self.query = query


def plot_sales_price(dfw, title='Sales and Price over Time'):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax2 = ax.twinx()

    ax.plot(dfw.demand)
    ax2.plot(dfw.sell_price, c='y')

    ax.set_ylim(bottom=0, top=dfw.demand.max()*1.1)
    ax2.set_ylim(bottom=0, top=dfw.sell_price.max()*1.05)
    ax.set(title=title)

    return fig


# %%

if __name__ == "__main__":

    m5data = M5Data()
    m5data.make_subset('1item1state')
    print(m5data.subset.head())
