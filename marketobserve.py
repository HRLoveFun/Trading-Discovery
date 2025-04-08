import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import datetime as dt
from matplotlib.ticker import PercentFormatter



def BullBearPlot(data, time_window):
    """
    input:
        data: pd.Series as price
        time_window: a list of strings or tuples
            string with format "number"+ time_unit, where time_unit is in [W,M,Q,Y] : dates in most recent time_window.int  time_window.time_unit.
            a tuple with format ("YYYYMMDD1", "YYYYMMDD2"): dates between "YYYYMMDD1"and "YYYYMMDD2"
    output: a line chart with n subcharts, n by 1, n is the length of  time_window.

    Example:
    >>> data = pd.Series([100, 110, 120, 105, 115], index=pd.date_range('20230101', periods=5))
    >>> time_window = ['1W', ('20230101', '20230103')]
    >>> BullBearPlot(data, time_window)
    """

    if not isinstance(time_window, (list, tuple)):
        raise ValueError("time_window must be a list or a tuple.")
    if not isinstance(data, pd.Series):
        raise ValueError("data must be a pandas Series.")

    n = len(time_window)
    fig = make_subplots(rows=n, cols=1, subplot_titles=[f"Plot {i + 1}" for i in range(n)])

    df = pd.DataFrame(data)
    df.columns = ["Close"]
    df["CumMax"] = df["Close"].cummax()
    df["IsBull"] = (df["Close"] - df["CumMax"] * 0.8).apply(np.sign)
    df.index = pd.to_datetime(df.index)

    for i, time_window_element in enumerate(time_window):
        start_date, end_date = parse_time_window(time_window_element, df.index[-1])
        # 处理不同类型的 time_window
        # In BullBearPlot function
        # Bug fix: change 'tw' to 'time_window_element'
        if isinstance(time_window_element, str):
            time_unit = time_window_element[-1]
            num = int(time_window_element[:-1])
            if time_unit == 'W':
                offset = pd.DateOffset(weeks=num)
            elif time_unit == 'M':
                offset = pd.DateOffset(months=num)
            elif time_unit == 'Q':
                offset = pd.DateOffset(months=3*num)
            elif time_unit == 'Y':
                offset = pd.DateOffset(years=num)
            else:
                raise ValueError("Invalid time unit. Allowed units are [W, M, Q, Y].")
            end_date = df.index[-1]
            start_date = end_date - offset
            selected_df = df[(df.index >= start_date) & (df.index <= end_date)]
            title_time_window = f"Recent {time_window_element}"
        # Bug fix: change 'tw' to 'time_window_element'
        elif isinstance(time_window_element, tuple) and len(time_window_element) == 2:
            start_date = pd.to_datetime(time_window_element[0], format="%Y%m%d")
            end_date = pd.to_datetime(time_window_element[1], format="%Y%m%d")
            selected_df = df[(df.index >= start_date) & (df.index <= end_date)]
            title_time_window = f"{time_window_element[0]}-{time_window_element[1]}"
    
        else:
            raise ValueError("Invalid time_window format.")

        # Loop through the data to plot segments with different colors
        for j in range(len(selected_df) - 1):
            x_vals = [selected_df.index[j], selected_df.index[j + 1]]
            y_vals = [selected_df["Close"].iloc[j], selected_df["Close"].iloc[j + 1]]
            color = 'red' if selected_df['IsBull'].iloc[j] < 0 else 'green'

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ), row=i + 1, col=1)

        fig.update_yaxes(type="log", row=i + 1, col=1)
        fig.update_xaxes(title_text="Date", row=i + 1, col=1)
        fig.update_yaxes(title_text="Price", row=i + 1, col=1)
        fig.layout.annotations[i].update(text=f"Bull and Bear Trend: {title_time_window}")

    fig.update_layout(height=400 * n)
    fig.show()


from matplotlib.ticker import PercentFormatter


def options_chain(symbol):

    tk = yf.Ticker(symbol)
    print(tk.info)
    # Expiration dates
    exps = tk.options
    # print(exps[1])
    # Get options for each expiration
    options = pd.DataFrame()
    options_list = []
    for e in exps:
        opt = tk.option_chain(e)
        calls_puts = pd.concat([opt.calls, opt.puts])
        calls_puts['expirationDate'] = e
        options_list.append(calls_puts)
    
    # Combine all options data
    options = pd.concat(options_list, ignore_index=True)

    # Bizarre error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + dt.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - dt.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options

def parse_time_window(window, latest_date):
    if isinstance(window, str):
        if window[-1] not in ['W', 'M', 'Q', 'Y']:
            raise ValueError("Invalid time window string format. Expected 'number+[W,M,Q,Y]'.")
        unit = window[-1]
        num = int(window[:-1])
        if unit == 'W':
            start_time = latest_date - pd.DateOffset(weeks=num)
        elif unit == 'M':
            start_time = latest_date - pd.DateOffset(months=num)
        elif unit == 'Q':
            start_time = latest_date - pd.DateOffset(months=3*num)
        elif unit == 'Y':
            start_time = latest_date - pd.DateOffset(years=num)
        # Return a tuple even for single date
        return start_time, latest_date
    elif isinstance(window, tuple):
        start_date = pd.to_datetime(window[0])
        end_date = pd.to_datetime(window[1])
        return start_date, end_date
    else:
        raise ValueError("Invalid time window format. Expected string or tuple.")

def ChangeDistPlot(data, time_windows=[1], frequencies=['W', 'M', 'Q', 'Y']):
    """  
    data: Daily frequency close price
    time_windows: a list of strings or tuples
        string with format "number"+ time_unit, where time_unit is in [W,M,Q,Y] : dates in most recent time_window.int  time_window.time_unit.
        a tuple with format ("YYYYMMDD1", "YYYYMMDD2"): dates between "YYYYMMDD1"and "YYYYMMDD2"
    frequencies: default=['W', 'M', 'Q', 'Y'], can change to any sublist of default
    return: a distribution chart matrix, n by 4, n is the length of  time_windows.
    """
    # 检查数据是否为空
    if data.empty:
        raise ValueError("Input data is empty.")
    
    # Convert index to datetime if it's not already
    data.index = pd.to_datetime(data.index)
    latest_date = data.index.max()

    num_freq = len(frequencies)
    num_windows = len(time_windows)

    # Adjust the layout if there is only one window or one frequency
    sub_figsize_width = 8
    if num_windows == 1 and num_freq == 1:
        fig, axes = plt.subplots(figsize=(sub_figsize_width, sub_figsize_width) )
        axes = np.array([[axes]])
    elif num_windows == 1:
        fig, axes = plt.subplots(1, num_freq, figsize=(sub_figsize_width * num_freq, sub_figsize_width))
        axes = axes.reshape(1, -1)
    elif num_freq == 1:
        fig, axes = plt.subplots(num_windows, 1, figsize=(sub_figsize_width, sub_figsize_width * num_windows))
        axes = axes.reshape(-1, 1)
    else:
        fig, axes = plt.subplots(num_windows, num_freq, figsize=(sub_figsize_width * num_freq, sub_figsize_width * num_windows))

    # 定义每个频率对应的 bin 宽度
    bin_widths = {'W': 0.01, 'M': 0.025, 'Q': 0.05, 'Y': 0.1}

    for i, window in enumerate(time_windows):
        if isinstance(window, str):
            # Parse the time unit and number
            if window[-1] not in ['W', 'M', 'Q', 'Y']:
                raise ValueError("Invalid time window string format. Expected 'number+[W,M,Q,Y]'.")
            unit = window[-1]
            num = int(window[:-1])
            if unit == 'W':
                start_time = latest_date - pd.DateOffset(weeks=num)
            elif unit == 'M':
                start_time = latest_date - pd.DateOffset(months=num)
            elif unit == 'Q':
                start_time = latest_date - pd.DateOffset(months=3*num)
            elif unit == 'Y':
                start_time = latest_date - pd.DateOffset(years=num)
            subset = data[start_time:].copy()
        elif isinstance(window, tuple):
            start_date = pd.to_datetime(window[0])
            end_date = pd.to_datetime(window[1])
            subset = data[start_date:end_date].copy()
        else:
            raise ValueError("Invalid time window format. Expected string or tuple.")

        time_span = (subset.index.max() - subset.index.min()).days

        for j, freq in enumerate(frequencies):
            if freq not in ['W', 'M', 'Q', 'Y']:
                raise ValueError(f"Invalid frequency: {freq}. Allowed frequencies are ['W', 'M', 'Q', 'Y'].")

            # Check if time_span is shorter than the frequency
            if (freq == 'W' and time_span < 7) or (freq == 'M' and time_span < 30) or (freq == 'Q' and time_span < 90) or (freq == 'Y' and time_span < 365):
                ax = axes[i, j]
                ax.set_title(f"{freq}-ly change in {window}")
                ax.set_xlabel("Change")
                ax.set_ylabel("Frequency")
                continue

            # 计算对数收益率
            log_returns = np.log(subset / subset.shift(1))
            resample_freq = freq if freq == "W" else freq + "E"
            change = log_returns.resample(resample_freq).sum()
            change = change.dropna()
            change = np.exp(change)-1
            change_max = np.max(change)
            change_min = np.min(change)
            # 获取对应的 bin 宽度
            bin_width = bin_widths[freq]
            # 计算 bin 的边界
            bins = np.arange(np.floor(change_min/bin_width)*bin_width-0.5*bin_width, np.ceil(change_max/bin_width)*bin_width+0.5*bin_width, bin_width)

            ax = axes[i, j]
            sns.histplot(change, kde=True, ax=ax, bins=bins, legend=False)
            ax.set_title(f"{freq}-ly change in {window}")
            # 设置 x 轴的数值单位为百分之一
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            # 设置 y 轴显示整数
            yticks = ax.get_yticks().astype(int)          
            ax.set_yticks(yticks)

            ax.set_xlabel("Change")
            ax.set_ylabel("Frequency")

            # Slice lastest values of pd.series change
            current_values = change.tail(4)
            current_values_lables = [f'{idx.date()}: {val.iloc[0]:.2%}' for idx, val in current_values.iterrows()]

            # show the values in the chart
            for k, label in enumerate(current_values_lables):
                ax.text(0.05, 0.9 - k * 0.05, label, transform=ax.transAxes, fontsize=12)

            # show the numbers of each bar
            count = len(change)

            for rect in ax.patches:
                height = rect.get_height()
                if height > 0:
                    change_freq = height/count
                    ax.text(rect.get_x() + rect.get_width()/2, height, f'{height}, {change_freq * 100:.0f}%', ha='center', va='bottom', fontsize=12)

            # show the expectation of change  in the chart, calculate by  middle point of each bar and the frequency
            # 计算每个柱子的中点和频率
            expectation = 0
            for rect in ax.patches:
                # 计算柱子的中点
                mid_point = rect.get_x() + rect.get_width() / 2
                # 获取柱子的高度
                height = rect.get_height()
                # 计算频率
                change_freq = height / count
                # print(mid_point, height, change_freq)
                # 累加期望值
                expectation += mid_point * change_freq
            # Determine the color based on the expectation value
            if expectation > 0:
                text_color = 'green'
            elif expectation < 0:
                text_color = 'red'
            else:
                text_color = 'black'
            
            deviation = np.std(change.values)

            # Add bold style to the text
            text_weight = 'bold'
            # 在图表中显示期望值
            ax.text(0.05, 0.95, f'Expectation: {expectation * 100:.2f}%', transform=ax.transAxes, fontsize=12, color=text_color, weight=text_weight)
            ax.text(0.05, 0.99, f'Deviation: {deviation * 100:.2f}%', transform=ax.transAxes, fontsize=12, weight=text_weight)

    plt.tight_layout()
    plt.show()

def yf_download(ticker, start_date, end_date, frequency='1d', progress=True, auto_adjust=False):
    """
    Download stock data from Yahoo Finance
    
    Parameters:
    - ticker: Stock symbol (e.g., "^HSI")
    - start_date: datetime.date or string in YYYY-MM-DD format
    - end_date: datetime.date or string in YYYY-MM-DD format
    - frequency: Data frequency ('1d', '1wk', '1mo')
    - progress: Show download progress bar
    - auto_adjust: Adjust all OHLC automatically
    
    Returns:
    - DataFrame with stock data
    """
    df = yf.download(
        ticker, 
        start=start_date, 
        end=end_date,
        interval=frequency,
        progress=progress, 
        auto_adjust=auto_adjust
    )
    df.columns = df.columns.droplevel(1)
    df.set_index(pd.DatetimeIndex(df.index), inplace=True)
    # Reorder columns as specified
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    return df

    # download data from local file
    # data = pd.read_excel("spx.xlsx",index_col="Date")
    # data.columns = ["Close"]
    # data = data.sort_index(ascending=True)

    # # download data from xbbg
    # from xbbg import blp
    # data = blp.bdh("SPX Index","PX_LAST","1900-01-01") # HSI, NKY, SPX
    # data.columns = ["Close"]
    # data = data.sort_index(ascending=True)