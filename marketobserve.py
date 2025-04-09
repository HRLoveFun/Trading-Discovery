import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import datetime as dt
from matplotlib.ticker import PercentFormatter
from scipy.stats import ks_2samp



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

def yf_download(ticker, start_date, end_date, interval='1d', progress=True, auto_adjust=False):
    """
    Download stock data from Yahoo Finance
    
    Parameters:
    - ticker: Stock symbol (e.g., "^HSI")
    - start_date: datetime.date or string in YYYY-MM-DD format
    - end_date: datetime.date or string in YYYY-MM-DD format
    - interval: Data interval ('1d', '1wk', '1mo')
    - progress: Show download progress bar
    - auto_adjust: Adjust all OHLC automatically
    
    Returns:
    - DataFrame with stock data
    """
    df = yf.download(
        ticker, 
        start=start_date, 
        end=end_date,
        interval=interval,
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

def create_data_sources(df, periods, all_period_start, frequency):
    """
    创建不同时间周期的数据来源
    """
    current_date = pd.Timestamp.now()

    # 根据 frequency 筛选数据到本周/本月/本季度的第一天
    if frequency == 'ME':
        end_date = current_date.replace(day=1)
    elif frequency == 'W':
        end_date = current_date - pd.DateOffset(days=current_date.weekday())
    elif frequency == 'QE':
        end_date = current_date - pd.tseries.offsets.QuarterBegin()
    else:
        raise ValueError("Invalid frequency value. Allowed values are 'ME', 'W', 'QE'.")

    df = df[df.index < end_date]
    last_date = df.index[-1]

    if all_period_start is None:
        all_period_start = "2010-01-01"

    data_sources = {}
    for period in periods:
        if isinstance(period, int):
            if frequency in ['ME', 'W']:
                start_date = last_date - pd.DateOffset(months=period - 1)
            elif frequency == 'QE':
                start_date = last_date - pd.DateOffset(quarters=period - 1)
            col_name = f"{start_date.strftime('%y%b')}-{last_date.strftime('%y%b')}"
            data_sources[col_name] = df.loc[df.index >= start_date]
        elif period == "ALL":
            col_name = f"{pd.to_datetime(all_period_start).strftime('%y%b')}-{last_date.strftime('%y%b')}"
            data_sources[col_name] = df.loc[df.index >= all_period_start]
        else:
            raise ValueError("Invalid period value")

    return data_sources


def refrequency(df, frequency: str):
    """
    对数据进行重采样
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    if not {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain OHLC columns")

    try:
        refrequency_df = df.resample(frequency).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Adj Close': 'last',
            'Volume': 'sum'
        }).dropna()

    except KeyError as e:
        import logging
        logging.error(f"Missing column {e} in DataFrame")
        raise ValueError(f"Error processing data: Missing column {e}")
    except Exception as e:
        import logging
        logging.error(f"Unexpected error: {str(e)}")
        raise ValueError(f"Error processing data: {str(e)}")

    return refrequency_df


def oscillation(df):
    """
    计算震荡指标
    """
    data = df[['Open', 'High', 'Low', 'Close']].copy()
    data['LastClose'] = data["Close"].shift(1)
    data["Oscillation"] = data["High"] - data["Low"]
    data["OscillationPct"] = (data["Oscillation"] / data['LastClose'])
    data = data.dropna()
    return data


def tail_stats(df, feature, frequency, periods: list = [12, 36, 60, "ALL"], all_period_start: str = None, interpolation: str = "linear"):
    """
    计算不同时间周期的统计指标
    """
    if not isinstance(periods, list):
        raise TypeError("periods must be a list")
    if not all(isinstance(p, (int, str)) for p in periods):
        raise ValueError("periods must contain integers or strings")

    data_sources = create_data_sources(df, periods, all_period_start, frequency)

    stats_index = pd.Index(["mean", "std", "skew", "kurt", "max", "99th", "95th", "90th"])
    stats_df = pd.DataFrame(index=stats_index)

    for period_name, data in data_sources.items():
        stats_df[period_name] = [
            data[feature].mean(),
            data[feature].std(),
            data[feature].skew(),
            data[feature].kurtosis(),
            data[feature].max(),
            data[feature].quantile(0.99, interpolation=interpolation),
            data[feature].quantile(0.95, interpolation=interpolation),
            data[feature].quantile(0.90, interpolation=interpolation)
        ]

    return stats_df


def tail_plot(df, feature, frequency, periods: list = [12, 36, 60, "ALL"], all_period_start: str = None, interpolation: str = "linear"):
    """
    绘制不同时间周期的特征值分布
    """
    data_sources = create_data_sources(df, periods, all_period_start, frequency)

    if frequency == "ME":
        bin_range = list(np.arange(0, 0.35, 0.05))
    elif frequency == "W":
        bin_range = list(np.arange(0, 0.18, 0.03))

    for period_name, data in data_sources.items():
        plt.figure(figsize=(10, 6))
        sns.set_style("darkgrid")
        n, bins, patches = plt.hist(data[feature], bins=bin_range, alpha=0.5, color='skyblue', density=True, cumulative=True)

        n_diff = np.insert(np.diff(n), 0, n[0])
        for rect, h_diff, h in zip(patches, n_diff, n):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, f'{h_diff:.0%}/{h:.0%}', ha='center', va='bottom', size=12)

        percentiles = [data[feature].quantile(p, interpolation=interpolation) for p in [0.90, 0.95, 0.99]]
        for p, val in zip([90, 95, 99], percentiles):
            plt.axvline(val, color='red', linestyle=':', alpha=0.3, label=f'{p}th: {val:.1%}')

        last_three = data[feature].iloc[-3:]
        last_three_dates = last_three.index.strftime('%b%d')
        for val, date, grayscale in zip(last_three, last_three_dates, np.arange(0.7, 0, -0.3)):
            plt.scatter(val, 0, color=str(grayscale), s=100, zorder=5, label=f'{date}: {val:.1%}')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"Distribution of {feature} - {period_name}")
        plt.xlabel(f"{feature} (%)")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def calculate_projections(data, feature, percentile, interpolation, bias_weight):
    data["ProjectHigh"] = data["LastClose"] + data["LastClose"] * data[feature].quantile(percentile, interpolation=interpolation) / 100 * bias_weight
    data["ProjectLow"] = data["LastClose"] - data["LastClose"] * data[feature].quantile(percentile, interpolation=interpolation) / 100 * (1 - bias_weight)
    data["ActualClosingStatus"] = np.where(data["Close"] > data["ProjectHigh"], 1,
                                           np.where(data["Close"] < data["ProjectLow"], -1, 0))
    realized_bias = ((data["ActualClosingStatus"] == 1).sum() - ((data["ActualClosingStatus"] == -1).sum())) / len(data)

    return realized_bias


def volatility_projection(df, feature, frequency: str = 'ME', percentile: float = 0.90, prefer_bias: float = None, periods: list = [12, 36, 60, "ALL"], all_period_start: str = None, interpolation: str = "linear"):
    """
    计算不同时间周期的波动率预测
    """
    if not isinstance(periods, list):
        raise TypeError("periods must be a list")
    if not all(isinstance(p, (int, str)) for p in periods):
        raise ValueError("periods must contain integers or strings")

    if feature == "OscillationPct":
        refrequency_data = refrequency(df, frequency=frequency)
        refrequency_feature = oscillation(refrequency_data)

        data_sources = create_data_sources(refrequency_feature, periods, all_period_start, frequency)

        volatility_projection_index = pd.Index(
            [
                f"Last: {refrequency_feature.index[-2].strftime('%y%b%d')}",
                f"{percentile}th {feature}",
                "RealizedBias%",
                "ProjectedHighWeight%",
                "ProjHigh",
                "ProjLow",
                f"Today: {df.index[-1].strftime('%y%b%d')}"
            ]
        )
        volatility_projection_df = pd.DataFrame(index=volatility_projection_index)

        for period_name, data in data_sources.items():
            period_end_close = data["Close"].iloc[-1]
            assumed_volatility = data[feature].quantile(percentile, interpolation=interpolation)

            if prefer_bias is not None:
                # 寻找最佳 bias_weight 的逻辑
                proj_high_weights = np.linspace(0.1, 0.9, 90)  # 在 0 到 1 之间生成 100 个等间距的 bias_weight 值
                min_error = float('inf')
                best_proj_high_weight = 0
                for proj_high_weight in proj_high_weights:
                    realized_bias = calculate_projections(data.copy(), feature, percentile, interpolation, proj_high_weight)
                    error = abs(realized_bias - prefer_bias)
                    if error < min_error:
                        min_error = error
                        best_proj_high_weight = proj_high_weight
                proj_high_weight = best_proj_high_weight

            else:
                proj_high_weight = 0.5

            realized_bias = calculate_projections(data, feature, percentile, interpolation, proj_high_weight)

            proj_high = period_end_close + period_end_close * assumed_volatility * proj_high_weight
            proj_low = period_end_close - period_end_close * assumed_volatility * (1 - proj_high_weight)

            last_close = df["Close"].iloc[-1]

            volatility_projection_df[period_name] = [
                period_end_close,
                assumed_volatility,
                realized_bias * 100,
                proj_high_weight * 100,
                proj_high,
                proj_low,
                last_close
            ]
        return volatility_projection_df
    else:
        raise ValueError("Invalid feature value")


def days_of_frequency(frequency):
    if frequency == "W":
        days = 5
    elif frequency == "ME":
        days = 21
    elif frequency == "QE":
        days = 63
    else:
        raise ValueError("Invalid frequency, input one of ['W', 'ME', 'QE']")

    return days


def tail_table(df, feature, frequency, periods: list = [12, 36, 60, "ALL"], all_period_start: str = None, interpolation: str = "linear"):
    """
    **To Modify Ouput Table Information** 按每个时间段计算表格，输出每个时间段及其对应的表格，最后将结果存储在字典中返回
    """
    data_sources = create_data_sources(df, periods, all_period_start, frequency)

    if frequency == "ME":
        bin_range = list(np.arange(0, 0.35, 0.05))
    elif frequency == "W":
        bin_range = list(np.arange(0, 0.18, 0.03))
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Supported frequencies are 'ME' and 'W'.")

    result_df = pd.DataFrame()
    for period_name, data in data_sources.items():
        # 计算直方图的累积密度
        n, bins = np.histogram(data[feature], bins=bin_range, density=True)
        cumulative_n = np.cumsum(n * np.diff(bins))
        n_diff = np.insert(np.diff(cumulative_n), 0, cumulative_n[0])

        percentiles = [0.90, 0.95, 0.99]

        # 计算百分位数
        percentile_values = [data[feature].quantile(p, interpolation=interpolation) for p in percentiles]

        # 获取最后三个数据点
        last_three_values = data[feature].iloc[-3:]
        last_three_dates = last_three_values.index.strftime('%b%d')
        bin_intervals = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

        bin_info = {
            "Period": period_name,
        }

        for _ in range(len(bin_intervals)):
            bin_info[f"{bin_intervals[_]}"] = f'{n_diff[_]:.0%}'

        for _ in range(len(percentiles)):
            bin_info[f"{percentiles[_]}th"] = f'{percentile_values[_]:.1%}'

        for _ in range(len(last_three_dates)):
            bin_info[f"{last_three_dates[_]}"] = f'{last_three_values.iloc[_]:.1%}'

        table = pd.DataFrame([bin_info])
        result_df = pd.concat([result_df, table], ignore_index=True)

    return result_df


def period_gap_stats(df, feature, frequency, periods: list = [12, 36, 60, "ALL"], all_period_start: str = None, interpolation: str = "linear"):
    """
    计算不同时间周期的频率缺口统计
    Given df, feature, and frequency,
        for each period in frequency,
            compute the gap_return = percentage change of first date open over last period close
            compute statistics of gap_return
            compute frequency_return = percentage change of last date close over last period close
            set days_of_period = len(df[rows only in the period])
            compare distribution of (gap_return+1)**days_of_period-1 with frequency_return distribution
        return distribution table of gap_return 
    """
    data_sources = create_data_sources(df, periods, all_period_start, frequency)

    stats_index = pd.Index(["mean", "std", "skew", "kurt", "max", "99th", "95th", "90th", "10th", "05th", "01st", "min", "p-value"])
    gap_return_stats_df = pd.DataFrame(index=stats_index)

    if feature == "PeriodGap":
        for period_name, data in data_sources.items():
            if len(data) > 0:
                # 计算 gap_return
                gap_return = (data["Open"] / data["LastClose"] - 1)
                period_return = (data["Close"] / data["LastClose"] - 1)

                # 计算 days_of_period
                days_of_period = days_of_frequency(frequency)

                # 计算 (gap_return+1)**days_of_period-1
                compounded_gap_return = (1 + gap_return) ** days_of_period - 1
                # 计算 period_return 和 compounded_gap_return 相似程度的统计检验
                _, p_value = ks_2samp(compounded_gap_return, period_return)

                # 计算 gap_return 的统计信息
                gap_return_stats_df[period_name] = [
                    gap_return.mean(),
                    gap_return.std(),
                    gap_return.skew(),
                    gap_return.kurtosis(),
                    gap_return.max(),
                    gap_return.quantile(0.99, interpolation=interpolation),
                    gap_return.quantile(0.95, interpolation=interpolation),
                    gap_return.quantile(0.90, interpolation=interpolation),
                    gap_return.quantile(0.10, interpolation=interpolation),
                    gap_return.quantile(0.05, interpolation=interpolation),
                    gap_return.quantile(0.01, interpolation=interpolation),
                    gap_return.min(),
                    p_value
                ]

    return gap_return_stats_df


def option_matrix(ticker, option_position):
    """
    option_position: dataframe. columns: option_type: values of [LC,SC,LP,SP]; strike: integer, quantity: integer, premium: float
    """

    # Get the last price of ticker
    close = yf.download(ticker, start=dt.datetime.now().date())[["Close"]].iloc[-1,-1]

    # Create default option_matrix dataframe, columns = ['Price', 'SC', 'SP', 'LC', 'LP'], index = range of close*0.9, close*1.1
    change_range = np.linspace(0.9, 1.1, 20)
    option_matrix_df = pd.DataFrame(index=change_range)
    option_matrix_df['price'] = (close*change_range).astype(int)
    option_matrix_df['SC'] = 0.0
    option_matrix_df['SP'] = 0.0
    option_matrix_df['LC'] = 0.0
    option_matrix_df['LP'] = 0.0

    for _, row in option_position.iterrows():
        option_type = row["option_type"]
        strike = row["strike"]
        quantity = row["quantity"]
        premium = row["premium"]

        print(f"Option type: {option_type}, Strike: {strike}, Premium: {premium}")

        if option_type == 'SC':
            option_matrix_df.loc[option_matrix_df['price'] < strike, 'SC'] = premium
            option_matrix_df.loc[option_matrix_df['price'] >= strike, 'SC'] = premium + (strike - option_matrix_df.loc[option_matrix_df['price'] >= strike, 'price'])
            option_matrix_df['SC'] *= quantity
        elif option_type == 'SP':
            option_matrix_df.loc[option_matrix_df['price'] > strike, 'SP'] = premium
            option_matrix_df.loc[option_matrix_df['price'] <= strike, 'SP'] = premium - (strike - option_matrix_df.loc[option_matrix_df['price'] <= strike, 'price'])
            option_matrix_df['SP'] *= quantity

        elif option_type == 'LC':
            option_matrix_df.loc[option_matrix_df['price'] > strike, 'LC'] = option_matrix_df.loc[option_matrix_df['price'] > strike, 'price'] - strike - premium
            option_matrix_df.loc[option_matrix_df['price'] <= strike, 'LC'] = - premium
            option_matrix_df['LC'] *= quantity
        
        elif option_type == 'LP':
            option_matrix_df.loc[option_matrix_df['price'] > strike, 'LP'] = - premium
            option_matrix_df.loc[option_matrix_df['price'] <= strike, 'LP'] = - option_matrix_df.loc[option_matrix_df['price'] <= strike, 'price'] + strike - premium
            option_matrix_df['LP'] *= quantity

        else:
            raise ValueError("Invalid option type")


    # Calculate the PnL
    option_matrix_df['PnL'] = option_matrix_df[['SC', 'SP', 'LC', 'LP']].sum(axis=1)



    return option_matrix_df

    