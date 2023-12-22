import pandas as pd
import numpy as np
def sma(data, window=20):
    return data['Close'].rolling(window=window).mean()


# Exponential Moving Average (EMA)
def ema(data, span=20):
    return data['Close'].ewm(span=span, adjust=False).mean()


# Moving Average Convergence Divergence (MACD)
def macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = ema(data, span=short_window)
    long_ema = ema(data, span=long_window)
    # Calculate DIF, DEA, and MACD
    data['DIF_' + str(short_window) + '_' + str(long_window)] = short_ema - long_ema
    data['DEA_' + str(short_window) + '_' + str(long_window)] = data['DIF_' + str(short_window) + '_' + str(
        long_window)].shift() * 0.2 + data['DIF_' + str(short_window) + '_' + str(long_window)] * 0.8
    data['MACD_' + str(short_window) + '_' + str(long_window)] = (data['DIF_' + str(short_window) + '_' + str(
        long_window)] - data['DEA_' + str(short_window) + '_' + str(long_window)]) * 2
    data['EMA_' + str(signal_window)] = ema(data, span=signal_window)
    return data


# Bollinger Bands
def bollinger_bands(data, window=20, num_std=2):
    data['SMA_' + str(window)] = sma(data, window=window)
    data['STD_' + str(window)] = data['Close'].rolling(window=window).std()
    data['Upper_Band_' + str(window) + '_' + str(num_std)] = data['SMA_' + str(window)] + (
            data['STD_' + str(window)] * num_std)
    data['Lower_Band_' + str(window) + '_' + str(num_std)] = data['SMA_' + str(window)] - (
            data['STD_' + str(window)] * num_std)
    return data


# Relative Strength Index (RSI)
def rsi(data, window=14):
    daily_returns = data['Close'].pct_change()
    gain = daily_returns.where(daily_returns > 0, 0)
    loss = -daily_returns.where(daily_returns < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    data['RSI_' + str(window)] = 100 - (100 / (1 + avg_gain / avg_loss))
    return data


# Average True Range (ATR)
def atr(data, window=14):
    tr = pd.DataFrame(index=data.index)
    tr['H-L_' + str(window)] = data['High'] - data['Low']
    tr['H-PC_' + str(window)] = abs(data['High'] - data['Close'].shift())
    tr['L-PC_' + str(window)] = abs(data['Low'] - data['Close'].shift())
    tr['TR_' + str(window)] = tr[['H-L_' + str(window), 'H-PC_' + str(window), 'L-PC_' + str(window)]].max(axis=1)
    data['ATR_' + str(window)] = tr['TR_' + str(window)].rolling(window=window).mean()
    return data


# Ichimoku Cloud (Note: This is a simplified version)
def ichimoku_cloud(data, conversion_line_period=9, base_line_period=26, leading_span_b_period=52,
                   lagging_span_window=26):
    data['Conversion_Line_' + str(conversion_line_period) + '_' + str(base_line_period) + '_' + str(
        leading_span_b_period) + '_' + str(lagging_span_window)] = (data['High'].rolling(
        window=conversion_line_period).max() + data['Low'].rolling(window=conversion_line_period).min()) / 2
    data['Base_Line_' + str(conversion_line_period) + '_' + str(base_line_period) + '_' + str(
        leading_span_b_period) + '_' + str(lagging_span_window)] = (data['High'].rolling(
        window=base_line_period).max() + data['Low'].rolling(window=base_line_period).min()) / 2
    data['Leading_Span_A_' + str(conversion_line_period) + '_' + str(base_line_period) + '_' + str(
        leading_span_b_period) + '_' + str(lagging_span_window)] = (data['Conversion_Line_' + str(
        conversion_line_period) + '_' + str(base_line_period) + '_' + str(leading_span_b_period) + '_' + str(
        lagging_span_window)] + data['Base_Line_' + str(conversion_line_period) + '_' + str(
        base_line_period) + '_' + str(leading_span_b_period) + '_' + str(lagging_span_window)]) / 2
    data['Leading_Span_B_' + str(conversion_line_period) + '_' + str(base_line_period) + '_' + str(
        leading_span_b_period) + '_' + str(lagging_span_window)] = (data['High'].rolling(
        window=leading_span_b_period).max() + data['Low'].rolling(window=leading_span_b_period).min()) / 2
    data['Lagging_Span_' + str(conversion_line_period) + '_' + str(base_line_period) + '_' + str(
        leading_span_b_period) + '_' + str(lagging_span_window)] = data['Close'].shift(-lagging_span_window)
    return data


# Parabolic SAR (Stop and Reverse)
def parabolic_sar(data, acceleration=0.02, max_acceleration=0.2):
    high = data['High']
    low = data['Low']
    close = data['Close']

    data['SAR_' + str(acceleration) + '_' + str(max_acceleration)] = np.nan

    # Initial values
    sar = low[0]
    ep = high[0]
    acceleration_factor = acceleration
    trend = 1  # 1 for uptrend, -1 for downtrend

    for i in range(2, len(data)):
        sar = sar + acceleration_factor * (ep - sar)

        # Check for trend reversal
        if trend == 1 and close[i] < sar:
            trend = -1
            sar = ep
            acceleration_factor = acceleration
        elif trend == -1 and close[i] > sar:
            trend = 1
            sar = ep
            acceleration_factor = acceleration

        # Update acceleration factor if not at the maximum
        if acceleration_factor < max_acceleration:
            acceleration_factor += acceleration

        # Update extreme price (EP) for the next iteration
        ep = max(ep, high[i]) if trend == 1 else min(ep, low[i])

        data.at[data.index[i], 'SAR_' + + str(acceleration) + '_' + str(max_acceleration)] = sar

    return data


def parabolic_sar_trend(data, acceleration=0.02, max_acceleration=0.2):
    data = parabolic_sar(data, acceleration=acceleration, max_acceleration=max_acceleration)

    # Calculate SAR trend
    data['SAR_Trend_' + str(acceleration) + '_' + str(max_acceleration)] = np.where(
        data['Close'] > data['SAR_' + str(acceleration) + '_' + str(max_acceleration)], 1, -1)

    return data


# Stochastic Oscillator
def stochastic_oscillator(data, k_period=14, d_period=3):
    data['Lowest_Low_' + str(k_period) + '_' + str(d_period)] = data['Low'].rolling(window=k_period).min()
    data['Highest_High_' + str(k_period) + '_' + str(d_period)] = data['High'].rolling(window=k_period).max()
    data['%K_' + str(k_period) + '_' + str(d_period)] = ((data['Close'] - data[
        'Lowest_Low_' + str(k_period) + '_' + str(d_period)]) / (data['Highest_High_' + str(k_period) + '_' + str(
        d_period)] - data['Lowest_Low_' + str(k_period) + '_' + str(d_period)])) * 100
    data['%D_' + str(k_period) + '_' + str(d_period)] = data['%K_' + str(k_period) + '_' + str(d_period)].rolling(
        window=d_period).mean()
    return data


# Average Directional Index (ADX)
def adx(data, window=14):
    diff_high = data['High'].diff()
    diff_low = -data['Low'].diff()
    true_range = pd.concat([diff_high, diff_low], axis=1).max(axis=1)
    data['+DM_' + str(window)] = np.where((data['High'] - data['High'].shift()) > (data['Low'].shift() - data['Low']),
                                          diff_high, 0)
    data['-DM_' + str(window)] = np.where((data['Low'].shift() - data['Low']) > (data['High'] - data['High'].shift()),
                                          diff_low, 0)
    data['+DI_' + str(window)] = (data['+DM_' + str(window)].rolling(window=window).sum() / true_range.rolling(
        window=window).sum()) * 100
    data['-DI_' + str(window)] = (data['-DM_' + str(window)].rolling(window=window).sum() / true_range.rolling(
        window=window).sum()) * 100
    data['ADX_' + str(window)] = ((data['+DI_' + str(window)] - data['-DI_' + str(window)]).abs() / (
            data['+DI_' + str(window)] + data['-DI_' + str(window)])) * 100
    return data


# Commodity Channel Index (CCI)
def cci(data, window=20):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    mean_deviation = typical_price - typical_price.rolling(window=window).mean()
    data['CCI_' + str(window)] = (typical_price - typical_price.rolling(window=window).mean()) / (
            0.015 * mean_deviation.rolling(window=window).mean())
    return data


# Candlestick Patterns

# Doji
def doji(data):
    data['Is_Doji'] = np.where(abs(data['Open'] - data['Close']) <= 0.1 * (data['High'] - data['Low']), 1, 0)
    return data


# Hammer
def hammer(data):
    data['Is_Hammer'] = np.where((data['High'] - data['Low'] > 3 * (data['Open'] - data['Close'])) &
                                 (data['Close'] - data['Low'] > 0.6 * (data['High'] - data['Low'])) &
                                 (data['Close'] - data['Open'] > 0.6 * (data['High'] - data['Low'])), 1, 0)
    return data


# Shooting Star
def shooting_star(data):
    data['Is_Shooting_Star'] = np.where((data['High'] - data['Low'] > 3 * (data['Open'] - data['Close'])) &
                                        (data['High'] - data['Close'] > 0.6 * (data['High'] - data['Low'])) &
                                        (data['Close'] - data['Open'] > 0.6 * (data['High'] - data['Low'])), 1, 0)
    return data


# Engulfing Patterns

# Bullish Engulfing
def bullish_engulfing(data):
    data['Is_Bullish_Engulfing'] = np.where((data['Close'].shift() < data['Open'].shift()) &
                                            (data['Close'] > data['Open']) &
                                            (data['Close'] > data['Open'].shift()) &
                                            (data['Close'].shift() < data['Open']), 1, 0)
    return data


# Bearish Engulfing
def bearish_engulfing(data):
    data['Is_Bearish_Engulfing'] = np.where((data['Close'].shift() > data['Open'].shift()) &
                                            (data['Close'] < data['Open']) &
                                            (data['Close'] < data['Open'].shift()) &
                                            (data['Close'].shift() > data['Open']), 1, 0)
    return data


# Morning Star
def morning_star(data):
    data['Is_Morning_Star'] = np.where((data['Close'].shift(2) > data['Open'].shift(2)) &
                                       (data['Close'].shift() < data['Open'].shift()) &
                                       (data['Close'] > data['Open']) &
                                       (data['Close'] > data['Close'].shift(2)) &
                                       (data['Open'] < data['Close'].shift()) &
                                       (data['Open'] < data['Open'].shift(2)), 1, 0)
    return data


# Evening Star
def evening_star(data):
    data['Is_Evening_Star'] = np.where((data['Close'].shift(2) < data['Open'].shift(2)) &
                                       (data['Close'].shift() > data['Open'].shift()) &
                                       (data['Close'] < data['Open']) &
                                       (data['Close'] < data['Close'].shift(2)) &
                                       (data['Open'] > data['Close'].shift()) &
                                       (data['Open'] > data['Open'].shift(2)), 1, 0)
    return data


# Dark Cloud Cover
def dark_cloud_cover(data):
    data['Previous_Close'] = data['Close'].shift(1)
    data['Previous_Open'] = data['Open'].shift(1)
    data['Body'] = abs(data['Open'] - data['Close'])
    data['Midpoint'] = (data['Open'] + data['Close']) / 2

    # Dark Cloud Cover condition
    dark_cloud_cover_pattern = ((data['Previous_Close'] > data['Previous_Open']) &  # Previous candle is bullish
                                (data['Open'] > data['Previous_Close']) &  # Current candle opens higher
                                (data['Close'] < data['Midpoint']))  # Current candle closes below midpoint

    data['Dark_Cloud_Cover'] = dark_cloud_cover_pattern.astype(int)
    data = data.drop(['Previous_Close', 'Previous_Open', 'Body', 'Midpoint'], axis=1)

    return data


# Piercing Line
def piercing_line(data):
    data['Previous_Close'] = data['Close'].shift(1)
    data['Previous_Open'] = data['Open'].shift(1)
    data['Body'] = abs(data['Open'] - data['Close'])
    data['Midpoint'] = (data['Open'] + data['Close']) / 2

    # Piercing Line condition
    piercing_line_pattern = ((data['Previous_Close'] > data['Previous_Open']) &  # Previous candle is bearish
                             (data['Open'] < data['Previous_Close']) &  # Current candle opens lower
                             (data['Close'] > data['Midpoint']))  # Current candle closes above midpoint

    data['Piercing_Line'] = piercing_line_pattern.astype(int)
    data = data.drop(['Previous_Close', 'Previous_Open', 'Body', 'Midpoint'], axis=1)

    return data


def simple_features_generate(data):
    data['Body'] = abs(data['Open'] - data['Close'])
    data['Midpoint'] = (data['Open'] + data['Close']) / 2
    return data


def vix(data, window=20):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Assuming 252 trading days in a year
    vix = volatility * 100  # Scaling for a percentage
    data['VIX_' + str(window)] = vix
    return data

def create_features(df):
    df['SMA_5'] = sma(df, window=5)
    df['SMA_9'] = sma(df, window=9)
    df['SMA_12'] = sma(df, window=12)
    df['SMA_14'] = sma(df, window=14)
    df['SMA_26'] = sma(df, window=26)

    df['EMA_5'] = sma(df, window=5)
    df['EMA_9'] = sma(df, window=9)
    df['EMA_12'] = sma(df, window=12)
    df['EMA_14'] = sma(df, window=14)
    df['EMA_26'] = sma(df, window=26)

    df = macd(df)

    df = bollinger_bands(df)

    df = rsi(df, window=5)
    df = rsi(df, window=9)
    df = rsi(df, window=12)
    df = rsi(df, window=14)
    df = rsi(df, window=26)

    df = atr(df, window=5)
    df = atr(df, window=9)
    df = atr(df, window=12)
    df = atr(df, window=14)
    df = atr(df, window=26)

    df = ichimoku_cloud(df)
    df = ichimoku_cloud(df, conversion_line_period=9, base_line_period=12, leading_span_b_period=26,
                        lagging_span_window=12)

    df = stochastic_oscillator(df, k_period=14, d_period=3)
    df = stochastic_oscillator(df, k_period=21, d_period=5
                               )
    df = adx(df, window=10)
    df = adx(df, window=14)
    df = adx(df, window=20)
    df = adx(df, window=30)

    df = cci(df, window=10)
    df = cci(df, window=14)
    df = cci(df, window=20)

    df = simple_features_generate(df)

    df = doji(df)
    df = hammer(df)
    df = shooting_star(df)
    df = bullish_engulfing(df)
    df = bearish_engulfing(df)
    df = morning_star(df)
    df = evening_star(df)
    df = dark_cloud_cover(df)
    df = piercing_line(df)

    return df