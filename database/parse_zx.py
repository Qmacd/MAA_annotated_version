
import pandas as pd
import numpy as np

# === 1. 读取文件 ===
file_path = r"E:\Coding_path\GCA_lite\database\黄金.csv"
df = pd.read_csv(file_path)

# 若存在时间列则设为索引
if 'time' in df.columns and not np.issubdtype(df['time'].dtype, np.datetime64):
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

# === 2. 指标计算函数 ===
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()  # 使用 rolling-mean 计算移动平均 :contentReference[oaicite:0]{index=0}

# === 3. 计算指标（若不存在则新增） ===
# 3.1 MA
for w in (5, 15, 30, 60, 120):
    col = f"MA{w}"
    if col not in df.columns:
        # 需要的数据：'close' 列，即每日收盘价
        df[col] = sma(df['close'], w)  # 当窗口长度为 w 时，计算简单移动平均 :contentReference[oaicite:1]{index=1}

# 3.2 MACD
if not {'MACD', 'DIF', 'DEA'}.issubset(df.columns):
    fast, slow, signal = 12, 26, 9
    ema_fast = ema(df['close'], fast)
    ema_slow = ema(df['close'], slow)
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = ema(df['DIF'], signal)
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])

# 3.3 ATR
if 'ATR' not in df.columns:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

# 3.4 Bollinger Bands
if not {'BOLL_MID', 'BOLL_UP', 'BOLL_LOW'}.issubset(df.columns):
    window = 20
    mid = sma(df['close'], window)
    std = df['close'].rolling(window).std()
    df['BOLL_MID'] = mid
    df['BOLL_UP'] = mid + 2 * std
    df['BOLL_LOW'] = mid - 2 * std

# 3.5 RSI
if 'RSI' not in df.columns:
    period = 14
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

# 3.6 KDJ
if not {'K', 'D', 'J'}.issubset(df.columns):
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(alpha=1/3).mean()
    df['D'] = df['K'].ewm(alpha=1/3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']



# === 3.7 SAR ===
def parabolic_sar(high, low, init_af=0.02, step=0.02, max_af=0.2):
    length = len(high)
    sar = np.zeros(length)
    ep = high.iloc[0]
    af = init_af
    up_trend = True
    sar[0] = low.iloc[0]

    for i in range(1, length):
        prev_sar = sar[i-1]
        sar[i] = prev_sar + af * (ep - prev_sar)
        if up_trend:
            sar[i] = min(sar[i], low.iloc[i-1], low.iloc[i])
        else:
            sar[i] = max(sar[i], high.iloc[i-1], high.iloc[i])
        if up_trend and low.iloc[i] < sar[i]:
            up_trend = False
            sar[i] = ep
            ep = low.iloc[i]
            af = init_af
        elif not up_trend and high.iloc[i] > sar[i]:
            up_trend = True
            sar[i] = ep
            ep = high.iloc[i]
            af = init_af
        if up_trend and high.iloc[i] > ep:
            ep = high.iloc[i]
            af = min(af + step, max_af)
        elif not up_trend and low.iloc[i] < ep:
            ep = low.iloc[i]
            af = min(af + step, max_af)

    return pd.Series(sar, index=high.index)

if 'SAR' not in df.columns:
    df['SAR'] = parabolic_sar(df['high'], df['low'])


# === 3.8 W%R (Williams %R) ===
def williams_r(high, low, close, period=14):
    """
    计算 Williams %R 指标
    :param high:   pd.Series, 最高价
    :param low:    pd.Series, 最低价
    :param close:  pd.Series, 收盘价
    :param period: int, 窗口期（默认 14）
    :return:       pd.Series, Williams %R 值
    """
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low   = low.rolling(window=period, min_periods=1).min()
    wr = (highest_high - close) / (highest_high - lowest_low) * -100
    return wr

# 在 DataFrame 中写入 W%R
if 'WILLR' not in df.columns:
    df['WILLR'] = williams_r(df['high'], df['low'], df['close'], period=14)

# === 3.9 bias ===
def bias(series: pd.Series, period: int) -> pd.Series:
    """
    计算 BIAS（乖离率）。
    :param series: pd.Series，通常为收盘价序列
    :param period: int，计算移动平均的窗口期 N
    :return: pd.Series，BIAS 值
    """
    # 2.1 计算 N 日简单移动平均
    ma = series.rolling(window=period, min_periods=1).mean()  # Rolling.mean 计算滚动均值 :contentReference[oaicite:5]{index=5}
    # 2.2 计算乖离率公式
    return (series - ma) / ma * 100   # BIAS 公式： (C - MA) / MA * 100 :contentReference[oaicite:6]{index=6}


# 批量计算多周期 BIAS（例如 5, 15, 30 日）
for p in (5, 15, 30, 60, 120):
    col = f"BIAS{p}"
    if col not in df.columns:
        df[col] = bias(df['close'], p)         # 多列写入示例 :contentReference[oaicite:9]{index=9}


# === 3.10 涨跌幅 (Pct Change) ===
if 'Pct_Change' not in df.columns:
    df['Pct_Change'] = df['close'].pct_change(periods=1) * 100


# === 4. 保留 6 位小数 ===
df = df.round(6)

# === 5. 保存并展示 ===
output_path = "E:\Coding_path\GCA_lite\database\zx_processed_黄金_day.csv"
df.to_csv(output_path, index=False,float_format='%.6f')

df.tail()
