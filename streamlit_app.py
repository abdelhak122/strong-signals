import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="مؤشر سكالبينج متطور",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for RTL support and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
    }
    
    .main {
        background-color: #1E1E2E;
        color: white;
    }
    
    h1, h2, h3 {
        color: white;
    }
    
    .stButton>button {
        background-color: #8C52FF;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #7540e0;
        transform: translateY(-2px);
    }
    
    /* Professional table styling */
    .signal-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .signal-table th {
        background-color: #2A2A3C;
        color: white;
        padding: 12px 15px;
        text-align: center;
        font-weight: bold;
        border-bottom: 2px solid #3A3A4C;
    }
    
    .signal-table td {
        padding: 10px 15px;
        text-align: center;
        border-bottom: 1px solid #3A3A4C;
    }
    
    .signal-table tr {
        background-color: #2A2A3C;
    }
    
    .signal-table tr:hover {
        background-color: #3A3A4C;
    }
    
    .signal-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Signal type styling */
    .signal-buy {
        background-color: #52FF7D;
        color: #1E1E2E;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-sell {
        background-color: #FF5252;
        color: #1E1E2E;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-neutral {
        background-color: #CCCCCC;
        color: #1E1E2E;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Timeframe badge */
    .timeframe-badge {
        background-color: #3A3A4C;
        color: white;
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
        margin-right: 5px;
    }
    
    /* Confidence badge */
    .confidence-badge {
        color: white;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Profit and loss values */
    .profit-value {
        color: #52FF7D;
        font-weight: bold;
    }
    
    .loss-value {
        color: #FF5252;
        font-weight: bold;
    }
    
    /* Last update indicator */
    .last-update {
        text-align: center;
        color: #AAAAAA;
        margin-bottom: 15px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2A2A3C;
    }
    
    /* Fix for RTL in dataframes */
    .dataframe {
        direction: rtl;
    }
    
    /* Section headers */
    .section-header {
        background-color: #3A3A4C;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* Symbol column styling */
    .symbol-column {
        font-weight: bold;
        font-size: 1.1em;
    }
    
    /* Price column styling */
    .price-column {
        font-family: monospace;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Dictionary of symbols for different markets
symbols = {
    # Forex pairs
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    
    # Commodities
    'XAU/USD': 'GC=F',  # Gold
    'XAG/USD': 'SI=F',  # Silver
    'OIL/USD': 'CL=F',  # Crude Oil
    
    # Crypto
    'BTC/USD': 'BTC-USD',
    'ETH/USD': 'ETH-USD',
    
    # Indices
    'NAS100': '^NDX',
    'S&P500': '^GSPC',
    'DOW': '^DJI'
}

# Timeframes for analysis
timeframes = {
    '1م': '1m',
    '5م': '5m',
    '15م': '15m'
}

# Confidence range mapping
def get_confidence_range(confidence):
    """Convert confidence percentage to range and color"""
    if confidence >= 95:
        return "95-100%", '#8C52FF'  # Royal Purple
    elif confidence >= 90:
        return "90-94%", '#5271FF'  # Royal Blue
    elif confidence >= 85:
        return "85-89%", '#52B5FF'  # Sky Blue
    elif confidence >= 80:
        return "80-84%", '#52FFD0'  # Turquoise
    elif confidence >= 75:
        return "75-79%", '#52FF7D'  # Mint Green
    elif confidence >= 70:
        return "70-74%", '#9DFF52'  # Lime Green
    elif confidence >= 65:
        return "65-69%", '#D6FF52'  # Yellow Green
    elif confidence >= 60:
        return "60-64%", '#FFE252'  # Gold
    elif confidence >= 55:
        return "55-59%", '#FFBD52'  # Orange
    elif confidence >= 50:
        return "50-54%", '#FF8652'  # Light Orange
    elif confidence >= 45:
        return "45-49%", '#FF5252'  # Red
    elif confidence >= 40:
        return "40-44%", '#FF527A'  # Pink
    elif confidence >= 35:
        return "35-39%", '#FF52B5'  # Hot Pink
    elif confidence >= 30:
        return "30-34%", '#D052FF'  # Purple
    elif confidence >= 25:
        return "25-29%", '#8952FF'  # Violet
    elif confidence >= 20:
        return "20-24%", '#5257FF'  # Blue
    elif confidence >= 15:
        return "15-19%", '#52BDFF'  # Light Blue
    elif confidence >= 10:
        return "10-14%", '#52FFE2'  # Aqua
    elif confidence >= 5:
        return "5-9%", '#52FF94'  # Light Green
    else:
        return "0-4%", '#CCCCCC'  # Gray

# Data fetching function with error handling and retries
@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_data(symbol, period='1d', interval='5m', retries=3):
    """
    Fetch market data with retry mechanism
    
    Parameters:
    - symbol: Market symbol
    - period: Time period to fetch (default: 1d)
    - interval: Candle interval (default: 5m)
    - retries: Number of retry attempts
    
    Returns:
    - DataFrame with OHLCV data or error message
    """
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    return None, f"لا توجد بيانات للرمز {symbol}"
            
            # Ensure volume data is available (some forex data might not have volume)
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                df['Volume'] = 1000  # Assign default volume
            
            return df, None
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            else:
                return None, f"فشل في جلب البيانات للرمز {symbol}: {str(e)}"

# Technical Indicators
class TechnicalIndicators:
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data['Close'].rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data['Close'].ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 0.00001)
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands"""
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

# Candlestick Pattern Recognition
class CandlestickPatterns:
    @staticmethod
    def doji(df, tolerance=0.05):
        """
        Detect Doji candlestick pattern
        A Doji has open and close prices that are virtually equal
        """
        body_size = abs(df['Close'] - df['Open'])
        candle_range = df['High'] - df['Low']
        
        # Avoid division by zero
        candle_range = candle_range.replace(0, 0.00001)
        
        body_to_range_ratio = body_size / candle_range
        return body_to_range_ratio < tolerance
    
    @staticmethod
    def hammer(df, body_ratio=0.3, shadow_ratio=2.0):
        """
        Detect Hammer candlestick pattern
        A Hammer has a small body at the top with a long lower shadow
        """
        body_size = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        candle_range = df['High'] - df['Low']
        
        # Avoid division by zero
        candle_range = candle_range.replace(0, 0.00001)
        
        # Conditions for a hammer
        small_body = body_size / candle_range < body_ratio
        long_lower_shadow = lower_shadow / body_size > shadow_ratio
        small_upper_shadow = upper_shadow < body_size
        
        return small_body & long_lower_shadow & small_upper_shadow & (df['Close'] > df['Open'])
    
    @staticmethod
    def shooting_star(df, body_ratio=0.3, shadow_ratio=2.0):
        """
        Detect Shooting Star candlestick pattern
        A Shooting Star has a small body at the bottom with a long upper shadow
        """
        body_size = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        candle_range = df['High'] - df['Low']
        
        # Avoid division by zero
        candle_range = candle_range.replace(0, 0.00001)
        
        # Conditions for a shooting star
        small_body = body_size / candle_range < body_ratio
        long_upper_shadow = upper_shadow / body_size > shadow_ratio
        small_lower_shadow = lower_shadow < body_size
        
        return small_body & long_upper_shadow & small_lower_shadow & (df['Close'] < df['Open'])
    
    @staticmethod
    def engulfing(df):
        """
        Detect Bullish and Bearish Engulfing patterns
        """
        # Shift to get previous candle
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)
        curr_open = df['Open']
        curr_close = df['Close']
        
        # Bullish engulfing
        bullish_engulfing = (
            (curr_close > curr_open) &  # Current candle is bullish
            (prev_close < prev_open) &  # Previous candle is bearish
            (curr_open < prev_close) &  # Current open is below previous close
            (curr_close > prev_open)    # Current close is above previous open
        )
        
        # Bearish engulfing
        bearish_engulfing = (
            (curr_close < curr_open) &  # Current candle is bearish
            (prev_close > prev_open) &  # Previous candle is bullish
            (curr_open > prev_close) &  # Current open is above previous close
            (curr_close < prev_open)    # Current close is below previous open
        )
        
        return bullish_engulfing, bearish_engulfing

# Support and Resistance Detection
class SupportResistance:
    @staticmethod
    def find_levels(df, window=10, threshold=0.01):
        """
        Find support and resistance levels using local minima and maxima
        
        Parameters:
        - df: DataFrame with price data
        - window: Window size for peak detection
        - threshold: Minimum percentage difference between levels
        
        Returns:
        - support_levels: List of support levels
        - resistance_levels: List of resistance levels
        """
        # Find local minima and maxima
        df['min'] = df['Low'].rolling(window=window, center=True).min()
        df['max'] = df['High'].rolling(window=window, center=True).max()
        
        # Identify potential support and resistance points
        support_points = df[df['Low'] == df['min']]['Low'].values
        resistance_points = df[df['High'] == df['max']]['High'].values
        
        # Cluster similar levels
        support_levels = []
        resistance_levels = []
        
        # Process support levels
        for point in support_points:
            # Check if the point is close to any existing level
            if not any(abs(point - level) / level < threshold for level in support_levels):
                support_levels.append(point)
        
        # Process resistance levels
        for point in resistance_points:
            # Check if the point is close to any existing level
            if not any(abs(point - level) / level < threshold for level in resistance_levels):
                resistance_levels.append(point)
        
        return support_levels, resistance_levels
    
    @staticmethod
    def is_near_level(price, levels, threshold=0.002):
        """Check if price is near any support/resistance level"""
        for level in levels:
            if abs(price - level) / level < threshold:
                return True, level
        return False, None

# Volume Analysis
class VolumeAnalysis:
    @staticmethod
    def volume_trend(df, window=5):
        """
        Analyze volume trend
        
        Returns:
        - increasing: Boolean series indicating increasing volume
        - decreasing: Boolean series indicating decreasing volume
        - spike: Boolean series indicating volume spike
        """
        # Calculate average volume
        avg_volume = df['Volume'].rolling(window=window).mean()
        
        # Identify volume trends
        increasing = df['Volume'] > df['Volume'].shift(1)
        decreasing = df['Volume'] < df['Volume'].shift(1)
        
        # Identify volume spikes (volume > 2x average)
        spike = df['Volume'] > (2 * avg_volume)
        
        return increasing, decreasing, spike
    
    @staticmethod
    def price_volume_divergence(df, window=14):
        """
        Detect price-volume divergence
        
        Returns:
        - bullish_divergence: Boolean series indicating bullish divergence
        - bearish_divergence: Boolean series indicating bearish divergence
        """
        # Calculate price and volume trends
        price_trend = df['Close'].diff().rolling(window=window).sum()
        volume_trend = df['Volume'].diff().rolling(window=window).sum()
        
        # Bullish divergence: Price down, volume up
        bullish_divergence = (price_trend < 0) & (volume_trend > 0)
        
        # Bearish divergence: Price up, volume down
        bearish_divergence = (price_trend > 0) & (volume_trend < 0)
        
        return bullish_divergence, bearish_divergence

# Signal Generator
class ScalpingSignalGenerator:
    def __init__(self):
        self.ti = TechnicalIndicators()
        self.cp = CandlestickPatterns()
        self.sr = SupportResistance()
        self.va = VolumeAnalysis()
    
    def analyze_timeframe(self, df, timeframe, symbol_name):
        """
        Analyze a specific timeframe and generate signals
        
        Parameters:
        - df: DataFrame with price data
        - timeframe: Timeframe being analyzed
        - symbol_name: Name of the symbol being analyzed
        
        Returns:
        - signal: 'buy', 'sell', or 'neutral'
        - entry_price: Suggested entry price
        - tp: Take profit level
        - sl: Stop loss level
        - description: Signal description
        - confidence: Signal confidence (0-100%)
        - pips_potential: Potential pips gain
        - risk_reward: Risk to reward ratio
        """
        # Add technical indicators
        df['EMA_8'] = self.ti.ema(df, 8)
        df['EMA_13'] = self.ti.ema(df, 13)
        df['EMA_21'] = self.ti.ema(df, 21)
        df['RSI_14'] = self.ti.rsi(df, 14)
        df['MACD'], df['Signal'], df['Histogram'] = self.ti.macd(df)
        df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = self.ti.bollinger_bands(df)
        
        # Detect candlestick patterns
        df['Doji'] = self.cp.doji(df)
        df['Hammer'] = self.cp.hammer(df)
        df['Shooting_Star'] = self.cp.shooting_star(df)
        df['Bullish_Engulfing'], df['Bearish_Engulfing'] = self.cp.engulfing(df)
        
        # Volume analysis
        df['Volume_Increasing'], df['Volume_Decreasing'], df['Volume_Spike'] = self.va.volume_trend(df)
        df['Bullish_Divergence'], df['Bearish_Divergence'] = self.va.price_volume_divergence(df)
        
        # Find support and resistance levels
        support_levels, resistance_levels = self.sr.find_levels(df)
        
        # Get the latest data point
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize signal variables
        signal = "لا توجد إشارة"  # No signal
        entry_price = None
        tp = None
        sl = None
        description = ""
        confidence = 0
        pips_potential = 0
        risk_reward = 0
        reasons = []
        
        # Current price
        current_price = latest['Close']
        
        # Check for EMA alignment (Multi-EMA Strategy from Forex Factory)
        ema_bullish = (latest['EMA_8'] > latest['EMA_13'] > latest['EMA_21'])
        ema_bearish = (latest['EMA_8'] < latest['EMA_13'] < latest['EMA_21'])
        
        # Check for EMA crossovers
        ema_8_13_bull_cross = (prev['EMA_8'] <= prev['EMA_13']) and (latest['EMA_8'] > latest['EMA_13'])
        ema_8_13_bear_cross = (prev['EMA_8'] >= prev['EMA_13']) and (latest['EMA_8'] < latest['EMA_13'])
        
        # Check RSI conditions
        rsi_oversold = latest['RSI_14'] < 30
        rsi_overbought = latest['RSI_14'] > 70
        rsi_bullish = (prev['RSI_14'] < 50) and (latest['RSI_14'] > 50)
        rsi_bearish = (prev['RSI_14'] > 50) and (latest['RSI_14'] < 50)
        
        # Check MACD conditions
        macd_bullish = (prev['MACD'] <= prev['Signal']) and (latest['MACD'] > latest['Signal'])
        macd_bearish = (prev['MACD'] >= prev['Signal']) and (latest['MACD'] < latest['Signal'])
        
        # Check Bollinger Bands conditions
        bb_lower_touch = latest['Low'] <= latest['Lower_Band']
        bb_upper_touch = latest['High'] >= latest['Upper_Band']
        
        # Check candlestick patterns
        doji = latest['Doji']
        hammer = latest['Hammer']
        shooting_star = latest['Shooting_Star']
        bullish_engulfing = latest['Bullish_Engulfing']
        bearish_engulfing = latest['Bearish_Engulfing']
        
        # Check volume conditions
        volume_spike = latest['Volume_Spike']
        bullish_divergence = latest['Bullish_Divergence']
        bearish_divergence = latest['Bearish_Divergence']
        
        # Check support/resistance
        near_support, support_level = self.sr.is_near_level(current_price, support_levels)
        near_resistance, resistance_level = self.sr.is_near_level(current_price, resistance_levels)
        
        # Generate Buy Signal
        buy_points = 0
        if ema_bullish: buy_points += 15
        if ema_8_13_bull_cross: buy_points += 10
        if rsi_oversold: buy_points += 10
        if rsi_bullish: buy_points += 5
        if macd_bullish: buy_points += 10
        if bb_lower_touch: buy_points += 10
        if hammer: buy_points += 10
        if bullish_engulfing: buy_points += 15
        if volume_spike and latest['Close'] > latest['Open']: buy_points += 5
        if bullish_divergence: buy_points += 10
        if near_support: buy_points += 15
        
        # Generate Sell Signal
        sell_points = 0
        if ema_bearish: sell_points += 15
        if ema_8_13_bear_cross: sell_points += 10
        if rsi_overbought: sell_points += 10
        if rsi_bearish: sell_points += 5
        if macd_bearish: sell_points += 10
        if bb_upper_touch: sell_points += 10
        if shooting_star: sell_points += 10
        if bearish_engulfing: sell_points += 15
        if volume_spike and latest['Close'] < latest['Open']: sell_points += 5
        if bearish_divergence: sell_points += 10
        if near_resistance: sell_points += 15
        
        # Determine final signal
        if buy_points > sell_points and buy_points >= 30:
            signal = "شراء"
            confidence = min(buy_points, 100)
            entry_price = current_price
            
            # Calculate take profit and stop loss
            if near_resistance and resistance_level is not None:
                tp = resistance_level
            else:
                tp = round(entry_price * 1.01, 4)  # Default 1% take profit
                
            if near_support and support_level is not None:
                sl = support_level * 0.998  # Just below support
            else:
                sl = round(entry_price * 0.995, 4)  # Default 0.5% stop loss
            
            # Build description
            if ema_bullish: reasons.append("تحالف المتوسطات المتحركة الأسية بشكل إيجابي")
            if ema_8_13_bull_cross: reasons.append("تقاطع إيجابي للمتوسطات المتحركة EMA 8 و EMA 13")
            if rsi_oversold: reasons.append("مؤشر القوة النسبية RSI في منطقة التشبع البيعي")
            if macd_bullish: reasons.append("تقاطع إيجابي لمؤشر MACD")
            if bb_lower_touch: reasons.append("لمس السعر للحد السفلي لمؤشر البولنجر باند")
            if hammer: reasons.append("ظهور نمط شمعة المطرقة")
            if bullish_engulfing: reasons.append("ظهور نمط شمعة الابتلاع الصاعد")
            if near_support: reasons.append(f"السعر بالقرب من مستوى دعم قوي عند {support_level:.4f}")
            
            description = " مع ".join(reasons)
            
        elif sell_points > buy_points and sell_points >= 30:
            signal = "بيع"
            confidence = min(sell_points, 100)
            entry_price = current_price
            
            # Calculate take profit and stop loss
            if near_support and support_level is not None:
                tp = support_level
            else:
                tp = round(entry_price * 0.99, 4)  # Default 1% take profit
                
            if near_resistance and resistance_level is not None:
                sl = resistance_level * 1.002  # Just above resistance
            else:
                sl = round(entry_price * 1.005, 4)  # Default 0.5% stop loss
            
            # Build description
            if ema_bearish: reasons.append("تحالف المتوسطات المتحركة الأسية بشكل سلبي")
            if ema_8_13_bear_cross: reasons.append("تقاطع سلبي للمتوسطات المتحركة EMA 8 و EMA 13")
            if rsi_overbought: reasons.append("مؤشر القوة النسبية RSI في منطقة التشبع الشرائي")
            if macd_bearish: reasons.append("تقاطع سلبي لمؤشر MACD")
            if bb_upper_touch: reasons.append("لمس السعر للحد العلوي لمؤشر البولنجر باند")
            if shooting_star: reasons.append("ظهور نمط شمعة النجمة الساقطة")
            if bearish_engulfing: reasons.append("ظهور نمط شمعة الابتلاع الهابط")
            if near_resistance: reasons.append(f"السعر بالقرب من مستوى مقاومة قوي عند {resistance_level:.4f}")
            
            description = " مع ".join(reasons)
        
        # Calculate pips potential and risk-reward ratio
        if signal != "لا توجد إشارة":
            # For forex pairs, 1 pip is usually 0.0001
            pip_multiplier = 10000
            
            # For JPY pairs, 1 pip is 0.01
            if 'JPY' in symbol_name:
                pip_multiplier = 100
                
            # For crypto and indices, use percentage
            if 'BTC' in symbol_name or 'ETH' in symbol_name or 'NAS100' in symbol_name or 'S&P500' in symbol_name or 'DOW' in symbol_name:
                if signal == "شراء":
                    pips_potential = round((tp - entry_price) / entry_price * 100, 2)  # Percentage
                else:
                    pips_potential = round((entry_price - tp) / entry_price * 100, 2)  # Percentage
            else:
                if signal == "شراء":
                    pips_potential = round((tp - entry_price) * pip_multiplier, 1)
                else:
                    pips_potential = round((entry_price - tp) * pip_multiplier, 1)
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - sl)
            reward = abs(entry_price - tp)
            risk_reward = round(reward / risk, 2)
        
        return {
            'symbol': symbol_name,
            'timeframe': timeframe,
            'signal': signal,
            'entry_price': entry_price,
            'take_profit': tp,
            'stop_loss': sl,
            'description': description,
            'confidence': confidence,
            'pips_potential': pips_potential,
            'risk_reward': risk_reward,
            'current_price': current_price
        }

# Initialize session state for auto-refresh
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.datetime.now()

# Initialize session state for timeframe selection
if 'global_timeframe' not in st.session_state:
    st.session_state.global_timeframe = '1م'

# Main app layout
def main():
    # Header
    st.title("مؤشر سكالبينج متطور")
    
    # Sidebar
    st.sidebar.title("الإعدادات")
    
    # Global timeframe selection
    st.sidebar.subheader("الإطار الزمني")
    global_timeframe = st.sidebar.selectbox(
        "اختر الإطار الزمني لجميع الأزواج",
        list(timeframes.keys()),
        index=list(timeframes.keys()).index(st.session_state.global_timeframe)
    )
    st.session_state.global_timeframe = global_timeframe
    
    # Symbol selection (multi-select)
    st.sidebar.subheader("الأزواج")
    selected_symbols = st.sidebar.multiselect(
        "اختر الأزواج",
        list(symbols.keys()),
        default=["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
    )
    
    # Period selection
    st.sidebar.subheader("الفترة الزمنية")
    period = st.sidebar.selectbox(
        "اختر الفترة الزمنية",
        ["1 يوم", "5 أيام", "1 أسبوع", "1 شهر"],
        index=0
    )
    
    # Map period selection to yfinance format
    period_map = {
        "1 يوم": "1d",
        "5 أيام": "5d",
        "1 أسبوع": "1wk",
        "1 شهر": "1mo"
    }
    
    # Show only best opportunities option
    st.sidebar.subheader("خيارات العرض")
    show_best_only = st.sidebar.checkbox("عرض أفضل الفرص فقط", value=False)
    
    # Auto-refresh toggle
    st.sidebar.subheader("التحديث التلقائي")
    auto_refresh = st.sidebar.checkbox("تفعيل التحديث التلقائي كل 60 ثانية", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    # Show usage guide in sidebar
    with st.sidebar.expander("عرض دليل الاستخدام", expanded=False):
        st.markdown("""
        ### كيفية استخدام المؤشر
        
        1. **اختر الإطار الزمني** الذي ترغب في استخدامه لجميع الأزواج
        2. **اختر الأزواج** التي ترغب في تحليلها
        3. **حدد الفترة الزمنية** للبيانات التاريخية
        4. **اختر خيارات العرض** حسب تفضيلاتك
        5. **فعّل/أوقف التحديث التلقائي** حسب الحاجة
        
        ### فهم نطاقات الثقة
        
        * **95-100%**: ثقة عالية جداً، إشارة قوية للغاية
        * **85-89%**: ثقة عالية، إشارة قوية
        * **75-79%**: ثقة جيدة، إشارة مناسبة
        * **60-64%**: ثقة متوسطة، تحتاج لتأكيد إضافي
        * **أقل من 50%**: ثقة منخفضة، يفضل تجنبها
        """)
    
    # Main content
    if not selected_symbols:
        st.warning("الرجاء اختيار زوج واحد على الأقل للتحليل")
    else:
        # Initialize signal generator
        signal_gen = ScalpingSignalGenerator()
        
        # Last update indicator
        current_time = datetime.datetime.now()
        st.session_state.last_update = current_time
        st.markdown(f'<div class="last-update">آخر تحديث: {current_time.strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
        
        # Store results for sorting
        all_results = []
        
        # Process each selected symbol
        for symbol_name in selected_symbols:
            # Fetch data
            symbol_code = symbols[symbol_name]
            timeframe_code = timeframes[global_timeframe]
            period_code = period_map[period]
            
            df, error = fetch_data(symbol_code, period_code, timeframe_code)
            
            if error:
                st.error(error)
            else:
                # Generate signal
                result = signal_gen.analyze_timeframe(df, global_timeframe, symbol_name)
                all_results.append(result)
        
        # Filter results if show_best_only is checked
        if show_best_only:
            display_results = [r for r in all_results if r['signal'] != "لا توجد إشارة"]
        else:
            display_results = all_results
        
        # Sort results by confidence
        display_results = sorted(display_results, key=lambda x: x['confidence'], reverse=True)
        
        # Display results in a professional table
        if display_results:
            st.markdown('<div class="section-header"><h2>إشارات التداول</h2></div>', unsafe_allow_html=True)
            
            # Create DataFrame for display
            data = []
            for result in display_results:
                # Signal class and HTML
                if result['signal'] == "شراء":
                    signal_html = f'<span class="signal-buy">{result["signal"]}</span> <span class="timeframe-badge">{result["timeframe"]}</span>'
                elif result['signal'] == "بيع":
                    signal_html = f'<span class="signal-sell">{result["signal"]}</span> <span class="timeframe-badge">{result["timeframe"]}</span>'
                else:
                    signal_html = f'<span class="signal-neutral">{result["signal"]}</span> <span class="timeframe-badge">{result["timeframe"]}</span>'
                
                # Confidence range and color
                confidence_range, confidence_color = get_confidence_range(result['confidence'])
                confidence_html = f'<span class="confidence-badge" style="background-color:{confidence_color}">{confidence_range}</span>'
                
                # Format values
                entry_price = f"{result['entry_price']:.5f}" if result['entry_price'] is not None else "-"
                take_profit = f'<span class="profit-value">{result["take_profit"]:.5f}</span>' if result['take_profit'] is not None else "-"
                stop_loss = f'<span class="loss-value">{result["stop_loss"]:.5f}</span>' if result['stop_loss'] is not None else "-"
                pips_potential = f'<span class="profit-value">{result["pips_potential"]}</span>' if result['pips_potential'] != 0 else "-"
                risk_reward = f"{result['risk_reward']}" if result['risk_reward'] != 0 else "-"
                
                data.append({
                    "الزوج": f'<span class="symbol-column">{result["symbol"]}</span>',
                    "السعر الحالي": f'<span class="price-column">{result["current_price"]:.5f}</span>',
                    "الإشارة": signal_html,
                    "نسبة الثقة": confidence_html,
                    "سعر الدخول": entry_price,
                    "هدف الربح": take_profit,
                    "وقف الخسارة": stop_loss,
                    "النقاط المحتملة": pips_potential,
                    "نسبة المخاطرة/المكافأة": risk_reward
                })
            
            # Convert to DataFrame
            df_display = pd.DataFrame(data)
            
            # Display as HTML table
            st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Display best opportunities section
            valid_results = [r for r in all_results if r['signal'] != "لا توجد إشارة"]
            if valid_results:
                st.markdown('<div class="section-header"><h2>أفضل الفرص</h2></div>', unsafe_allow_html=True)
                
                # Sort by confidence
                best_results = sorted(valid_results, key=lambda x: x['confidence'], reverse=True)[:3]  # Top 3
                
                # Create DataFrame for best opportunities
                best_data = []
                for result in best_results:
                    # Signal class and HTML
                    if result['signal'] == "شراء":
                        signal_html = f'<span class="signal-buy">{result["signal"]}</span> <span class="timeframe-badge">{result["timeframe"]}</span>'
                    else:
                        signal_html = f'<span class="signal-sell">{result["signal"]}</span> <span class="timeframe-badge">{result["timeframe"]}</span>'
                    
                    # Confidence range and color
                    confidence_range, confidence_color = get_confidence_range(result['confidence'])
                    confidence_html = f'<span class="confidence-badge" style="background-color:{confidence_color}">{confidence_range}</span>'
                    
                    # Format values
                    entry_price = f"{result['entry_price']:.5f}" if result['entry_price'] is not None else "-"
                    take_profit = f'<span class="profit-value">{result["take_profit"]:.5f}</span>' if result['take_profit'] is not None else "-"
                    stop_loss = f'<span class="loss-value">{result["stop_loss"]:.5f}</span>' if result['stop_loss'] is not None else "-"
                    pips_potential = f'<span class="profit-value">{result["pips_potential"]}</span>' if result['pips_potential'] != 0 else "-"
                    risk_reward = f"{result['risk_reward']}" if result['risk_reward'] != 0 else "-"
                    
                    best_data.append({
                        "الزوج": f'<span class="symbol-column">{result["symbol"]}</span>',
                        "السعر الحالي": f'<span class="price-column">{result["current_price"]:.5f}</span>',
                        "الإشارة": signal_html,
                        "نسبة الثقة": confidence_html,
                        "سعر الدخول": entry_price,
                        "هدف الربح": take_profit,
                        "وقف الخسارة": stop_loss,
                        "النقاط المحتملة": pips_potential,
                        "نسبة المخاطرة/المكافأة": risk_reward
                    })
                
                # Convert to DataFrame
                df_best = pd.DataFrame(best_data)
                
                # Display as HTML table
                st.write(df_best.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("لا توجد إشارات صالحة حالياً")
        else:
            st.info("لا توجد نتائج للعرض. حاول تغيير الإعدادات أو إضافة المزيد من الأزواج.")
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(1)  # Small delay to prevent excessive CPU usage
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()
