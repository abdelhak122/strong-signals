import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        background-color: #F8F9FA;
    }
    
    h1, h2, h3 {
        color: #8C52FF;
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
    
    .card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .confidence-range {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
    }
    
    /* Confidence range colors */
    .range-95-100 { background-color: #8C52FF; }
    .range-90-94 { background-color: #5271FF; }
    .range-85-89 { background-color: #52B5FF; }
    .range-80-84 { background-color: #52FFD0; }
    .range-75-79 { background-color: #52FF7D; }
    .range-70-74 { background-color: #9DFF52; }
    .range-65-69 { background-color: #D6FF52; }
    .range-60-64 { background-color: #FFE252; }
    .range-55-59 { background-color: #FFBD52; }
    .range-50-54 { background-color: #FF8652; }
    .range-45-49 { background-color: #FF5252; }
    
    /* Alert styles */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-right: 4px solid;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-color: #52B5FF;
        color: #0c5460;
    }
    
    .alert-success {
        background-color: #d4edda;
        border-color: #52FF7D;
        color: #155724;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border-color: #FFE252;
        color: #856404;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border-color: #FF5252;
        color: #721c24;
    }
    
    /* Fix for RTL in dataframes */
    .dataframe {
        direction: rtl;
    }
    
    /* Fix for plotly charts */
    .js-plotly-plot {
        direction: ltr;
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
    '1 دقيقة': '1m',
    '5 دقائق': '5m',
    '15 دقيقة': '15m'
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
    else:
        return "أقل من 45%", '#CCCCCC'  # Gray

# Data fetching function with error handling and retries
@st.cache_data(ttl=300)  # Cache for 5 minutes
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
            'signal': signal,
            'entry_price': entry_price,
            'take_profit': tp,
            'stop_loss': sl,
            'description': description,
            'confidence': confidence,
            'pips_potential': pips_potential,
            'risk_reward': risk_reward
        }

# Function to plot chart
def plot_chart(df, symbol_name, timeframe):
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{symbol_name} - {timeframe}", "حجم التداول"))
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="شموع السعر",
        increasing_line_color='#52FF7D',
        decreasing_line_color='#FF5252'
    ), row=1, col=1)
    
    # Add EMA indicators
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA_8'],
        name="EMA 8",
        line=dict(color='#5271FF', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA_21'],
        name="EMA 21",
        line=dict(color='#FF8652', width=1)
    ), row=1, col=1)
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Upper_Band'],
        name="البولنجر العلوي",
        line=dict(color='#D6FF52', width=1, dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Lower_Band'],
        name="البولنجر السفلي",
        line=dict(color='#D6FF52', width=1, dash='dash')
    ), row=1, col=1)
    
    # Add volume bars
    colors = ['#52FF7D' if row['Close'] >= row['Open'] else '#FF5252' for _, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name="الحجم",
        marker_color=colors
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"تحليل {symbol_name} على الإطار الزمني {timeframe}",
        xaxis_title="التاريخ",
        yaxis_title="السعر",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="السعر", row=1, col=1)
    fig.update_yaxes(title_text="الحجم", row=2, col=1)
    
    return fig

# Main app layout
def main():
    # Header
    st.title("مؤشر سكالبينج متطور")
    st.markdown('<div class="alert alert-info">أداة قوية للتداول في الأسواق المالية المتعددة</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("الإعدادات")
    
    # Symbol selection
    symbol_name = st.sidebar.selectbox(
        "اختر الرمز",
        list(symbols.keys())
    )
    
    # Timeframe selection
    timeframe_name = st.sidebar.selectbox(
        "اختر الإطار الزمني",
        list(timeframes.keys())
    )
    
    # Period selection
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
    
    # Add analyze button
    analyze_button = st.sidebar.button("تحليل السوق", key="analyze")
    
    # Show usage guide in sidebar
    with st.sidebar.expander("عرض دليل الاستخدام", expanded=False):
        st.markdown("""
        ### كيفية استخدام المؤشر
        
        1. **اختر الرمز** من القائمة المنسدلة
        2. **حدد الإطار الزمني** المناسب لاستراتيجيتك
        3. **اختر الفترة الزمنية** للتحليل
        4. انقر على زر **تحليل السوق**
        5. راجع النتائج والإشارات في الصفحة الرئيسية
        
        ### فهم نطاقات الثقة
        
        * **95-100%**: ثقة عالية جداً، إشارة قوية للغاية
        * **85-89%**: ثقة عالية، إشارة قوية
        * **75-79%**: ثقة جيدة، إشارة مناسبة
        * **60-64%**: ثقة متوسطة، تحتاج لتأكيد إضافي
        * **أقل من 50%**: ثقة منخفضة، يفضل تجنبها
        
        ### الأسواق المدعومة
        
        * **الفوركس**: EUR/USD, GBP/USD, USD/JPY, AUD/USD
        * **السلع**: XAU/USD (الذهب), XAG/USD (الفضة), OIL/USD (النفط)
        * **العملات الرقمية**: BTC/USD, ETH/USD
        * **المؤشرات**: NAS100, S&P500, DOW
        """)
    
    # Main content
    if analyze_button:
        with st.spinner('جاري تحليل البيانات...'):
            # Fetch data
            symbol_code = symbols[symbol_name]
            timeframe_code = timeframes[timeframe_name]
            period_code = period_map[period]
            
            df, error = fetch_data(symbol_code, period_code, timeframe_code)
            
            if error:
                st.error(error)
            else:
                # Initialize signal generator
                signal_gen = ScalpingSignalGenerator()
                
                # Add indicators for chart
                ti = TechnicalIndicators()
                df['EMA_8'] = ti.ema(df, 8)
                df['EMA_21'] = ti.ema(df, 21)
                df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = ti.bollinger_bands(df)
                
                # Generate signal
                result = signal_gen.analyze_timeframe(df, timeframe_name, symbol_name)
                
                # Display chart
                fig = plot_chart(df, symbol_name, timeframe_name)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display signal results
                st.subheader("نتائج التحليل")
                
                # Create columns for signal display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # Signal type with color
                    if result['signal'] == "شراء":
                        st.markdown(f'<h3 style="color:#52FF7D;">إشارة {result["signal"]}</h3>', unsafe_allow_html=True)
                    elif result['signal'] == "بيع":
                        st.markdown(f'<h3 style="color:#FF5252;">إشارة {result["signal"]}</h3>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<h3 style="color:#CCCCCC;">{result["signal"]}</h3>', unsafe_allow_html=True)
                    
                    # Only show details if there's a signal
                    if result['signal'] != "لا توجد إشارة":
                        # Confidence level
                        confidence_range, color = get_confidence_range(result['confidence'])
                        st.markdown(f'<p>مستوى الثقة: <span class="confidence-range" style="background-color:{color};">{confidence_range}</span> ({result["confidence"]}%)</p>', unsafe_allow_html=True)
                        
                        # Entry, TP, SL
                        st.markdown(f'<p>سعر الدخول: <strong>{result["entry_price"]:.4f}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p>مستوى جني الأرباح: <strong>{result["take_profit"]:.4f}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p>مستوى وقف الخسارة: <strong>{result["stop_loss"]:.4f}</strong></p>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # Only show details if there's a signal
                    if result['signal'] != "لا توجد إشارة":
                        # Description
                        st.markdown(f'<p><strong>وصف الإشارة:</strong> {result["description"]}</p>', unsafe_allow_html=True)
                        
                        # Pips potential
                        if 'BTC' in symbol_name or 'ETH' in symbol_name or 'NAS100' in symbol_name or 'S&P500' in symbol_name or 'DOW' in symbol_name:
                            st.markdown(f'<p>العائد المحتمل: <strong>{result["pips_potential"]}%</strong></p>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p>النقاط المحتملة: <strong>{result["pips_potential"]}</strong></p>', unsafe_allow_html=True)
                        
                        # Risk-reward ratio
                        st.markdown(f'<p>نسبة المخاطرة إلى العائد: <strong>1:{result["risk_reward"]}</strong></p>', unsafe_allow_html=True)
                        
                        # Current time
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.markdown(f'<p>وقت التحليل: {current_time}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p>لا توجد إشارة تداول في الوقت الحالي. حاول اختيار رمز آخر أو إطار زمني مختلف.</p>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display market data
                st.subheader("بيانات السوق")
                st.dataframe(df.tail(10).sort_index(ascending=False))
    else:
        # Welcome message when app first loads
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ## مرحباً بك في مؤشر سكالبينج متطور
        
        هذا المؤشر يوفر إشارات تداول دقيقة للسكالبينج في أسواق متعددة مع دعم لإطارات زمنية متعددة.
        
        ### المميزات الرئيسية:
        
        * تحليل متعدد الإطارات الزمنية (1 دقيقة، 5 دقائق، 15 دقيقة)
        * مؤشرات فنية متعددة (EMA, RSI, Bollinger Bands)
        * تحليل أنماط الشموع اليابانية
        * تحليل الحجم
        * كشف مستويات الدعم والمقاومة
        * عرض نطاق الثقة في الإشارات
        * حساب النقاط المحتملة
        * توصية بسعر الدخول
        * نسبة المخاطرة إلى العائد
        
        ### للبدء:
        
        استخدم القائمة الجانبية لاختيار الرمز والإطار الزمني، ثم انقر على زر "تحليل السوق".
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Features section
        st.subheader("المميزات الرئيسية")
        
        # Create three columns for features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
            ### تحليل متعدد الإطارات
            
            تحليل متزامن للإطارات الزمنية المختلفة يساعد في تأكيد الإشارات وزيادة دقة التوقعات.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
            ### مؤشرات فنية متقدمة
            
            مجموعة من المؤشرات الفنية المتقدمة تعمل معاً لتوفير إشارات دقيقة وموثوقة.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
            ### تحليل أنماط الشموع
            
            التعرف التلقائي على أنماط الشموع اليابانية الرئيسية وتحليل دلالاتها.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
