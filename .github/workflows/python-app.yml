import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

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
    '1m': '1m',
    '5m': '5m',
    '15m': '15m'
}

# Data fetching function with error handling and retries
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
                    return {"error": f"No data for symbol {symbol}"}
            
            # Ensure volume data is available (some forex data might not have volume)
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                df['Volume'] = 1000  # Assign default volume
            
            # Convert DataFrame to dictionary for JSON serialization
            result = df.reset_index().to_dict(orient='records')
            return {"data": result}
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            else:
                return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

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

# API routes
@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    return jsonify(symbols)

@app.route('/api/timeframes', methods=['GET'])
def get_timeframes():
    return jsonify(timeframes)

@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    symbol_name = request.args.get('symbol', 'EUR/USD')
    timeframe = request.args.get('timeframe', '5m')
    period = request.args.get('period', '1d')
    
    if symbol_name not in symbols:
        return jsonify({"error": "Invalid symbol"}), 400
    
    if timeframe not in timeframes:
        return jsonify({"error": "Invalid timeframe"}), 400
    
    symbol_code = symbols[symbol_name]
    result = fetch_data(symbol_code, period, timeframes[timeframe])
    
    return jsonify(result)

@app.route('/api/analyze', methods=['GET'])
def analyze_market():
    symbol_name = request.args.get('symbol', 'EUR/USD')
    timeframe = request.args.get('timeframe', '5m')
    
    if symbol_name not in symbols:
        return jsonify({"error": "Invalid symbol"}), 400
    
    if timeframe not in timeframes:
        return jsonify({"error": "Invalid timeframe"}), 400
    
    # This would normally call the analysis functions from the original script
    # For now, we'll return a sample analysis result
    confidence = np.random.randint(50, 100)
    confidence_range, color = get_confidence_range(confidence)
    
    signal_type = "شراء" if np.random.random() > 0.5 else "بيع"
    entry_price = round(np.random.uniform(1.0, 2000.0), 2)
    
    # Calculate TP and SL based on signal type
    if signal_type == "شراء":
        tp = round(entry_price * 1.01, 2)
        sl = round(entry_price * 0.995, 2)
        pips = round((tp - entry_price) * 10000, 1)
    else:
        tp = round(entry_price * 0.99, 2)
        sl = round(entry_price * 1.005, 2)
        pips = round((entry_price - tp) * 10000, 1)
    
    risk_reward = round(abs(entry_price - tp) / abs(entry_price - sl), 2)
    
    # Generate a sample description
    descriptions = [
        "تقاطع المتوسطات المتحركة الأسية EMA 8 و EMA 21 مع حجم تداول مرتفع",
        "مؤشر القوة النسبية RSI يشير إلى تشبع بيعي مع نمط شمعة إيجابي",
        "ارتداد من مستوى دعم قوي مع تباعد إيجابي في MACD",
        "كسر مستوى مقاومة رئيسي مع تأكيد من مؤشر البولنجر باند",
        "نمط شمعة ابتلاع صاعد مع تأكيد من مستويات فيبوناتشي"
    ]
    
    description = np.random.choice(descriptions)
    
    result = {
        "symbol": symbol_name,
        "timeframe": timeframe,
        "signal": signal_type,
        "entry_price": entry_price,
        "take_profit": tp,
        "stop_loss": sl,
        "description": description,
        "confidence": confidence,
        "confidence_range": confidence_range,
        "confidence_color": color,
        "pips_potential": pips,
        "risk_reward": risk_reward,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return jsonify(result)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
