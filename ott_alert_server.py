import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, List, Dict
import warnings
import pytz

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ott_alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize alerts history
alerts_history = []


def is_market_open(now=None):
    """Check if market is open (9:00 AM to 3:30 PM IST, Monday to Friday)"""
    if now is None:
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_stock_data(symbol: str, period: str = "5d", interval: str = "30m") -> Optional[pd.DataFrame]:
    """Fetch stock data from Yahoo Finance with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            logger.warning(f"No data found for symbol: {symbol}")
            return None
        if len(data) < 20:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} points")
            return None
        return data.dropna()
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


def calculate_var_function(src: pd.Series, length: int = 5) -> pd.Series:
    """Calculate VAR using Pine Script logic"""
    try:
        valpha = 2 / (length + 1)
        vud1 = pd.Series(np.where(src > src.shift(1), src - src.shift(1), 0), index=src.index)
        vdd1 = pd.Series(np.where(src < src.shift(1), src.shift(1) - src, 0), index=src.index)

        vUD = vud1.rolling(window=9, min_periods=1).sum()
        vDD = vdd1.rolling(window=9, min_periods=1).sum()

        denominator = vUD + vDD
        vCMO = np.where(denominator != 0, (vUD - vDD) / denominator, 0)
        vCMO = pd.Series(vCMO, index=src.index).fillna(0)

        VAR = pd.Series(index=src.index, dtype=float)
        VAR.iloc[0] = src.iloc[0]

        for i in range(1, len(src)):
            alpha_factor = valpha * abs(vCMO.iloc[i])
            VAR.iloc[i] = alpha_factor * src.iloc[i] + (1 - alpha_factor) * VAR.iloc[i - 1]

        return VAR
    except Exception as e:
        logger.error(f"Error in VAR calculation: {str(e)}")
        return pd.Series(index=src.index, dtype=float)


def calculate_ott(data: pd.DataFrame, length: int = 5, percent: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate OTT indicator"""
    try:
        if data is None or data.empty or len(data) < length:
            logger.warning("Insufficient data for OTT calculation")
            empty = pd.Series(index=data.index if data is not None else [], dtype=float)
            return empty, empty, empty

        src = data['Close'].copy()
        VAR = calculate_var_function(src, length)
        MAvg = VAR.copy()

        fark = MAvg * percent * 0.01
        longStop = MAvg - fark
        shortStop = MAvg + fark

        longStop_adj = pd.Series(index=data.index, dtype=float)
        shortStop_adj = pd.Series(index=data.index, dtype=float)
        dir_series = pd.Series(index=data.index, dtype=int)
        MT = pd.Series(index=data.index, dtype=float)
        OTT = pd.Series(index=data.index, dtype=float)

        longStop_adj.iloc[0] = longStop.iloc[0]
        shortStop_adj.iloc[0] = shortStop.iloc[0]
        dir_series.iloc[0] = 1

        for i in range(1, len(data)):
            if pd.notna(MAvg.iloc[i]) and pd.notna(longStop_adj.iloc[i - 1]):
                longStop_adj.iloc[i] = max(longStop.iloc[i], longStop_adj.iloc[i - 1]) if MAvg.iloc[i] > longStop_adj.iloc[i - 1] else longStop.iloc[i]
            else:
                longStop_adj.iloc[i] = longStop.iloc[i]

            if pd.notna(MAvg.iloc[i]) and pd.notna(shortStop_adj.iloc[i - 1]):
                shortStop_adj.iloc[i] = min(shortStop.iloc[i], shortStop_adj.iloc[i - 1]) if MAvg.iloc[i] < shortStop_adj.iloc[i - 1] else shortStop.iloc[i]
            else:
                shortStop_adj.iloc[i] = shortStop.iloc[i]

            if dir_series.iloc[i - 1] == -1 and pd.notna(MAvg.iloc[i]) and pd.notna(shortStop_adj.iloc[i - 1]) and MAvg.iloc[i] > shortStop_adj.iloc[i - 1]:
                dir_series.iloc[i] = 1
            elif dir_series.iloc[i - 1] == 1 and pd.notna(MAvg.iloc[i]) and pd.notna(longStop_adj.iloc[i - 1]) and MAvg.iloc[i] < longStop_adj.iloc[i - 1]:
                dir_series.iloc[i] = -1
            else:
                dir_series.iloc[i] = dir_series.iloc[i - 1]

        for i in range(len(data)):
            MT.iloc[i] = longStop_adj.iloc[i] if dir_series.iloc[i] == 1 else shortStop_adj.iloc[i]
            if pd.notna(MAvg.iloc[i]) and pd.notna(MT.iloc[i]):
                OTT.iloc[i] = MT.iloc[i] * (200 + percent) / 200 if MAvg.iloc[i] > MT.iloc[i] else MT.iloc[i] * (200 - percent) / 200
            else:
                OTT.iloc[i] = MT.iloc[i]

        return MAvg, OTT, dir_series

    except Exception as e:
        logger.error(f"Error in OTT calculation: {str(e)}")
        empty = pd.Series(index=data.index if data is not None else [], dtype=float)
        return empty, empty, empty


def detect_signals(MAvg: pd.Series, OTT: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Detect buy and sell signals"""
    try:
        if MAvg.empty or OTT.empty:
            empty = pd.Series([False] * len(MAvg), index=MAvg.index)
            return empty, empty

        OTT_shifted = OTT.shift(2)
        buy_signal = (MAvg > OTT_shifted) & (MAvg.shift(1) <= OTT_shifted.shift(1))
        sell_signal = (MAvg < OTT_shifted) & (MAvg.shift(1) >= OTT_shifted.shift(1))

        return buy_signal.fillna(False), sell_signal.fillna(False)

    except Exception as e:
        logger.error(f"Error in signal detection: {str(e)}")
        empty = pd.Series([False] * len(MAvg), index=MAvg.index)
        return empty, empty


def validate_email_settings(email_settings: dict) -> bool:
    """Validate email settings"""
    return all(email_settings.get(f) for f in ['email', 'password', 'recipient'])


def send_compiled_email_alert(signals: List[Dict], email_settings: dict) -> bool:
    """
    Send a single compiled email with all signals from the current scan.

    Args:
        signals: List of dicts with keys: symbol, signal_type, price
        email_settings: SMTP configuration dict
    """
    if not signals:
        logger.info("No signals to send in this scan.")
        return False

    if not validate_email_settings(email_settings):
        logger.warning("Email settings incomplete")
        return False

    try:
        scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        buy_signals = [s for s in signals if s['signal_type'] == 'BUY']
        sell_signals = [s for s in signals if s['signal_type'] == 'SELL']

        subject = f"OTT Scan Alert [{scan_time}] — {len(buy_signals)} BUY | {len(sell_signals)} SELL"

        # Build HTML body
        rows_html = ""
        for s in signals:
            color = "#1a7f37" if s['signal_type'] == 'BUY' else "#b91c1c"
            bg = "#d1fae5" if s['signal_type'] == 'BUY' else "#fee2e2"
            rows_html += f"""
            <tr style="background:{bg};">
                <td style="padding:8px 12px; font-weight:bold;">{s['symbol']}</td>
                <td style="padding:8px 12px; color:{color}; font-weight:bold;">{s['signal_type']}</td>
                <td style="padding:8px 12px;">₹{s['price']:.2f}</td>
            </tr>"""

        html_body = f"""
        <html><body style="font-family: Arial, sans-serif; color: #222;">
            <h2 style="color:#1e3a5f;">📊 OTT Strategy — Scan Summary</h2>
            <p><strong>Scan Time:</strong> {scan_time}</p>
            <p><strong>Total Signals:</strong> {len(signals)} &nbsp;|&nbsp;
               <span style="color:#1a7f37;">BUY: {len(buy_signals)}</span> &nbsp;|&nbsp;
               <span style="color:#b91c1c;">SELL: {len(sell_signals)}</span></p>

            <table border="1" cellspacing="0" cellpadding="0"
                   style="border-collapse:collapse; width:100%; max-width:500px; border-color:#ddd;">
                <thead>
                    <tr style="background:#1e3a5f; color:white;">
                        <th style="padding:8px 12px; text-align:left;">Symbol</th>
                        <th style="padding:8px 12px; text-align:left;">Signal</th>
                        <th style="padding:8px 12px; text-align:left;">Price</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>

            <p style="margin-top:20px; font-size:12px; color:#888;">
                This is an automated alert from your OTT trading system.
            </p>
        </body></html>
        """

        # Plain text fallback
        plain_rows = "\n".join(
            f"  {s['symbol']:<20} {s['signal_type']:<6} ₹{s['price']:.2f}"
            for s in signals
        )
        plain_body = f"""OTT Strategy — Scan Summary
Scan Time : {scan_time}
Signals   : {len(signals)} total | BUY: {len(buy_signals)} | SELL: {len(sell_signals)}

{'Symbol':<20} {'Signal':<6} Price
{'-'*40}
{plain_rows}

This is an automated alert from your OTT trading system.
"""

        msg = MIMEMultipart("alternative")
        msg['From'] = email_settings['email']
        msg['To'] = email_settings['recipient']
        msg['Subject'] = subject
        msg.attach(MIMEText(plain_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port']) as server:
            server.starttls()
            server.login(email_settings['email'], email_settings['password'])
            server.sendmail(email_settings['email'], email_settings['recipient'], msg.as_string())

        logger.info(f"Compiled email sent: {len(buy_signals)} BUY, {len(sell_signals)} SELL signals.")
        return True

    except Exception as e:
        logger.error(f"Email sending failed: {str(e)}")
        return False


def add_alert_to_history(symbol: str, signal_type: str, price: float):
    """Add alert to history with deduplication (5-minute window)"""
    global alerts_history
    alert = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'signal': signal_type,
        'price': price
    }
    current_time = datetime.now()
    for existing in alerts_history[:5]:
        existing_time = datetime.strptime(existing['timestamp'], '%Y-%m-%d %H:%M:%S')
        if (existing['symbol'] == symbol and
                existing['signal'] == signal_type and
                (current_time - existing_time).total_seconds() < 300):
            return
    alerts_history.insert(0, alert)
    if len(alerts_history) > 100:
        alerts_history = alerts_history[:100]


def main():
    """Main monitoring loop"""
    watchlist =[
    "^NSEI",
    "^BSESN",
    "^NSEBANK",
    "AARTIIND.NS",
    "ABB.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASHOKLEY.NS",
    "ASTRAL.NS",
    "AUROPHARMA.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "BALKRISIND.NS",
    "BDL.NS",
    "BEL.NS",
    "BHARATFORG.NS",
    "BHARTIARTL.NS",
    "BSE.NS",
    "CAMS.NS",
    "CDSL.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "COFORGE.NS",
    "COLPAL.NS",
    "CUMMINSIND.NS",
    "DIVISLAB.NS",
    "DIXON.NS",
    "DLF.NS",
    "DMART.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "FORTIS.NS",
    "GODREJCP.NS",
    "GODREJPROP.NS",
    "GRASIM.NS",
    "HAL.NS",
    "HAVELLS.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HEROMOTOCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "IEX.NS",
    "INDIGO.NS",
    "INFY.NS",
    "IOC.NS",
    "IRCTC.NS",
    "ITC.NS",
    "JINDALSTEL.NS",
    "JIOFIN.NS",
    "JSL.NS",
    "JUBLFOOD.NS",
    "KOTAKBANK.NS",
    "LAURUSLABS.NS",
    "LODHA.NS",
    "LT.NS",
    "M&M.NS",
    "MARICO.NS",
    "MARUTI.NS",
    "MAZDOCK.NS",
    "MCX.NS",
    "MGL.NS",
    "MOTHERSON.NS",
    "MPHASIS.NS",
    "NAUKRI.NS",
    "OBEROIRLTY.NS",
    "OFSS.NS",
    "OIL.NS",
    "ONGC.NS",
    "PAGEIND.NS",
    "PAYTM.NS",
    "PIRAMALFIN.NS",
    "PPLPHARMA.NS",
    "PERSISTENT.NS",
    "PFC.NS",
    "PIDILITIND.NS",
    "PIIND.NS",
    "PNBHOUSING.NS",
    "POLYCAB.NS",
    "POONAWALLA.NS",
    "SBICARD.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "SUPREMEIND.NS",
    "TATACONSUM.NS",
    "TMCV.NS",
    "TMPV.NS",
    "TATAPOWER.NS",
    "TCS.NS",
    "TITAN.NS",
    "TRENT.NS",
    "TVSMOTOR.NS",
    "VBL.NS",
    "VEDL.NS",
    "VOLTAS.NS",
    "WIPRO.NS",
]

    email_settings = {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "email": "rakesh.msr@gmail.com",
        "password": "oick mmdo veiv dujt",
        "recipient": "niftystockalert@gmail.com",
        "enabled": True
    }

    ott_period = 5
    ott_percent = 1.5
    scan_interval = 900  # seconds

    logger.info("Starting OTT Alert Server")

    while True:
        now = datetime.now(pytz.timezone('Asia/Kolkata'))

        # Wait for market to open
        while not is_market_open(now):
            logger.info("Market is closed. Waiting for market to open...")
            time.sleep(900)
            now = datetime.now(pytz.timezone('Asia/Kolkata'))

        if not is_market_open(now):
            continue

        logger.info(f"Starting new scan at {now.strftime('%Y-%m-%d %H:%M:%S')}")

        # Collect all signals during this scan
        scan_signals: List[Dict] = []

        for symbol in watchlist:
            current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
            if not is_market_open(current_time):
                logger.info("Market has closed during scan. Stopping current scan.")
                break

            try:
                data = get_stock_data(symbol, period="5d", interval="30m")

                if data is not None and len(data) > 20:
                    MAvg, OTT, dir_series = calculate_ott(data, ott_period, ott_percent)
                    buy_signals, sell_signals = detect_signals(MAvg, OTT)

                    recent_buy = buy_signals.iloc[-1] and not buy_signals.iloc[-2]
                    recent_sell = sell_signals.iloc[-1] and not sell_signals.iloc[-2]
                    current_price = data['Close'].iloc[-1]

                    if recent_buy and not recent_sell:
                        logger.info(f"BUY Signal — {symbol} at ₹{current_price:.2f}")
                        add_alert_to_history(symbol, "BUY", current_price)
                        scan_signals.append({"symbol": symbol, "signal_type": "BUY", "price": current_price})

                    elif recent_sell and not recent_buy:
                        logger.info(f"SELL Signal — {symbol} at ₹{current_price:.2f}")
                        add_alert_to_history(symbol, "SELL", current_price)
                        scan_signals.append({"symbol": symbol, "signal_type": "SELL", "price": current_price})
                else:
                    logger.warning(f"Unable to fetch data for {symbol}")

                time.sleep(0.5)  # Prevent rate limiting

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")

        # Send ONE compiled email for all signals found in this scan
        end_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        if scan_signals and email_settings.get('enabled', False):
            send_compiled_email_alert(scan_signals, email_settings)
        else:
            logger.info("No new signals in this scan — no email sent.")

        if not is_market_open(end_time):
            logger.info(f"Scan completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Market has closed.")
            continue

        logger.info(f"Scan completed. Found {len(scan_signals)} signal(s). Sleeping {scan_interval}s...")
        time.sleep(scan_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down OTT Alert Server")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")