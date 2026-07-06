import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import ccxt
import pandas as pd
import ta
import requests
import time
from datetime import datetime

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

TIMEFRAME = "1h"
CANDLES_LIMIT = 500  # نحتاج 200+ شمعة لـ EMA 200

SYMBOLS = [
    "BTC/USDT"
]

DEFAULT_IMAGE = "https://t.me/PYTHON_SIGNALS_BS/38"

# Scalp System Parameters
RSI_LENGTH = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
EMA_FAST = 150
EMA_SLOW = 200
EMA_FAR_THRESHOLD = 1.5   # max distance between EMA150 and EMA200 (%)
MAX_EMA200_DIST = 3.0      # max distance between price and EMA200 (%)
COOLDOWN_BARS = 10          # 10 hours on 1H

# TP/SL
SL_PCT = 1.5
TP1_PCT = 1.0
TP2_PCT = 3.0
TP3_PCT = 4.0


def get_decimals(price):
    if price > 100:
        return 2
    elif price > 1:
        return 3
    elif price > 0.01:
        return 5
    else:
        return 8


def send_signal(coin_name, direction, entry, tp1, tp2, tp3, sl,
                rsi, ema150, ema200, ema_distance, price_ema200_dist, image_url):
    """Send scalping signal to Telegram with beautiful format."""

    if direction == "LONG":
        header_icon = "🟢"
        trend_icon = "📈"
        dir_text = "LONG  ▲"
        trend_label = "BULLISH (EMA150 > EMA200)"
    else:
        header_icon = "🔴"
        trend_icon = "📉"
        dir_text = "SHORT  ▼"
        trend_label = "BEARISH (EMA150 < EMA200)"

    now = datetime.now().strftime('%Y-%m-%d  %H:%M')

    text = (
        f"╔═══════════════════════════════════════╗\n"
        f"║  {header_icon}  SCALP SIGNALS  {header_icon}  ║\n"
        f"╠═══════════════════════════════════════╣\n"
        f"\n"
        f"  {trend_icon}  <b>{coin_name}</b>  —  <code>{dir_text}</code>\n"
        f"\n"
        f"  ⏰  {now}\n"
        f"  🧠  <b>Strategy:</b>  <code>Scalp (RSI + EMA Trend)</code>\n"
        f"  📊  <b>Trend:</b>  <i>{trend_label}</i>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🎯  <b>Entry:</b>  <code>{entry}</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🟩  <b>TP1 (50%):</b>  <code>{tp1}</code>  <i>(+{TP1_PCT}%)</i>\n"
        f"  🟦  <b>TP2 (25%):</b>  <code>{tp2}</code>  <i>(+{TP2_PCT}%)</i>\n"
        f"  🟪  <b>TP3 (25%):</b>  <code>{tp3}</code>  <i>(+{TP3_PCT}%)</i>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🛑  <b>Stop Loss:</b>  <code>{sl}</code>  <i>(-{SL_PCT}% → BE after TP1)</i>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📊  <b>Indicators:</b>\n"
        f"  ┆  RSI (14):  <code>{rsi:.1f}</code>\n"
        f"  ┆  EMA 150/200 Dist:  <code>{ema_distance:.2f}%</code>\n"
        f"  ┆  Price/EMA200 Dist:  <code>{price_ema200_dist:.2f}%</code>\n"
        f"\n"
        f"╚═══════════════════════════════════════╝\n"
        f"\n"
        f"  <i>⚠️ Trade responsibly — Not financial advice</i>\n"
        f"  <i>⏱ Timeframe: 1H  |  🤖 Dr Python Scalp Bot</i>"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendAnimation"
    payload = {
        "chat_id": CHANNEL_ID,
        "animation": image_url,
        "caption": text,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.json().get('ok'):
            print(f"  ✅ Signal sent: {coin_name} {direction}")
        else:
            print(f"  ❌ TELEGRAM ERROR: {response.json().get('description')}")
    except Exception as e:
        print(f"  ❌ Network error: {e}")


def analyze_and_trade():
    print("=" * 55)
    print("  🤖 SCALP BOT STARTED — 1H Timeframe")
    print("  RSI + EMA 150/200 Trend Filter")
    print("=" * 55)

    exchange = ccxt.mexc()

    for symbol in SYMBOLS:
        try:
            print(f"\n  Analyzing {symbol}...", end=" ")

            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES_LIMIT)

            if len(ohlcv) < 250:
                print("SKIP (not enough data)")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # ─── RSI (14) ───
            df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_LENGTH)

            # ─── EMA 150 & EMA 200 ───
            df['ema150'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

            # نحتاج على الأقل 200 شمعة بعد حساب EMA
            # نبدأ الفحص من الشموع بعد الـ 200
            start_idx = max(200, len(df) - 50)  # نفحص آخر 50 شمعة

            # ─── Current values ───
            curr_close = df['close'].iloc[-1]
            curr_rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            curr_ema150 = df['ema150'].iloc[-1]
            curr_ema200 = df['ema200'].iloc[-1]

            if pd.isna(curr_rsi) or pd.isna(curr_ema150) or pd.isna(curr_ema200):
                print("SKIP (NaN in indicators)")
                continue

            # ─── EMA 150 vs 200 Trend ───
            ema150_above_ema200 = curr_ema150 > curr_ema200
            ema150_below_ema200 = curr_ema150 < curr_ema200

            # ─── EMA Distance Filter ───
            ema_distance = abs((curr_ema150 - curr_ema200) / curr_ema200) * 100
            is_ema_far = ema_distance >= EMA_FAR_THRESHOLD

            if is_ema_far:
                print(f"SKIP (EMA distance {ema_distance:.2f}% >= {EMA_FAR_THRESHOLD}%)")
                continue

            # ─── Price near EMA 200 ───
            price_ema200_dist = abs((curr_close - curr_ema200) / curr_ema200) * 100
            is_near_ema200 = price_ema200_dist <= MAX_EMA200_DIST

            if not is_near_ema200:
                print(f"SKIP (Price/EMA200 distance {price_ema200_dist:.2f}%)")
                continue

            # ─── RSI Cross Detection ───
            prev_rsi_up = prev_rsi > RSI_OVERBOUGHT
            curr_rsi_up = curr_rsi > RSI_OVERBOUGHT
            buy_base = curr_rsi_up and not prev_rsi_up  # cross above 65

            prev_rsi_down = prev_rsi < RSI_OVERSOLD
            curr_rsi_down = curr_rsi < RSI_OVERSOLD
            sell_base = curr_rsi_down and not prev_rsi_down  # cross below 35

            # ─── Cooldown Check (last 10 candles) ───
            # نتأكد أنه لم يحدث إشارة شراء أو بيع في آخر 10 شموع
            recent = df.iloc[-(COOLDOWN_BARS + 1):-1]

            recent_rsi = recent['rsi']
            recent_close = recent['close']
            recent_ema150 = recent['ema150']
            recent_ema200 = recent['ema200']

            cooldown_buy_hit = False
            cooldown_sell_hit = False

            for idx in range(len(recent) - 1):
                r = recent_rsi.iloc[idx]
                r_prev = recent_rsi.iloc[idx - 1] if idx > 0 else r
                c = recent_close.iloc[idx]
                e150 = recent_ema150.iloc[idx]
                e200 = recent_ema200.iloc[idx]

                if not pd.isna(r) and not pd.isna(r_prev):
                    # Check buy cooldown
                    if (r > RSI_OVERBOUGHT and r_prev <= RSI_OVERBOUGHT
                            and c > e150 and e150 > e200):
                        cooldown_buy_hit = True

                    # Check sell cooldown
                    if (r < RSI_OVERSOLD and r_prev >= RSI_OVERSOLD
                            and c < e150 and e150 < e200):
                        cooldown_sell_hit = True

            # ─── BUY Signal ───
            if buy_base and ema150_above_ema200:
                if curr_close > curr_ema150:
                    if not cooldown_buy_hit:
                        decimals = get_decimals(curr_close)
                        entry = round(curr_close, decimals)
                        tp1 = round(entry * (1 + TP1_PCT / 100), decimals)
                        tp2 = round(entry * (1 + TP2_PCT / 100), decimals)
                        tp3 = round(entry * (1 + TP3_PCT / 100), decimals)
                        sl = round(entry * (1 - SL_PCT / 100), decimals)

                        print(f"🟢 BUY SIGNAL! RSI: {curr_rsi:.1f}")

                        send_signal(
                            symbol, "LONG", str(entry),
                            str(tp1), str(tp2), str(tp3), str(sl),
                            curr_rsi, curr_ema150, curr_ema200,
                            ema_distance, price_ema200_dist, DEFAULT_IMAGE
                        )
                        time.sleep(2)
                    else:
                        print("SKIP (buy cooldown active)")
                else:
                    print("SKIP (price below EMA 150)")
            # ─── SELL Signal ───
            elif sell_base and ema150_below_ema200:
                if curr_close < curr_ema150:
                    if not cooldown_sell_hit:
                        decimals = get_decimals(curr_close)
                        entry = round(curr_close, decimals)
                        tp1 = round(entry * (1 - TP1_PCT / 100), decimals)
                        tp2 = round(entry * (1 - TP2_PCT / 100), decimals)
                        tp3 = round(entry * (1 - TP3_PCT / 100), decimals)
                        sl = round(entry * (1 + SL_PCT / 100), decimals)

                        print(f"🔴 SELL SIGNAL! RSI: {curr_rsi:.1f}")

                        send_signal(
                            symbol, "SHORT", str(entry),
                            str(tp1), str(tp2), str(tp3), str(sl),
                            curr_rsi, curr_ema150, curr_ema200,
                            ema_distance, price_ema200_dist, DEFAULT_IMAGE
                        )
                        time.sleep(2)
                    else:
                        print("SKIP (sell cooldown active)")
                else:
                    print("SKIP (price above EMA 150)")
            else:
                print(f"No signal (RSI: {curr_rsi:.1f})")

        except Exception as e:
            print(f"❌ ERROR: {e}")

    print("\n" + "=" * 55)
    print("  ✅ Scan completed.")
    print("=" * 55)


if __name__ == "__main__":
    print("🤖 Scalp Bot started...")
    analyze_and_trade()