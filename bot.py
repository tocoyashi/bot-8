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

TIMEFRAME = "15m"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "TRX/USDT", "POL/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT",
    "ATOM/USDT", "XLM/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT",
    "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT", "FIL/USDT",
    "AAVE/USDT", "GRT/USDT", "PEPE/USDT", "QNT/USDT", "FET/USDT"
]

DEFAULT_IMAGE = "https://t.me/PYTHON_SIGNALS_BS/38"


def get_decimals(price):
    if price > 100:
        return 2
    elif price > 1:
        return 3
    elif price > 0.01:
        return 5
    else:
        return 8


def get_signal_strength(ema_diff_pct, macd_val, rsi, volume_ratio):
    """
    تقييم قوة الإشارة من 1 إلى 5 نجوم بناءً على:
    - فرق EMA (كلما كان التقاطع أقوى = أفضل)
    - قوة MACD
    - موقع RSI (بعيد عن التشبع = أفضل)
    - حجم التداول (أعلى من المتوسط = أفضل)
    """
    score = 0

    # EMA diff contribution (0-1.5 points)
    if abs(ema_diff_pct) > 0.5:
        score += 1.5
    elif abs(ema_diff_pct) > 0.3:
        score += 1.0
    elif abs(ema_diff_pct) > 0.15:
        score += 0.5

    # MACD contribution (0-1.5 points)
    if abs(macd_val) > 0:
        score += min(1.5, abs(macd_val) * 10)

    # RSI contribution (0-1.0 points)
    if 40 <= rsi <= 60:
        score += 1.0
    elif 30 <= rsi <= 70:
        score += 0.5

    # Volume contribution (0-1.0 points)
    if volume_ratio > 2.0:
        score += 1.0
    elif volume_ratio > 1.5:
        score += 0.75
    elif volume_ratio > 1.0:
        score += 0.5

    # Convert to stars (minimum 1 star, max 5)
    stars = max(1, min(5, round(score)))
    return stars


def get_rsi_zone(rsi):
    """تحديد منطقة RSI"""
    if rsi >= 70:
        return "🔴 تشبع شرائي (Overbought)"
    elif rsi >= 55:
        return "🟢 منطقة قوية (Bullish)"
    elif rsi >= 45:
        return "🟡 منطقة محايدة (Neutral)"
    elif rsi >= 30:
        return "🟠 منطقة ضعيفة (Bearish)"
    else:
        return "🔴 تشبع بيعي (Oversold)"


def get_volume_status(volume_ratio):
    """تحديد حالة حجم التداول"""
    if volume_ratio >= 2.5:
        return "🚀 حجم مرتفع جداً"
    elif volume_ratio >= 1.8:
        return "📈 حجم مرتفع"
    elif volume_ratio >= 1.2:
        return "📊 حجم فوق المتوسط"
    elif volume_ratio >= 0.8:
        return "➖ حجم عادي"
    else:
        return "📉 حجم ضعيف"


def send_crypto_signal(coin_name, direction, strategy, entry, leverage,
                       tp1, tp2, tp3, tp4, tp5, sl,
                       rsi, rsi_zone, volume_ratio, volume_status,
                       strength_stars, ema_diff_pct, image_url):
    """إرسال إشارة تداول بتنسيق جميل"""

    is_long = direction.lower() == "long"

    # إعدادات الألوان والرموز حسب الاتجاه
    if is_long:
        header_icon = "🟢"
        trend_icon = "📈"
        dir_text = "LONG  ▲"
        dir_color = "#00E676"
    else:
        header_icon = "🔴"
        trend_icon = "📉"
        dir_text = "SHORT  ▼"
        dir_color = "#FF1744"

    # نجوم القوة
    star_display = "⭐" * strength_stars + "☆" * (5 - strength_stars)
    strength_label = {
        1: "ضعيفة",
        2: "متوسطة",
        3: "جيدة",
        4: "قوية",
        5: "قوية جداً"
    }[strength_stars]

    # وقت الإشارة
    now = datetime.now().strftime('%Y-%m-%d  %H:%M')

    text = (
        f"╔═══════════════════════════════════════╗\n"
        f"║  {header_icon}  DR PYTHON SIGNALS  {header_icon}  ║\n"
        f"╠═══════════════════════════════════════╣\n"
        f"\n"
        f"  {trend_icon}  <b>{coin_name}</b>  —  <code>{dir_text}</code>\n"
        f"\n"
        f"  ⏰  {now}\n"
        f"\n"
        f"  🧠  <b>الاستراتيجية:</b>  <code>{strategy}</code>\n"
        f"  💪  <b>قوة الإشارة:</b>  {star_display}  <i>({strength_label})</i>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🎯  <b>الدخول (Entry):</b>  <code>{entry}</code>\n"
        f"  ⚡  <b>الرافعة (Leverage):</b>  <code>{leverage}x</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🟩  <b>TP1:</b>  <code>{tp1}</code>  <i>(+0.65%)</i>\n"
        f"  🟦  <b>TP2:</b>  <code>{tp2}</code>  <i>(+1.70%)</i>\n"
        f"  🟧  <b>TP3:</b>  <code>{tp3}</code>  <i>(+3.20%)</i>\n"
        f"  🟪  <b>TP4:</b>  <code>{tp4}</code>  <i>(+5.80%)</i>\n"
        f"  🟥  <b>TP5:</b>  <code>{tp5}</code>  <i>(+7.20%)</i>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🛑  <b>وقف الخسارة (SL):</b>  <code>{sl}</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📊  <b>المؤشرات الفنية:</b>\n"
        f"  ┆  RSI (14):  <code>{rsi:.1f}</code>  —  {rsi_zone}\n"
        f"  ┆  Volume:  <code>{volume_ratio:.2f}x</code>  —  {volume_status}\n"
        f"  ┆  EMA Diff:  <code>{ema_diff_pct:+.3f}%</code>\n"
        f"\n"
        f"╚═══════════════════════════════════════╝\n"
        f"\n"
        f"  <i>⚠️ تداول بمسؤولية — ليست نصيحة مالية</i>\n"
        f"  <i>🤖 Dr Python Cornix Signals</i>"
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
            print(f"✅ Signal sent: {coin_name} | {strategy} | Strength: {'⭐' * strength_stars}")
        else:
            print(f"❌ TELEGRAM ERROR for {coin_name}: {response.json().get('description')}")
    except Exception as e:
        print(f"❌ Network error: {e}")


def analyze_and_trade():
    print("=" * 50)
    print("🚀 Starting scan (15m) — EMA + MACD + RSI + Volume")
    print("=" * 50)

    exchange = ccxt.mexc()

    for symbol in SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # ─── EMA 9 & 21 ───
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

            curr_ema9 = df['ema_9'].iloc[-1]
            curr_ema21 = df['ema_21'].iloc[-1]
            prev_ema9 = df['ema_9'].iloc[-2]
            prev_ema21 = df['ema_21'].iloc[-2]

            ema_buy = (prev_ema9 < prev_ema21) and (curr_ema9 > curr_ema21)
            ema_sell = (prev_ema9 > prev_ema21) and (curr_ema9 < curr_ema21)
            ema_diff_pct = ((curr_ema9 - curr_ema21) / curr_ema21) * 100

            # ─── MACD ───
            macd_hist = ta.trend.macd_diff(df['close'])
            curr_macd = macd_hist.iloc[-1]
            prev_macd = macd_hist.iloc[-2]

            macd_buy = (prev_macd < 0) and (curr_macd > 0)
            macd_sell = (prev_macd > 0) and (curr_macd < 0)

            # ─── RSI (14) ───
            rsi_series = ta.momentum.rsi(df['close'], window=14)
            curr_rsi = rsi_series.iloc[-1]

            # ─── Volume Filter ───
            # مقارنة حجم الشمعة الحالية بمتوسط آخر 20 شمعة
            recent_volumes = df['volume'].iloc[-21:-1]  # آخر 20 شمعة (بدون الحالية)
            avg_volume = recent_volumes.mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # ─── Volume Threshold: نطلب أن يكون الحجم أعلى من المتوسط ───
            volume_confirmed = volume_ratio >= 1.0

            current_close = df['close'].iloc[-1]
            decimals = get_decimals(current_close)

            # ─── توليد الإشارات ───
            buy_signal = ema_buy or macd_buy
            sell_signal = ema_sell or macd_sell

            if buy_signal and volume_confirmed:
                # فلتر RSI للشراء: لا نشتري إذا كان RSI فوق 75 (تشبع شرائي)
                if curr_rsi < 75:
                    strategy_name = "EMA + MACD Crossover" if (ema_buy and macd_buy) else ("EMA Crossover" if ema_buy else "MACD Crossover")

                    strength = get_signal_strength(ema_diff_pct, curr_macd, curr_rsi, volume_ratio)

                    # فلتر قوة الإشارة: نرسل فقط 2 نجوم فأعلى
                    if strength >= 2:
                        print(f"🟢 BUY SIGNAL: {symbol} | {strategy_name} | {'⭐' * strength} | RSI: {curr_rsi:.1f} | Vol: {volume_ratio:.2f}x")

                        entry = round(current_close, decimals)
                        tp1 = round(entry * 1.0065, decimals)
                        tp2 = round(entry * 1.017, decimals)
                        tp3 = round(entry * 1.032, decimals)
                        tp4 = round(entry * 1.058, decimals)
                        tp5 = round(entry * 1.072, decimals)
                        sl = round(entry * 0.95, decimals)

                        rsi_zone = get_rsi_zone(curr_rsi)
                        volume_status = get_volume_status(volume_ratio)

                        send_crypto_signal(
                            symbol, "LONG", strategy_name,
                            str(entry), "10",
                            str(tp1), str(tp2), str(tp3), str(tp4), str(tp5), str(sl),
                            curr_rsi, rsi_zone, volume_ratio, volume_status,
                            strength, ema_diff_pct, DEFAULT_IMAGE
                        )
                        time.sleep(2)
                    else:
                        print(f"⚪ SKIP (weak): {symbol} | Strength: {'⭐' * strength}")
                else:
                    print(f"⚪ SKIP (RSI overbought): {symbol} | RSI: {curr_rsi:.1f}")

            elif sell_signal and volume_confirmed:
                # فلتر RSI للبيع: لا نبيع إذا كان RSI تحت 25 (تشبع بيعي)
                if curr_rsi > 25:
                    strategy_name = "EMA + MACD Crossover" if (ema_sell and macd_sell) else ("EMA Crossover" if ema_sell else "MACD Crossover")

                    strength = get_signal_strength(abs(ema_diff_pct), curr_macd, curr_rsi, volume_ratio)

                    if strength >= 2:
                        print(f"🔴 SELL SIGNAL: {symbol} | {strategy_name} | {'⭐' * strength} | RSI: {curr_rsi:.1f} | Vol: {volume_ratio:.2f}x")

                        entry = round(current_close, decimals)
                        tp1 = round(entry * 0.9935, decimals)
                        tp2 = round(entry * 0.983, decimals)
                        tp3 = round(entry * 0.968, decimals)
                        tp4 = round(entry * 0.942, decimals)
                        tp5 = round(entry * 0.928, decimals)
                        sl = round(entry * 1.05, decimals)

                        rsi_zone = get_rsi_zone(curr_rsi)
                        volume_status = get_volume_status(volume_ratio)

                        send_crypto_signal(
                            symbol, "SHORT", strategy_name,
                            str(entry), "10",
                            str(tp1), str(tp2), str(tp3), str(tp4), str(tp5), str(sl),
                            curr_rsi, rsi_zone, volume_ratio, volume_status,
                            strength, ema_diff_pct, DEFAULT_IMAGE
                        )
                        time.sleep(2)
                    else:
                        print(f"⚪ SKIP (weak): {symbol} | Strength: {'⭐' * strength}")
                else:
                    print(f"⚪ SKIP (RSI oversold): {symbol} | RSI: {curr_rsi:.1f}")

            else:
                reason = []
                if not buy_signal and not sell_signal:
                    reason.append("no crossover")
                if not volume_confirmed:
                    reason.append(f"low volume ({volume_ratio:.2f}x)")
                print(f"⚪ No signal: {symbol} | {' | '.join(reason)}")

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")

    print("=" * 50)
    print("✅ Scan completed.")
    print("=" * 50)


if __name__ == "__main__":
    print("🤖 Bot started successfully...")
    analyze_and_trade()