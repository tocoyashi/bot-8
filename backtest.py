import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import ccxt
import pandas as pd
import ta
import requests
import json
from datetime import datetime, timedelta

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

TIMEFRAME = "15m"
CANDLES_LIMIT = 672  # 7 أيام × 24 ساعة × 4 شمعات (كل 15 دقيقة)

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "TRX/USDT", "POL/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT",
    "ATOM/USDT", "XLM/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT",
    "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT", "FIL/USDT",
    "AAVE/USDT", "GRT/USDT", "PEPE/USDT", "QNT/USDT", "FET/USDT"
]

LEVERAGE = 10
SL_PCT = 0.05  # 5%
TP_PCTS = [0.0065, 0.017, 0.032, 0.058, 0.072]  # TP1-TP5

# نسب توزيع الأرباح على كل TP (مثلاً 30% على TP1، 25% على TP2...)
TP_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


def get_signal_strength(ema_diff_pct, macd_val, rsi, volume_ratio):
    score = 0
    if abs(ema_diff_pct) > 0.5:
        score += 1.5
    elif abs(ema_diff_pct) > 0.3:
        score += 1.0
    elif abs(ema_diff_pct) > 0.15:
        score += 0.5
    if abs(macd_val) > 0:
        score += min(1.5, abs(macd_val) * 10)
    if 40 <= rsi <= 60:
        score += 1.0
    elif 30 <= rsi <= 70:
        score += 0.5
    if volume_ratio > 2.0:
        score += 1.0
    elif volume_ratio > 1.5:
        score += 0.75
    elif volume_ratio > 1.0:
        score += 0.5
    return max(1, min(5, round(score)))


def simulate_trade(entry_price, direction, high_prices, low_prices, close_prices):
    """
    محاكاة الصفقة: نتتبع الشموع التالية لمعرفة أي TP أو SL تم ضربه أولاً
    يُرجع: (result, hit_tp_index, pnl_pct)
    result: 'win_tp1'..'win_tp5' أو 'loss_sl' أو 'timeout'
    """
    entry = entry_price

    for i in range(len(high_prices)):
        high = high_prices.iloc[i]
        low = low_prices.iloc[i]

        if direction == "LONG":
            # نتحقق من SL أولاً ثم TPs
            if low <= entry * (1 - SL_PCT):
                return "loss_sl", -1, -SL_PCT * LEVERAGE

            for tp_idx, tp_pct in enumerate(TP_PCTS):
                if high >= entry * (1 + tp_pct):
                    return f"win_tp{tp_idx + 1}", tp_idx, tp_pct * LEVERAGE

        else:  # SHORT
            if high >= entry * (1 + SL_PCT):
                return "loss_sl", -1, -SL_PCT * LEVERAGE

            for tp_idx, tp_pct in enumerate(TP_PCTS):
                if low <= entry * (1 - tp_pct):
                    return f"win_tp{tp_idx + 1}", tp_idx, tp_pct * LEVERAGE

    # لم يُضرب شيء — نغلق عند آخر سعر
    last_close = close_prices.iloc[-1]
    if direction == "LONG":
        pnl = ((last_close - entry) / entry) * LEVERAGE
    else:
        pnl = ((entry - last_close) / entry) * LEVERAGE

    if pnl >= 0:
        return "timeout_win", -1, pnl
    else:
        return "timeout_loss", -1, pnl


def calculate_weighted_pnl(hit_tp_idx, direction):
    """
    حساب الربح المرجح عند ضرب عدة TPs
    نفترض أن المتداول يغلق جزءاً عند كل TP
    """
    if hit_tp_idx < 0:
        return 0

    total_pnl = 0
    for i in range(hit_tp_idx + 1):
        tp_pct = TP_PCTS[i]
        weight = TP_WEIGHTS[i]
        total_pnl += tp_pct * weight * LEVERAGE

    return total_pnl


def run_backtest():
    print("=" * 60)
    print("🔬 BACKTEST STARTED — EMA + MACD + RSI + Volume")
    print("=" * 60)

    exchange = ccxt.mexc()

    all_trades = []
    symbol_stats = {}

    for symbol in SYMBOLS:
        try:
            print(f"📊 Analyzing {symbol}...")
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES_LIMIT)

            if len(ohlcv) < 100:
                print(f"  ⚠️ Not enough data for {symbol}, skipping.")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # المؤشرات
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['macd_hist'] = ta.trend.macd_diff(df['close'])
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # حجم التداول النسبي
            df['vol_ma20'] = df['volume'].rolling(window=20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma20']

            symbol_trades = []

            # المسح على البيانات التاريخية
            for i in range(50, len(df) - 20):  # نحتاج 20 شمعة على الأقل بعد الإشارة
                curr_ema9 = df['ema_9'].iloc[i]
                curr_ema21 = df['ema_21'].iloc[i]
                prev_ema9 = df['ema_9'].iloc[i - 1]
                prev_ema21 = df['ema_21'].iloc[i - 1]
                curr_macd = df['macd_hist'].iloc[i]
                prev_macd = df['macd_hist'].iloc[i - 1]
                curr_rsi = df['rsi'].iloc[i]
                vol_ratio = df['vol_ratio'].iloc[i]
                close = df['close'].iloc[i]

                ema_buy = (prev_ema9 < prev_ema21) and (curr_ema9 > curr_ema21)
                ema_sell = (prev_ema9 > prev_ema21) and (curr_ema9 < curr_ema21)
                macd_buy = (prev_macd < 0) and (curr_macd > 0)
                macd_sell = (prev_macd > 0) and (curr_macd < 0)

                buy_signal = ema_buy or macd_buy
                sell_signal = ema_sell or macd_sell

                if not (buy_signal or sell_signal):
                    continue
                if not (vol_ratio >= 1.0):
                    continue

                # فلتر RSI
                if buy_signal and curr_rsi >= 75:
                    continue
                if sell_signal and curr_rsi <= 25:
                    continue

                ema_diff_pct = ((curr_ema9 - curr_ema21) / curr_ema21) * 100
                strength = get_signal_strength(ema_diff_pct, curr_macd, curr_rsi, vol_ratio)

                if strength < 2:
                    continue

                direction = "LONG" if buy_signal else "SHORT"
                if ema_buy and macd_buy:
                    strategy = "EMA+MACD"
                elif ema_buy:
                    strategy = "EMA"
                else:
                    strategy = "MACD"

                # محاكاة الصفقة
                future_highs = df['high'].iloc[i + 1:i + 21]
                future_lows = df['low'].iloc[i + 1:i + 21]
                future_closes = df['close'].iloc[i + 1:i + 21]

                result, hit_tp, simple_pnl = simulate_trade(
                    close, direction, future_highs, future_lows, future_closes
                )

                # حساب الربح المرجح
                if result.startswith("win_tp"):
                    weighted_pnl = calculate_weighted_pnl(hit_tp, direction)
                else:
                    weighted_pnl = simple_pnl

                trade = {
                    "symbol": symbol,
                    "direction": direction,
                    "strategy": strategy,
                    "strength": strength,
                    "entry": close,
                    "result": result,
                    "pnl": weighted_pnl,
                    "rsi": curr_rsi,
                    "vol_ratio": vol_ratio,
                    "time": df['datetime'].iloc[i]
                }
                symbol_trades.append(trade)
                all_trades.append(trade)

            symbol_stats[symbol] = symbol_trades
            print(f"  ✅ {symbol}: {len(symbol_trades)} trades found")

        except Exception as e:
            print(f"  ❌ Error {symbol}: {e}")

    return all_trades, symbol_stats


def build_report(all_trades, symbol_stats):
    """بناء التقرير الإحصائي"""

    if not all_trades:
        return None

    df = pd.DataFrame(all_trades)

    # إحصائيات عامة
    total_trades = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100

    total_profit = df['pnl'].sum()
    avg_profit = df['pnl'].mean()
    total_win_pnl = wins['pnl'].sum()
    total_loss_pnl = abs(losses['pnl'].sum())
    profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')

    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

    max_win = df['pnl'].max()
    max_loss = df['pnl'].min()

    # أفضل 5 صفقات
    best_trades = df.nlargest(5, 'pnl')[['symbol', 'direction', 'strategy', 'pnl', 'strength']]

    # أسوأ 5 صفقات
    worst_trades = df.nsmallest(5, 'pnl')[['symbol', 'direction', 'strategy', 'pnl', 'strength']]

    # حسب الاتجاه
    longs = df[df['direction'] == 'LONG']
    shorts = df[df['direction'] == 'SHORT']

    long_wins = longs[longs['pnl'] > 0]
    short_wins = shorts[shorts['pnl'] > 0]
    long_win_rate = (len(long_wins) / len(longs) * 100) if len(longs) > 0 else 0
    short_win_rate = (len(short_wins) / len(shorts) * 100) if len(shorts) > 0 else 0
    long_pnl = longs['pnl'].sum()
    short_pnl = shorts['pnl'].sum()

    # حسب الاستراتيجية
    strategy_groups = df.groupby('strategy').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    ).round(2)

    # حسب قوة الإشارة
    strength_groups = df.groupby('strength').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        total_pnl=('pnl', 'sum'),
    ).round(2)

    # حسب النتيجة
    result_counts = df['result'].value_counts()

    # أفضل عملات
    coin_perf = df.groupby('symbol').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        total_pnl=('pnl', 'sum'),
    ).round(2).sort_values('total_pnl', ascending=False)

    # Max Drawdown (بسيط)
    df_sorted = df.sort_values('time').reset_index(drop=True)
    df_sorted['cum_pnl'] = df_sorted['pnl'].cumsum()
    df_sorted['peak'] = df_sorted['cum_pnl'].cummax()
    df_sorted['drawdown'] = df_sorted['cum_pnl'] - df_sorted['peak']
    max_drawdown = df_sorted['drawdown'].min()

    # فترة الباك تيست
    start_date = df['time'].min().strftime('%Y-%m-%d')
    end_date = df['time'].max().strftime('%Y-%m-%d')

    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "avg_profit": avg_profit,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win": max_win,
        "max_loss": max_loss,
        "max_drawdown": max_drawdown,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "long_stats": {
            "count": len(longs),
            "win_rate": long_win_rate,
            "pnl": long_pnl
        },
        "short_stats": {
            "count": len(shorts),
            "win_rate": short_win_rate,
            "pnl": short_pnl
        },
        "strategy_groups": strategy_groups,
        "strength_groups": strength_groups,
        "result_counts": result_counts,
        "coin_perf": coin_perf,
        "start_date": start_date,
        "end_date": end_date,
        "leverage": LEVERAGE,
    }


def format_report(report):
    """تحويل التقرير إلى رسالة تيليجرام جميلة"""

    if not report:
        return (
            "❌ <b>لم يتم العثور على أي صفقات</b>\n\n"
            "لا توجد إشارات كافية خلال فترة الباك تيست.\n"
            "جرب زيادة عدد الشموع أو تغيير الفلاتر."
        )

    r = report
    total = r['total_trades']

    # لون النسبة حسب الأداء
    if r['win_rate'] >= 60:
        wr_icon = "🟢"
    elif r['win_rate'] >= 45:
        wr_icon = "🟡"
    else:
        wr_icon = "🔴"

    if r['total_profit'] > 0:
        profit_icon = "📈"
        profit_sign = "+"
    else:
        profit_icon = "📉"
        profit_sign = ""

    if r['profit_factor'] >= 2.0:
        pf_icon = "🟢"
    elif r['profit_factor'] >= 1.0:
        pf_icon = "🟡"
    else:
        pf_icon = "🔴"

    # بناء النص
    text = (
        f"╔═══════════════════════════════════════╗\n"
        f"║   📊  BACKTEST REPORT  📊             ║\n"
        f"║   DR PYTHON SIGNALS BOT              ║\n"
        f"╠═══════════════════════════════════════╣\n"
        f"\n"
        f"  📅  الفترة:  <code>{r['start_date']}</code>  ←→  <code>{r['end_date']}</code>\n"
        f"  ⏱  الإطار الزمني:  <code>15m</code>\n"
        f"  ⚡  الرافعة:  <code>{r['leverage']}x</code>\n"
        f"  💰  العملات:  <code>30</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📋  <b>الإحصائيات العامة</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"  🎯  إجمالي الصفقات:  <code>{total}</code>\n"
        f"  ✅  صفقات رابحة:  <code>{r['win_count']}</code>\n"
        f"  ❌  صفقات خاسرة:  <code>{r['loss_count']}</code>\n"
        f"  {wr_icon}  <b>نسبة الفوز:</b>  <code>{r['win_rate']:.1f}%</code>\n"
        f"\n"
        f"  {profit_icon}  <b>إجمالي الربح:</b>  <code>{profit_sign}{r['total_profit']:.2f}%</code>\n"
        f"  📊  <b>متوسط الربح:</b>  <code>{r['avg_profit']:.2f}%</code> لكل صفقة\n"
        f"  {pf_icon}  <b>معامل الربح (PF):</b>  <code>{r['profit_factor']:.2f}</code>\n"
        f"\n"
        f"  🏆  أفضل صفقة:  <code>+{r['max_win']:.2f}%</code>\n"
        f"  💀  أسوأ صفقة:  <code>{r['max_loss']:.2f}%</code>\n"
        f"  📉  أقصى تراجع:  <code>{r['max_drawdown']:.2f}%</code>\n"
        f"\n"
    )

    # حسب الاتجاه
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📈  <b>حسب الاتجاه</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"  🟢  LONG:  <code>{r['long_stats']['count']}</code> صفقة  |  فوز: <code>{r['long_stats']['win_rate']:.1f}%</code>  |  ربح: <code>{r['long_stats']['pnl']:+.2f}%</code>\n"
        f"  🔴  SHORT:  <code>{r['short_stats']['count']}</code> صفقة  |  فوز: <code>{r['short_stats']['win_rate']:.1f}%</code>  |  ربح: <code>{r['short_stats']['pnl']:+.2f}%</code>\n"
        f"\n"
    )

    # حسب الاستراتيجية
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🧠  <b>حسب الاستراتيجية</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for strat, row in r['strategy_groups'].iterrows():
        icon = "🟢" if row['total_pnl'] > 0 else "🔴"
        text += f"  {icon}  <code>{strat}</code>  |  صفقات: <code>{int(row['trades'])}</code>  |  فوز: <code>{row['win_rate']:.1f}%</code>  |  ربح: <code>{row['total_pnl']:+.2f}%</code>\n"

    text += "\n"

    # حسب قوة الإشارة
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  💪  <b>حسب قوة الإشارة</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for strength, row in r['strength_groups'].iterrows():
        stars = "⭐" * int(strength) + "☆" * (5 - int(strength))
        icon = "🟢" if row['total_pnl'] > 0 else "🔴"
        text += f"  {icon}  {stars}  |  صفقات: <code>{int(row['trades'])}</code>  |  فوز: <code>{row['win_rate']:.1f}%</code>  |  ربح: <code>{row['total_pnl']:+.2f}%</code>\n"

    text += "\n"

    # نتائج الصفقات
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📊  <b>توزيع النتائج</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for result, count in r['result_counts'].items():
        pct = (count / total) * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        text += f"  <code>{result:>12}</code>  {bar}  <code>{count}</code>  (<code>{pct:.1f}%</code>)\n"

    text += "\n"

    # أفضل 5 عملات
    top5 = r['coin_perf'].head(5)
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🏆  <b>أفضل 5 عملات</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for rank, (coin, row) in enumerate(top5.iterrows(), 1):
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank - 1]
        text += f"  {medal}  <code>{coin}</code>  |  صفقات: <code>{int(row['trades'])}</code>  |  فوز: <code>{row['win_rate']:.1f}%</code>  |  ربح: <code>{row['total_pnl']:+.2f}%</code>\n"

    text += "\n"

    # أسوأ 5 عملات
    bottom5 = r['coin_perf'].tail(5).iloc[::-1]
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  💀  <b>أسوأ 5 عملات</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for rank, (coin, row) in enumerate(bottom5.iterrows(), 1):
        text += f"  {rank}️⃣  <code>{coin}</code>  |  صفقات: <code>{int(row['trades'])}</code>  |  فوز: <code>{row['win_rate']:.1f}%</code>  |  ربح: <code>{row['total_pnl']:+.2f}%</code>\n"

    text += "\n"

    # أفضل وأسوأ صفقات
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🎰  <b>أفضل 5 صفقات</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for _, t in r['best_trades'].iterrows():
        d_icon = "🟢" if t['direction'] == "LONG" else "🔴"
        text += f"  {d_icon}  <code>{t['symbol']}</code>  {t['direction']}  |  {t['strategy']}  |  <code>+{t['pnl']:.2f}%</code>  |  {'⭐' * int(t['strength'])}\n"

    text += "\n"
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  ⚠️  <b>أسوأ 5 صفقات</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for _, t in r['worst_trades'].iterrows():
        d_icon = "🟢" if t['direction'] == "LONG" else "🔴"
        text += f"  {d_icon}  <code>{t['symbol']}</code>  {t['direction']}  |  {t['strategy']}  |  <code>{t['pnl']:.2f}%</code>  |  {'⭐' * int(t['strength'])}\n"

    text += (
        f"\n"
        f"╚═══════════════════════════════════════╝\n"
        f"\n"
        f"  <i>⚡ تم إنشاء هذا التقرير تلقائياً بواسطة Backtest Bot</i>\n"
        f"  <i>⏰ {datetime.now().strftime('%Y-%m-%d  %H:%M')}</i>\n"
        f"  <i>🤖 Dr Python Signals</i>"
    )

    return text


def send_report(text):
    """إرسال التقرير إلى تيليجرام"""
    # إذا كان النص طويلاً جداً، نقسمه
    max_len = 4096

    if len(text) <= max_len:
        parts = [text]
    else:
        # نقسم عند الأسطر الفارغة
        lines = text.split('\n')
        parts = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > max_len - 100:
                parts.append(current)
                current = line + "\n"
            else:
                current += line + "\n"
        if current:
            parts.append(current)

    for i, part in enumerate(parts):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHANNEL_ID,
            "text": part,
            "parse_mode": "HTML"
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.json().get('ok'):
                print(f"✅ Report part {i + 1}/{len(parts)} sent!")
            else:
                print(f"❌ Error sending part {i + 1}: {response.json().get('description')}")
        except Exception as e:
            print(f"❌ Network error: {e}")

        if i < len(parts) - 1:
            time.sleep(1)


if __name__ == "__main__":
    print("🤖 Backtest Bot started...")
    all_trades, symbol_stats = run_backtest()
    print(f"\n📊 Total trades found: {len(all_trades)}")

    report = build_report(all_trades, symbol_stats)
    text = format_report(report)

    print("\n📤 Sending report to Telegram...")
    send_report(text)

    # طباعة ملخص في الكونسول
    if report:
        print(f"\n{'=' * 40}")
        print(f"  SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Trades: {report['total_trades']}")
        print(f"  Win Rate: {report['win_rate']:.1f}%")
        print(f"  Total PnL: {report['total_profit']:+.2f}%")
        print(f"  Profit Factor: {report['profit_factor']:.2f}")
        print(f"  Max Drawdown: {report['max_drawdown']:.2f}%")
        print(f"{'=' * 40}")