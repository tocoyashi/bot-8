import os
import ssl
import time
import json

ssl._create_default_https_context = ssl._create_unverified_context

import ccxt
import pandas as pd
import ta
import requests
from datetime import datetime, timedelta

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

TIMEFRAME = "15m"
CANDLES_LIMIT = 672  # 7 days x 24 hours x 4 candles (15m)

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "TRX/USDT", "POL/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT",
    "ATOM/USDT", "XLM/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT",
    "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT", "FIL/USDT",
    "AAVE/USDT", "GRT/USDT", "PEPE/USDT", "QNT/USDT", "FET/USDT"
]

LEVERAGE = 10
SL_PCT = 0.05
TP_PCTS = [0.0065, 0.017, 0.032, 0.058, 0.072]
TP_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


def get_signal_strength(ema_diff_pct, macd_val, rsi, volume_ratio):
    score = 0.0
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


def simulate_trade(entry_price, direction, future_highs, future_lows, future_closes):
    """
    Simulate a trade by checking future candles to see which TP or SL gets hit first.
    Returns: (result_label, hit_tp_index, simple_pnl_pct)
    """
    entry = entry_price
    n_candles = min(20, len(future_highs))

    for i in range(n_candles):
        high = future_highs.iloc[i]
        low = future_low = future_lows.iloc[i]

        if direction == "LONG":
            # Check SL first
            if low <= entry * (1 - SL_PCT):
                return "SL Hit", -1, -SL_PCT * LEVERAGE
            # Check TPs
            for tp_idx, tp_pct in enumerate(TP_PCTS):
                if high >= entry * (1 + tp_pct):
                    return f"TP{tp_idx + 1} Hit", tp_idx, tp_pct * LEVERAGE

        else:  # SHORT
            if high >= entry * (1 + SL_PCT):
                return "SL Hit", -1, -SL_PCT * LEVERAGE
            for tp_idx, tp_pct in enumerate(TP_PCTS):
                if low <= entry * (1 - tp_pct):
                    return f"TP{tp_idx + 1} Hit", tp_idx, tp_pct * LEVERAGE

    # Nothing hit - close at last close price
    last_close = future_closes.iloc[n_candles - 1]
    if direction == "LONG":
        pnl = ((last_close - entry) / entry) * LEVERAGE
    else:
        pnl = ((entry - last_close) / entry) * LEVERAGE

    label = "Timeout +" if pnl >= 0 else "Timeout -"
    return label, -1, pnl


def calculate_weighted_pnl(hit_tp_idx):
    """Calculate weighted PnL based on partial close at each TP level."""
    if hit_tp_idx < 0:
        return 0.0
    total = 0.0
    for i in range(hit_tp_idx + 1):
        total += TP_PCTS[i] * TP_WEIGHTS[i] * LEVERAGE
    return total


def run_backtest():
    print("=" * 60)
    print("  BACKTEST STARTED - EMA + MACD + RSI + Volume")
    print("=" * 60)

    exchange = ccxt.mexc()
    all_trades = []

    for symbol in SYMBOLS:
        try:
            print(f"  Analyzing {symbol}...", end=" ")
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES_LIMIT)

            if len(ohlcv) < 100:
                print("SKIP (not enough data)")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Indicators
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['macd_hist'] = ta.trend.macd_diff(df['close'])
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['vol_ma20'] = df['volume'].rolling(window=20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma20']

            count = 0

            for i in range(50, len(df) - 20):
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

                future_highs = df['high'].iloc[i + 1:i + 21]
                future_lows = df['low'].iloc[i + 1:i + 21]
                future_closes = df['close'].iloc[i + 1:i + 21]

                result, hit_tp, simple_pnl = simulate_trade(
                    close, direction, future_highs, future_lows, future_closes
                )

                if result.startswith("TP"):
                    weighted_pnl = calculate_weighted_pnl(hit_tp)
                else:
                    weighted_pnl = simple_pnl

                all_trades.append({
                    "symbol": symbol,
                    "direction": direction,
                    "strategy": strategy,
                    "strength": strength,
                    "entry": close,
                    "result": result,
                    "pnl": weighted_pnl,
                    "rsi": round(curr_rsi, 1),
                    "vol_ratio": round(vol_ratio, 2),
                    "time": df['datetime'].iloc[i]
                })
                count += 1

            print(f"OK ({count} trades)")

        except Exception as e:
            print(f"ERROR: {e}")

    return all_trades


def build_report(all_trades):
    """Build statistics from all trades."""
    if not all_trades:
        return None

    df = pd.DataFrame(all_trades)

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

    best_trades = df.nlargest(5, 'pnl')[['symbol', 'direction', 'strategy', 'pnl', 'strength']]
    worst_trades = df.nsmallest(5, 'pnl')[['symbol', 'direction', 'strategy', 'pnl', 'strength']]

    longs = df[df['direction'] == 'LONG']
    shorts = df[df['direction'] == 'SHORT']
    long_wins = longs[longs['pnl'] > 0]
    short_wins = shorts[shorts['pnl'] > 0]
    long_win_rate = (len(long_wins) / len(longs) * 100) if len(longs) > 0 else 0
    short_win_rate = (len(short_wins) / len(shorts) * 100) if len(shorts) > 0 else 0

    strategy_groups = df.groupby('strategy').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    ).round(2)

    strength_groups = df.groupby('strength').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        total_pnl=('pnl', 'sum'),
    ).round(2)

    result_counts = df['result'].value_counts()

    coin_perf = df.groupby('symbol').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        total_pnl=('pnl', 'sum'),
    ).round(2).sort_values('total_pnl', ascending=False)

    # Max Drawdown
    df_sorted = df.sort_values('time').reset_index(drop=True)
    df_sorted['cum_pnl'] = df_sorted['pnl'].cumsum()
    df_sorted['peak'] = df_sorted['cum_pnl'].cummax()
    df_sorted['drawdown'] = df_sorted['cum_pnl'] - df_sorted['peak']
    max_drawdown = df_sorted['drawdown'].min()

    # Consecutive wins / losses
    df_sorted['is_win'] = df_sorted['pnl'] > 0
    streaks = df_sorted['is_win'].astype(int).groupby(
        df_sorted['is_win'].ne(df_sorted['is_win'].shift()).cumsum()
    ).sum()
    max_consec_wins = streaks[df_sorted['is_win'].iloc[0] == 1].max() if len(streaks) > 0 else 0
    # Get max consecutive wins properly
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    current_type = None
    for val in df_sorted['is_win']:
        if val == current_type:
            current_streak += 1
        else:
            current_type = val
            current_streak = 1
        if current_type and current_streak > max_win_streak:
            max_win_streak = current_streak
        if not current_type and current_streak > max_loss_streak:
            max_loss_streak = current_streak

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
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "long_count": len(longs),
        "long_win_rate": long_win_rate,
        "long_pnl": longs['pnl'].sum(),
        "short_count": len(shorts),
        "short_win_rate": short_win_rate,
        "short_pnl": shorts['pnl'].sum(),
        "strategy_groups": strategy_groups,
        "strength_groups": strength_groups,
        "result_counts": result_counts,
        "coin_perf": coin_perf,
        "start_date": start_date,
        "end_date": end_date,
    }


def format_report(report):
    """Format the backtest report as a beautiful Telegram message."""
    if not report:
        return (
            "<b>No trades found during the backtest period.</b>\n\n"
            "The filters may be too strict or the market conditions "
            "didn't generate enough signals.\n\n"
            "Try increasing the candle count or relaxing the filters."
        )

    r = report
    total = r['total_trades']

    # Visual indicators
    wr_icon = "🟢" if r['win_rate'] >= 60 else ("🟡" if r['win_rate'] >= 45 else "🔴")
    profit_icon = "📈" if r['total_profit'] > 0 else "📉"
    profit_sign = "+" if r['total_profit'] > 0 else ""
    pf_icon = "🟢" if r['profit_factor'] >= 2.0 else ("🟡" if r['profit_factor'] >= 1.0 else "🔴")

    text = (
        f"╔═══════════════════════════════════════╗\n"
        f"║      📊  BACKTEST  REPORT  📊          ║\n"
        f"║     DR PYTHON SIGNALS BOT             ║\n"
        f"╠═══════════════════════════════════════╣\n"
        f"\n"
        f"  📅  Period:  <code>{r['start_date']}</code>  →  <code>{r['end_date']}</code>\n"
        f"  ⏱  Timeframe:  <code>15m</code>   |   ⚡  Leverage:  <code>{LEVERAGE}x</code>\n"
        f"  🪙  Coins Scanned:  <code>30</code>   |   📊  Candles:  <code>{CANDLES_LIMIT}</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📋  <b>OVERALL PERFORMANCE</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"  🎯  Total Trades:  <code>{total}</code>\n"
        f"  ✅  Wins:  <code>{r['win_count']}</code>     ❌  Losses:  <code>{r['loss_count']}</code>\n"
        f"  {wr_icon}  <b>Win Rate:</b>  <code>{r['win_rate']:.1f}%</code>\n"
        f"\n"
        f"  {profit_icon}  <b>Total PnL:</b>  <code>{profit_sign}{r['total_profit']:.2f}%</code>\n"
        f"  📊  <b>Avg PnL/Trade:</b>  <code>{r['avg_profit']:.2f}%</code>\n"
        f"  {pf_icon}  <b>Profit Factor:</b>  <code>{r['profit_factor']:.2f}</code>\n"
        f"\n"
        f"  🏆  Best Trade:  <code>+{r['max_win']:.2f}%</code>\n"
        f"  💀  Worst Trade:  <code>{r['max_loss']:.2f}%</code>\n"
        f"  📉  Max Drawdown:  <code>{r['max_drawdown']:.2f}%</code>\n"
        f"  🔥  Max Win Streak:  <code>{r['max_win_streak']}</code>   |   💔  Max Loss Streak:  <code>{r['max_loss_streak']}</code>\n"
        f"\n"
    )

    # By direction
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📈  <b>PERFORMANCE BY DIRECTION</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"  🟢  LONG:   <code>{r['long_count']}</code> trades  |  WR: <code>{r['long_win_rate']:.1f}%</code>  |  PnL: <code>{r['long_pnl']:+.2f}%</code>\n"
        f"  🔴  SHORT:  <code>{r['short_count']}</code> trades  |  WR: <code>{r['short_win_rate']:.1f}%</code>  |  PnL: <code>{r['short_pnl']:+.2f}%</code>\n"
        f"\n"
    )

    # By strategy
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🧠  <b>PERFORMANCE BY STRATEGY</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for strat, row in r['strategy_groups'].iterrows():
        icon = "🟢" if row['total_pnl'] > 0 else "🔴"
        text += (
            f"  {icon}  <code>{strat:<10}</code>  "
            f"Trades: <code>{int(row['trades']):>3}</code>  "
            f"WR: <code>{row['win_rate']:>5.1f}%</code>  "
            f"PnL: <code>{row['total_pnl']:>+8.2f}%</code>\n"
        )

    text += "\n"

    # By strength
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  💪  <b>PERFORMANCE BY SIGNAL STRENGTH</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for strength, row in r['strength_groups'].iterrows():
        stars = "⭐" * int(strength) + "☆" * (5 - int(strength))
        icon = "🟢" if row['total_pnl'] > 0 else "🔴"
        text += (
            f"  {icon}  {stars}  "
            f"Trades: <code>{int(row['trades']):>3}</code>  "
            f"WR: <code>{row['win_rate']:>5.1f}%</code>  "
            f"PnL: <code>{row['total_pnl']:>+8.2f}%</code>\n"
        )

    text += "\n"

    # Result distribution
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📊  <b>TRADE OUTCOME DISTRIBUTION</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for result, count in r['result_counts'].items():
        pct = (count / total) * 100
        bar_len = int(pct / 4)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        text += f"  <code>{result:>12}</code>  {bar}  <code>{count:>3}</code>  (<code>{pct:>5.1f}%</code>)\n"

    text += "\n"

    # Top 5 coins
    top5 = r['coin_perf'].head(5)
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🏆  <b>TOP 5 COINS</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    medals = ["🥇", "🥈", "🥉", " 4.", " 5."]
    for rank, (coin, row) in enumerate(top5.iterrows()):
        text += (
            f"  {medals[rank]}  <code>{coin:<12}</code>  "
            f"Trades: <code>{int(row['trades']):>3}</code>  "
            f"WR: <code>{row['win_rate']:>5.1f}%</code>  "
            f"PnL: <code>{row['total_pnl']:>+8.2f}%</code>\n"
        )

    text += "\n"

    # Bottom 5 coins
    bottom5 = r['coin_perf'].tail(5).iloc[::-1]
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  💀  <b>WORST 5 COINS</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for rank, (coin, row) in enumerate(bottom5.iterrows()):
        text += (
            f"  {rank + 1:>2}.  <code>{coin:<12}</code>  "
            f"Trades: <code>{int(row['trades']):>3}</code>  "
            f"WR: <code>{row['win_rate']:>5.1f}%</code>  "
            f"PnL: <code>{row['total_pnl']:>+8.2f}%</code>\n"
        )

    text += "\n"

    # Best 5 trades
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🎰  <b>TOP 5 TRADES</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for _, t in r['best_trades'].iterrows():
        d_icon = "🟢" if t['direction'] == "LONG" else "🔴"
        stars = "⭐" * int(t['strength'])
        text += (
            f"  {d_icon}  <code>{t['symbol']:<12}</code>  {t['direction']:<5}  "
            f"{t['strategy']:<10}  "
            f"PnL: <code>{t['pnl']:>+7.2f}%</code>  {stars}\n"
        )

    text += "\n"

    # Worst 5 trades
    text += (
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  ⚠️  <b>WORST 5 TRADES</b>\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
    )
    for _, t in r['worst_trades'].iterrows():
        d_icon = "🟢" if t['direction'] == "LONG" else "🔴"
        stars = "⭐" * int(t['strength'])
        text += (
            f"  {d_icon}  <code>{t['symbol']:<12}</code>  {t['direction']:<5}  "
            f"{t['strategy']:<10}  "
            f"PnL: <code>{t['pnl']:>+7.2f}%</code>  {stars}\n"
        )

    text += (
        f"\n"
        f"╚═══════════════════════════════════════╝\n"
        f"\n"
        f"  <i>⚡ Report generated automatically by Backtest Bot</i>\n"
        f"  <i>⏰ {datetime.now().strftime('%Y-%m-%d  %H:%M UTC')}</i>\n"
        f"  <i>🤖 Dr Python Signals</i>"
    )

    return text


def send_report(text):
    """Send report to Telegram, splitting if necessary."""
    MAX_LEN = 4096

    if len(text) <= MAX_LEN:
        parts = [text]
    else:
        lines = text.split('\n')
        parts = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > MAX_LEN - 200:
                parts.append(current.strip())
                current = line + "\n"
            else:
                current += line + "\n"
        if current.strip():
            parts.append(current.strip())

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
                print(f"  ✅ Part {i + 1}/{len(parts)} sent successfully!")
            else:
                print(f"  ❌ Error sending part {i + 1}: {response.json().get('description')}")
        except Exception as e:
            print(f"  ❌ Network error on part {i + 1}: {e}")

        if i < len(parts) - 1:
            time.sleep(1)


if __name__ == "__main__":
    print("🤖 Backtest Bot started...")
    print()

    all_trades = run_backtest()
    print()
    print(f"  Total trades found: {len(all_trades)}")
    print()

    report = build_report(all_trades)
    text = format_report(report)

    print("  Sending report to Telegram...")
    send_report(text)

    if report:
        print()
        print("=" * 45)
        print("  BACKTEST SUMMARY")
        print("=" * 45)
        print(f"  Total Trades:   {report['total_trades']}")
        print(f"  Win Rate:       {report['win_rate']:.1f}%")
        print(f"  Total PnL:      {report['total_profit']:+.2f}%")
        print(f"  Profit Factor:  {report['profit_factor']:.2f}")
        print(f"  Max Drawdown:   {report['max_drawdown']:.2f}%")
        print(f"  Avg PnL/Trade:  {report['avg_profit']:+.2f}%")
        print("=" * 45)