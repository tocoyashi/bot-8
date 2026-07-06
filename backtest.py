import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import ccxt
import pandas as pd
import ta
import time
import requests
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════
#  SCALP STRATEGY BACKTEST — 3 Months, 1H Timeframe
# ═══════════════════════════════════════════════════

SYMBOLS = [
    "BTC/USDT"
]

TIMEFRAME = "1h"

# Strategy Parameters
RSI_LENGTH = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
EMA_FAST = 150
EMA_SLOW = 200
EMA_FAR_THRESHOLD = 1.5
MAX_EMA200_DIST = 3.0
COOLDOWN_BARS = 10

# TP/SL
SL_PCT = 1.5
TP1_PCT = 1.0
TP2_PCT = 3.0
TP3_PCT = 6.0
TP_WEIGHTS = [0.50, 0.25, 0.25]

LEVERAGE = 10


def fetch_all_candles(symbol, exchange, months=3):
    """Fetch 3 months of 1H candles with pagination."""
    now = datetime.now(tz=None)
    since = int((now - timedelta(days=months * 30)).timestamp() * 1000)
    all_candles = []

    while True:
        candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        time.sleep(0.3)

    # Remove duplicates
    seen = set()
    unique = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    unique.sort(key=lambda x: x[0])
    return unique


def run_backtest():
    print("=" * 60)
    print("  SCALP STRATEGY BACKTEST - 3 Months")
    print("  Timeframe: 1H  |  Leverage: 10x")
    print("=" * 60)

    exchange = ccxt.mexc()

    all_trades = []
    start_date = None
    end_date = None

    for symbol in SYMBOLS:
        try:
            print(f"\n  Fetching {symbol}...", end=" ", flush=True)
            candles = fetch_all_candles(symbol, exchange, months=3)
            print(f"{len(candles)} candles")

            if len(candles) < 250:
                print(f"    SKIP - not enough data")
                continue

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            if start_date is None:
                start_date = df['datetime'].iloc[0]
                end_date = df['datetime'].iloc[-1]

            # ─── Indicators ───
            df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_LENGTH)
            df['ema150'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

            # ─── Scan for signals ───
            last_buy_bar = -100
            last_sell_bar = -100

            for i in range(201, len(df) - 20):
                row = df.iloc[i]
                prev = df.iloc[i - 1]

                if pd.isna(row['rsi']) or pd.isna(row['ema150']) or pd.isna(row['ema200']):
                    continue
                if pd.isna(prev['rsi']):
                    continue

                curr_close = row['close']
                curr_rsi = row['rsi']
                prev_rsi = prev['rsi']
                curr_ema150 = row['ema150']
                curr_ema200 = row['ema200']

                # ─── EMA Trend Filter ───
                ema150_above = curr_ema150 > curr_ema200
                ema150_below = curr_ema150 < curr_ema200

                # ─── EMA Distance Filter ───
                ema_distance = abs((curr_ema150 - curr_ema200) / curr_ema200) * 100
                if ema_distance >= EMA_FAR_THRESHOLD:
                    continue

                # ─── Price near EMA 200 ───
                price_ema200_dist = abs((curr_close - curr_ema200) / curr_ema200) * 100
                if price_ema200_dist > MAX_EMA200_DIST:
                    continue

                # ─── RSI Cross Detection ───
                buy_cross = (curr_rsi > RSI_OVERBOUGHT and prev_rsi <= RSI_OVERBOUGHT)
                sell_cross = (curr_rsi < RSI_OVERSOLD and prev_rsi >= RSI_OVERSOLD)

                # ─── BUY Signal ───
                if buy_cross and ema150_above and curr_close > curr_ema150:
                    if (i - last_buy_bar) < COOLDOWN_BARS:
                        continue

                    last_buy_bar = i
                    entry = curr_close
                    sl_price = entry * (1 - SL_PCT / 100)
                    tp1_price = entry * (1 + TP1_PCT / 100)
                    tp2_price = entry * (1 + TP2_PCT / 100)
                    tp3_price = entry * (1 + TP3_PCT / 100)

                    result = simulate_trade(df, i, "LONG", entry, sl_price, tp1_price, tp2_price, tp3_price)
                    if result:
                        result['symbol'] = symbol
                        result['entry_time'] = row['datetime']
                        result['rsi'] = curr_rsi
                        result['ema_dist'] = ema_distance
                        result['price_ema200_dist'] = price_ema200_dist
                        all_trades.append(result)

                # ─── SELL Signal ───
                elif sell_cross and ema150_below and curr_close < curr_ema150:
                    if (i - last_sell_bar) < COOLDOWN_BARS:
                        continue

                    last_sell_bar = i
                    entry = curr_close
                    sl_price = entry * (1 + SL_PCT / 100)
                    tp1_price = entry * (1 - TP1_PCT / 100)
                    tp2_price = entry * (1 - TP2_PCT / 100)
                    tp3_price = entry * (1 - TP3_PCT / 100)

                    result = simulate_trade(df, i, "SHORT", entry, sl_price, tp1_price, tp2_price, tp3_price)
                    if result:
                        result['symbol'] = symbol
                        result['entry_time'] = row['datetime']
                        result['rsi'] = curr_rsi
                        result['ema_dist'] = ema_distance
                        result['price_ema200_dist'] = price_ema200_dist
                        all_trades.append(result)

        except Exception as e:
            print(f"    ERROR: {e}")

    # ═══════════════════════════════════════════════
    #  RESULTS & REPORT
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    if not all_trades:
        print("  No trades generated in 3 months!")
        return

    trades_df = pd.DataFrame(all_trades)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_pct'] > 0])
    losses = len(trades_df[trades_df['pnl_pct'] <= 0])
    win_rate = (wins / total_trades) * 100

    total_pnl = trades_df['pnl_pct'].sum()
    total_pnl_leveraged = total_pnl * LEVERAGE

    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean()) if losses > 0 else 0

    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for _, t in trades_df.iterrows():
        if t['pnl_pct'] > 0:
            streak = streak + 1 if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    # Best & worst trades
    best_trade = trades_df.loc[trades_df['pnl_pct'].idxmax()]
    worst_trade = trades_df.loc[trades_df['pnl_pct'].idxmin()]

    # Per-coin stats
    print(f"\n  Period: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Wins: {wins}  |  Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg Win: +{avg_win:.2f}%  |  Avg Loss: -{avg_loss:.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Max Win Streak: {max_win_streak}  |  Max Loss Streak: {max_loss_streak}")
    print(f"\n  Total PnL (no leverage): {total_pnl:+.2f}%")
    print(f"  Total PnL ({LEVERAGE}x leverage): {total_pnl_leveraged:+.2f}%")
    print(f"  Best Trade: {best_trade['symbol']} {best_trade['direction']} ({best_trade['pnl_pct']:+.2f}%)")
    print(f"  Worst Trade: {worst_trade['symbol']} {worst_trade['direction']} ({worst_trade['pnl_pct']:+.2f}%)")

    # Exit type breakdown
    exit_counts = trades_df['exit_type'].value_counts()
    print(f"\n  Exit Breakdown:")
    for etype, count in exit_counts.items():
        etype_trades = trades_df[trades_df['exit_type'] == etype]
        avg_pnl = etype_trades['pnl_pct'].mean()
        print(f"    {etype}: {count} trades (avg: {avg_pnl:+.2f}%)")

    # Per-coin breakdown
    print(f"\n  Per-Coin Breakdown:")
    print(f"  {'Coin':<12} {'Trades':>7} {'Win%':>7} {'PnL%':>8} {'Best':>8} {'Worst':>8}")
    print(f"  {'-' * 52}")
    for sym in SYMBOLS:
        coin_trades = trades_df[trades_df['symbol'] == sym]
        if len(coin_trades) == 0:
            print(f"  {sym:<12} {'0':>7}")
            continue
        c_wins = len(coin_trades[coin_trades['pnl_pct'] > 0])
        c_wr = (c_wins / len(coin_trades)) * 100
        c_pnl = coin_trades['pnl_pct'].sum()
        c_best = coin_trades['pnl_pct'].max()
        c_worst = coin_trades['pnl_pct'].min()
        print(f"  {sym:<12} {len(coin_trades):>7} {c_wr:>6.1f}% {c_pnl:>+7.2f}% {c_best:>+7.2f}% {c_worst:>+7.2f}%")

    # Monthly breakdown
    print(f"\n  Monthly Breakdown:")
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    for month, mtrades in trades_df.groupby('month'):
        m_wins = len(mtrades[mtrades['pnl_pct'] > 0])
        m_wr = (m_wins / len(mtrades)) * 100 if len(mtrades) > 0 else 0
        m_pnl = mtrades['pnl_pct'].sum()
        m_pnl_lev = m_pnl * LEVERAGE
        print(f"    {month}: {len(mtrades):>3} trades  |  Win: {m_wr:.0f}%  |  PnL: {m_pnl:+.2f}%  ({m_pnl_lev:+.1f}% x{LEVERAGE})")

    # TP Hit Stats
    tp1_count = trades_df['tp1_hit'].sum()
    tp2_count = trades_df['tp2_hit'].sum()
    tp3_count = trades_df['tp3_hit'].sum()
    no_tp = total_trades - tp1_count

    print(f"\n  TP Hit Stats:")
    print(f"    TP1 ({TP1_PCT}%): {tp1_count}/{total_trades} trades ({tp1_count/total_trades*100:.0f}%)")
    print(f"    TP2 ({TP2_PCT}%): {tp2_count}/{total_trades} trades ({tp2_count/total_trades*100:.0f}%)")
    print(f"    TP3 ({TP3_PCT}%): {tp3_count}/{total_trades} trades ({tp3_count/total_trades*100:.0f}%)")
    print(f"    No TP hit:        {no_tp}/{total_trades} trades ({no_tp/total_trades*100:.0f}%)")

    # Direction breakdown
    print(f"\n  Direction Breakdown:")
    for direction in ["LONG", "SHORT"]:
        dtrades = trades_df[trades_df['direction'] == direction]
        if len(dtrades) == 0:
            continue
        d_wins = len(dtrades[dtrades['pnl_pct'] > 0])
        d_wr = (d_wins / len(dtrades)) * 100
        d_pnl = dtrades['pnl_pct'].sum()
        print(f"    {direction}: {len(dtrades)} trades  |  Win: {d_wr:.1f}%  |  PnL: {d_pnl:+.2f}%")

    print(f"\n{'=' * 60}")

    # ─── Telegram Report ───
    send_telegram_report(
        total_trades, wins, losses, win_rate,
        avg_win, avg_loss, profit_factor,
        max_win_streak, max_loss_streak,
        total_pnl, total_pnl_leveraged,
        best_trade, worst_trade,
        trades_df, exit_counts
    )


def simulate_trade(df, entry_idx, direction, entry, sl_price, tp1_price, tp2_price, tp3_price):
    """Simulate a trade with partial TP exits. SL stays at -1.5% (no breakeven)."""
    tp_remaining = 1.0
    realized_pnl = 0.0
    exit_type = None
    bars_held = 0

    # Track which TPs were hit
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False

    for j in range(entry_idx + 1, min(entry_idx + 120, len(df))):
        bar = df.iloc[j]
        high = bar['high']
        low = bar['low']
        bars_held += 1

        if direction == "LONG":
            # Check SL
            if low <= sl_price:
                pnl = (sl_price - entry) / entry * 100 * tp_remaining
                realized_pnl += pnl
                exit_type = "SL"
                break

            # TP1 (50%) — only if full position still open
            if tp_remaining > 0.50 and high >= tp1_price:
                tp_size = 0.50
                pnl = (tp1_price - entry) / entry * 100 * tp_size
                realized_pnl += pnl
                tp_remaining -= tp_size
                tp1_hit = True

            # TP2 (25%) — only if TP1 done but TP2 not
            if 0.24 < tp_remaining <= 0.50 and high >= tp2_price:
                tp_size = 0.25
                pnl = (tp2_price - entry) / entry * 100 * tp_size
                realized_pnl += pnl
                tp_remaining -= tp_size
                tp2_hit = True
                if tp_remaining < 0.01:
                    exit_type = "TP_ALL"
                    break

            # TP3 (25%) — only if TP2 done but TP3 not
            if 0.01 <= tp_remaining <= 0.25 and high >= tp3_price:
                pnl = (tp3_price - entry) / entry * 100 * tp_remaining
                realized_pnl += pnl
                tp_remaining = 0
                tp3_hit = True
                exit_type = "TP_ALL"
                break

        elif direction == "SHORT":
            # Check SL
            if high >= sl_price:
                pnl = (entry - sl_price) / entry * 100 * tp_remaining
                realized_pnl += pnl
                exit_type = "SL"
                break

            # TP1 (50%)
            if tp_remaining > 0.50 and low <= tp1_price:
                tp_size = 0.50
                pnl = (entry - tp1_price) / entry * 100 * tp_size
                realized_pnl += pnl
                tp_remaining -= tp_size
                tp1_hit = True

            # TP2 (25%)
            if 0.24 < tp_remaining <= 0.50 and low <= tp2_price:
                tp_size = 0.25
                pnl = (entry - tp2_price) / entry * 100 * tp_size
                realized_pnl += pnl
                tp_remaining -= tp_size
                tp2_hit = True
                if tp_remaining < 0.01:
                    exit_type = "TP_ALL"
                    break

            # TP3 (25%)
            if 0.01 <= tp_remaining <= 0.25 and low <= tp3_price:
                pnl = (entry - tp3_price) / entry * 100 * tp_remaining
                realized_pnl += pnl
                tp_remaining = 0
                tp3_hit = True
                exit_type = "TP_ALL"
                break

    else:
        # Timeout - close remaining at last price
        if tp_remaining > 0:
            last_close = df.iloc[min(entry_idx + 119, len(df) - 1)]['close']
            if direction == "LONG":
                pnl = (last_close - entry) / entry * 100 * tp_remaining
            else:
                pnl = (entry - last_close) / entry * 100 * tp_remaining
            realized_pnl += pnl
            exit_type = "TIMEOUT"

    if exit_type is None:
        exit_type = "TIMEOUT"

    return {
        'direction': direction,
        'entry': entry,
        'pnl_pct': realized_pnl,
        'exit_type': exit_type,
        'bars_held': bars_held,
        'tp1_hit': tp1_hit,
        'tp2_hit': tp2_hit,
        'tp3_hit': tp3_hit
    }


def send_telegram_report(total_trades, wins, losses, win_rate,
                          avg_win, avg_loss, profit_factor,
                          max_win_streak, max_loss_streak,
                          total_pnl, total_pnl_leveraged,
                          best_trade, worst_trade,
                          trades_df, exit_counts):
    """Send backtest report to Telegram."""
    BOT_TOKEN = os.environ.get("BOT_TOKEN")
    CHANNEL_ID = os.environ.get("CHANNEL_ID")

    if not BOT_TOKEN or not CHANNEL_ID:
        print("\n  [SKIP] Telegram - BOT_TOKEN or CHANNEL_ID not set")
        return

    pnl_emoji = "📈" if total_pnl > 0 else "📉"
    wr_emoji = "✅" if win_rate >= 50 else "⚠️"

    # TP Hit Stats
    tp1_count = int(trades_df['tp1_hit'].sum())
    tp2_count = int(trades_df['tp2_hit'].sum())
    tp3_count = int(trades_df['tp3_hit'].sum())
    no_tp = total_trades - tp1_count

    # Build per-coin lines
    coin_lines = ""
    for sym in SYMBOLS:
        ct = trades_df[trades_df['symbol'] == sym]
        if len(ct) == 0:
            coin_lines += f"  <code>{sym}</code>  -  no trades\n"
            continue
        cw = len(ct[ct['pnl_pct'] > 0])
        cwr = (cw / len(ct)) * 100
        cp = ct['pnl_pct'].sum()
        coin_lines += f"  <code>{sym}</code>  {len(ct):>2} trades  {cwr:>5.1f}%  {cp:>+6.2f}%\n"

    # Exit breakdown
    exit_lines = ""
    for etype, count in exit_counts.items():
        et = trades_df[trades_df['exit_type'] == etype]
        avg_p = et['pnl_pct'].mean()
        exit_lines += f"  {etype}: {count} ({avg_p:+.2f}%)\n"

    text = (
        f"╔══════════════════════════════════════╗\n"
        f"║   BACKTEST REPORT - SCALP STRATEGY   ║\n"
        f"╠══════════════════════════════════════╣\n"
        f"\n"
        f"  {pnl_emoji}  <b>Total PnL:</b>  <code>{total_pnl:+.2f}%</code>  "
        f"(<code>{total_pnl_leveraged:+.1f}% x10</code>)\n"
        f"  {wr_emoji}  <b>Win Rate:</b>  <code>{win_rate:.1f}%</code>  "
        f"({wins}W / {losses}L)\n"
        f"  📊  <b>Total Trades:</b>  <code>{total_trades}</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📋  <b>Statistics:</b>\n"
        f"  ┆  Avg Win:  <code>+{avg_win:.2f}%</code>\n"
        f"  ┆  Avg Loss:  <code>-{avg_loss:.2f}%</code>\n"
        f"  ┆  Profit Factor:  <code>{profit_factor:.2f}</code>\n"
        f"  ┆  Best Streak:  <code>{max_win_streak}W</code>\n"
        f"  ┆  Worst Streak:  <code>{max_loss_streak}L</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🎯  <b>TP Hit Rate:</b>\n"
        f"  ┆  TP1 ({TP1_PCT}%):  <code>{tp1_count}/{total_trades}</code>  ({tp1_count/total_trades*100:.0f}%)\n"
        f"  ┆  TP2 ({TP2_PCT}%):  <code>{tp2_count}/{total_trades}</code>  ({tp2_count/total_trades*100:.0f}%)\n"
        f"  ┆  TP3 ({TP3_PCT}%):  <code>{tp3_count}/{total_trades}</code>  ({tp3_count/total_trades*100:.0f}%)\n"
        f"  ┆  No TP:         <code>{no_tp}/{total_trades}</code>  ({no_tp/total_trades*100:.0f}%)\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🏆  <b>Best:</b>  <code>{best_trade['symbol']} {best_trade['direction']} ({best_trade['pnl_pct']:+.2f}%)</code>\n"
        f"  💀  <b>Worst:</b>  <code>{worst_trade['symbol']} {worst_trade['direction']} ({worst_trade['pnl_pct']:+.2f}%)</code>\n"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📌  <b>Exit Breakdown:</b>\n"
        f"{exit_lines}"
        f"\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  💰  <b>Per-Coin:</b>\n"
        f"{coin_lines}"
        f"\n"
        f"╚══════════════════════════════════════╝\n"
        f"\n"
        f"  <i>Strategy: RSI(14) + EMA 150/200 | 1H | SL 1.5%</i>\n"
        f"  <i>TP1: 1.0% (50%) | TP2: 3.0% (25%) | TP3: 6.0% (25%) | SL: 1.5%</i>"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHANNEL_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.json().get('ok'):
            print("\n  ✅ Report sent to Telegram!")
        else:
            print(f"\n  ❌ Telegram error: {response.json().get('description')}")
    except Exception as e:
        print(f"\n  ❌ Network error: {e}")


if __name__ == "__main__":
    run_backtest()