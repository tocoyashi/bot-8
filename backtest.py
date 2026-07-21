"""
=================================================================
  Bulls Bot Backtest — 100% Compatible with scalp_bot.py
=================================================================
  BTC/USDT | 1H | RSI(14) OB=65 OS=35 | EMA 150/200
  SL: 2.5% | TP1: 1%(50%) | TP2: 3%(25%) | TP3: 4%(25%)
  Breakeven after TP1 | Cooldown: 10 bars | Timeout: 168h (7 days)

  Usage:
      backtest_scalp.py          <- 1 year (365 days)
      backtest_scalp.py 6        <- 6 months
      backtest_scalp.py 3        <- 3 months
      backtest_scalp.py 12       <- 1 year
=================================================================
"""

import ccxt
import numpy as np
import time
import sys
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
#  Parameters — 100% matching scalp_bot.py
# ═══════════════════════════════════════════════════════════════
SYMBOL    = 'BTC/USDT'
TIMEFRAME = '1h'

RSI_LENGTH     = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD   = 35

EMA_FAST          = 150
EMA_SLOW          = 200
EMA_FAR_THRESHOLD = 1.5      # below this = EMAs are close (mature trend)
MAX_EMA200_DIST   = 3.0      # below this = price is near EMA200

COOLDOWN_BARS = 10           # 10 hours

SL_PCT  = 2.5
TP1_PCT = 1.0
TP2_PCT = 3.0
TP3_PCT = 4.0

TP1_ALLOC = 0.50
TP2_ALLOC = 0.25
TP3_ALLOC = 0.25

TIMEOUT_BARS = 168           # 7 days x 24 hours = 168 candles

# Default test period
DEFAULT_MONTHS = 12

# Weekday names
DAY_NAMES = {
    0: 'Mon', 1: 'Tue', 2: 'Wed',
    3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}


# ═══════════════════════════════════════════════════════════════
#  Indicator Calculations
# ═══════════════════════════════════════════════════════════════
def calc_ema(data, length):
    """Calculate EMA manually — matches pandas ewm"""
    alpha = 2.0 / (length + 1)
    result = np.empty(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def calc_rsi(data, length):
    """Calculate RSI manually — matches ta.momentum.rsi"""
    result = np.full(len(data), np.nan)
    if len(data) < length + 1:
        return result

    delta = np.diff(data)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])

    if avg_loss == 0:
        result[length] = 100
    else:
        result[length] = 100 - (100 / (1 + avg_gain / avg_loss))

    for i in range(length, len(delta)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        if avg_loss == 0:
            result[i + 1] = 100
        else:
            result[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))

    return result


# ═══════════════════════════════════════════════════════════════
#  Historical Data Fetching
# ═══════════════════════════════════════════════════════════════
def fetch_ohlcv(exchange, symbol, since_ms, until_ms):
    """Fetch full OHLCV data with deduplication"""
    candles = []
    cursor = since_ms

    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol, TIMEFRAME, since=cursor, limit=1000
            )
            if not batch:
                break
            candles.extend(batch)
            cursor = batch[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception:
            time.sleep(2)

    # Deduplicate
    candles.sort(key=lambda x: x[0])
    seen = set()
    unique = []
    for c in candles:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)

    return unique


# ═══════════════════════════════════════════════════════════════
#  Signal Detection — 100% matching scalp_bot.py logic
# ═══════════════════════════════════════════════════════════════
def find_signals(close, high, low, timestamps, rsi, ema150, ema200):
    """
    Detect signals using the same logic as the bot:
      - RSI cross above OB or below OS
      - EMA150 > EMA200 for BUY | EMA150 < EMA200 for SELL
      - Price above EMA150 for BUY | below EMA150 for SELL
      - EMA distance < EMA_FAR_THRESHOLD
      - Price/EMA200 distance < MAX_EMA200_DIST
      - Cooldown: no duplicate signal within 10 bars
    """
    signals = []
    last_buy_idx = -COOLDOWN_BARS - 1
    last_sell_idx = -COOLDOWN_BARS - 1

    for i in range(EMA_SLOW + 1, len(close)):
        # Validate values
        if (np.isnan(rsi[i]) or np.isnan(rsi[i - 1]) or
                np.isnan(ema150[i]) or np.isnan(ema200[i])):
            continue

        # EMA150/200 distance filter
        ema_dist = abs((ema150[i] - ema200[i]) / ema200[i]) * 100
        if ema_dist >= EMA_FAR_THRESHOLD:
            continue

        # Price/EMA200 distance filter
        price_dist = abs((close[i] - ema200[i]) / ema200[i]) * 100
        if price_dist > MAX_EMA200_DIST:
            continue

        # RSI cross detection
        rsi_cross_up   = rsi[i] > RSI_OVERBOUGHT and rsi[i - 1] <= RSI_OVERBOUGHT
        rsi_cross_down  = rsi[i] < RSI_OVERSOLD   and rsi[i - 1] >= RSI_OVERSOLD

        # BUY signal
        if (rsi_cross_up and
                ema150[i] > ema200[i] and
                close[i] > ema150[i] and
                (i - last_buy_idx) > COOLDOWN_BARS):
            signals.append((i, 'BUY'))
            last_buy_idx = i

        # SELL signal
        elif (rsi_cross_down and
                  ema150[i] < ema200[i] and
                  close[i] < ema150[i] and
                  (i - last_sell_idx) > COOLDOWN_BARS):
            signals.append((i, 'SELL'))
            last_sell_idx = i

    return signals


# ═══════════════════════════════════════════════════════════════
#  Trade Simulation — 100% matching scalp_bot.py
# ═══════════════════════════════════════════════════════════════
def simulate_trades(close, high, low, timestamps, signals):
    """
    Simulate each trade using the same logic as the bot:
      - TP1: 1% -> close 50%, move remaining to breakeven
      - TP2: 3% -> close 25%
      - TP3: 4% -> close 25%
      - After TP1: SL becomes breakeven (entry price)
      - Before TP1: SL = 2.5%
      - Timeout: 168 candles (7 days)
    """
    trades = []

    for bar_idx, direction in signals:
        entry_price = close[bar_idx]
        remaining   = 1.0
        pnl         = 0.0
        tp1_hit = tp2_hit = tp3_hit = sl_hit = be_hit = False
        exit_reason = 'timeout'

        if direction == 'BUY':
            tp1_price = entry_price * (1 + TP1_PCT / 100)
            tp2_price = entry_price * (1 + TP2_PCT / 100)
            tp3_price = entry_price * (1 + TP3_PCT / 100)
            sl_price  = entry_price * (1 - SL_PCT / 100)
        else:
            tp1_price = entry_price * (1 - TP1_PCT / 100)
            tp2_price = entry_price * (1 - TP2_PCT / 100)
            tp3_price = entry_price * (1 - TP3_PCT / 100)
            sl_price  = entry_price * (1 + SL_PCT / 100)

        # Scan forward candles
        end_bar = min(bar_idx + TIMEOUT_BARS, len(close))

        for j in range(bar_idx + 1, end_bar):
            if direction == 'BUY':
                # Check TPs
                if not tp1_hit and high[j] >= tp1_price:
                    tp1_hit = True
                    pnl += TP1_PCT * TP1_ALLOC
                    remaining -= TP1_ALLOC

                if not tp2_hit and high[j] >= tp2_price:
                    tp2_hit = True
                    pnl += TP2_PCT * TP2_ALLOC
                    remaining -= TP2_ALLOC

                if not tp3_hit and high[j] >= tp3_price:
                    tp3_hit = True
                    pnl += TP3_PCT * TP3_ALLOC
                    remaining -= TP3_ALLOC

                # Check SL / Breakeven
                if tp1_hit:
                    # After TP1: SL = breakeven (entry price)
                    if low[j] <= entry_price:
                        be_hit = True
                        remaining = 0
                        exit_reason = 'breakeven'
                        break
                else:
                    # Before TP1: SL = 2.5%
                    if low[j] <= sl_price:
                        sl_hit = True
                        pnl += -SL_PCT * remaining
                        remaining = 0
                        exit_reason = 'SL'
                        break

            else:  # SELL
                # Check TPs
                if not tp1_hit and low[j] <= tp1_price:
                    tp1_hit = True
                    pnl += TP1_PCT * TP1_ALLOC
                    remaining -= TP1_ALLOC

                if not tp2_hit and low[j] <= tp2_price:
                    tp2_hit = True
                    pnl += TP2_PCT * TP2_ALLOC
                    remaining -= TP2_ALLOC

                if not tp3_hit and low[j] <= tp3_price:
                    tp3_hit = True
                    pnl += TP3_PCT * TP3_ALLOC
                    remaining -= TP3_ALLOC

                # Check SL / Breakeven
                if tp1_hit:
                    # After TP1: SL = breakeven
                    if high[j] >= entry_price:
                        be_hit = True
                        remaining = 0
                        exit_reason = 'breakeven'
                        break
                else:
                    # Before TP1: SL = 2.5%
                    if high[j] >= sl_price:
                        sl_hit = True
                        pnl += -SL_PCT * remaining
                        remaining = 0
                        exit_reason = 'SL'
                        break

            # All targets hit
            if remaining <= 0.01:
                exit_reason = 'all_TP'
                break

        # Timeout — close at last candle's close price
        if remaining > 0.01:
            last_price = close[min(bar_idx + TIMEOUT_BARS - 1, len(close) - 1)]
            if direction == 'BUY':
                change = (last_price - entry_price) / entry_price * 100
            else:
                change = (entry_price - last_price) / entry_price * 100
            pnl += change * remaining

        tps_hit_count = sum([tp1_hit, tp2_hit, tp3_hit])

        trades.append({
            'dir':     direction,
            'entry':   entry_price,
            'bar_idx': bar_idx,
            'time':    datetime.fromtimestamp(
                           timestamps[bar_idx] / 1000, tz=timezone.utc
                       ),
            'tps':     tps_hit_count,
            'exit':    exit_reason,
            'pnl':     pnl,
        })

    return trades


# ═══════════════════════════════════════════════════════════════
#  Statistics & Reports
# ═══════════════════════════════════════════════════════════════
def print_report(trades, months):
    """Print comprehensive backtest report"""
    total = len(trades)
    if total == 0:
        print("\n  No trades found!")
        return

    days = months * 30
    wins   = sum(1 for t in trades if t['pnl'] > 0)
    losses = total - wins
    wr     = wins / total * 100

    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl   = total_pnl / total

    gross_profit  = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss    = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    # Max Drawdown
    cum_pnl = np.cumsum([t['pnl'] for t in trades])
    max_dd = 0
    peak = cum_pnl[0]
    for v in cum_pnl:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    # Consecutive wins/losses
    max_cw = max_cl = cw = cl = 0
    for t in trades:
        if t['pnl'] > 0:
            cw += 1; cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0
            max_cl = max(max_cl, cl)

    # Exit reasons
    sl_exits  = sum(1 for t in trades if t['exit'] == 'SL')
    be_exits  = sum(1 for t in trades if t['exit'] == 'breakeven')
    tp_exits  = sum(1 for t in trades if t['exit'] == 'all_TP')
    to_exits  = sum(1 for t in trades if t['exit'] == 'timeout')
    avg_tps   = np.mean([t['tps'] for t in trades])

    # BUY/SELL breakdown
    buys  = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']

    # ═══ Print Results ═══
    print(f"\n{'=' * 64}")
    print(f"  SCALP BOT BACKTEST — {SYMBOL} | {months} Month(s)")
    print(f"  SL={SL_PCT}% | TP1={TP1_PCT}%({int(TP1_ALLOC*100)}%) | "
          f"TP2={TP2_PCT}%({int(TP2_ALLOC*100)}%) | TP3={TP3_PCT}%({int(TP3_ALLOC*100)}%)")
    print(f"  RSI({RSI_LENGTH}) OB={RSI_OVERBOUGHT}/OS={RSI_OVERSOLD} | "
          f"EMA {EMA_FAST}/{EMA_SLOW} | CD={COOLDOWN_BARS}h | Timeout={TIMEOUT_BARS}h")
    print(f"{'=' * 64}")

    print(f"\n  Overall Performance:")
    print(f"  -----------------------------------------------")
    print(f"  Total Trades:      {total}")
    print(f"  Daily Signals:     {total/days:.2f}/day")
    print(f"  Weekly Signals:    {total/(days/7):.1f}/week")
    print(f"  Win Rate:          {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg PnL/Trade:     {avg_pnl:+.3f}%")
    print(f"  Total PnL:         {total_pnl:+.2f}%")
    print(f"  Monthly Avg:       {total_pnl/(days/30):+.2f}%/month")
    print(f"  Profit Factor:     {profit_factor:.2f}")
    print(f"  Max Drawdown:      -{max_dd:.2f}%")
    print(f"  Max Consec Win:    {max_cw}")
    print(f"  Max Consec Loss:   {max_cl}")
    print(f"  Avg TPs Hit:       {avg_tps:.1f}/3")

    print(f"\n  Exit Reasons:")
    print(f"  -----------------------------------------------")
    print(f"    Stop Loss (SL):      {sl_exits:>3} ({sl_exits/total*100:.1f}%)")
    print(f"    Breakeven (BE):      {be_exits:>3} ({be_exits/total*100:.1f}%)")
    print(f"    All Targets (AllTP): {tp_exits:>3} ({tp_exits/total*100:.1f}%)")
    print(f"    Timeout (TO):        {to_exits:>3} ({to_exits/total*100:.1f}%)")

    if buys:
        b_pnl = sum(t['pnl'] for t in buys)
        b_wins = sum(1 for t in buys if t['pnl'] > 0)
        print(f"\n  LONG  {len(buys)} trades | WR: {b_wins/len(buys)*100:.1f}% | PnL: {b_pnl:+.2f}%")
    if sells:
        s_pnl = sum(t['pnl'] for t in sells)
        s_wins = sum(1 for t in sells if t['pnl'] > 0)
        print(f"  SHORT {len(sells)} trades | WR: {s_wins/len(sells)*100:.1f}% | PnL: {s_pnl:+.2f}%")

    # ═══ Monthly Breakdown ═══
    monthly = {}
    for t in trades:
        m = t['time'].strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'trades': [], 'pnl': 0}
        monthly[m]['trades'].append(t)
        monthly[m]['pnl'] += t['pnl']

    print(f"\n  Monthly Breakdown:")
    print(f"  -----------------------------------------------")
    print(f"  {'Month':<10} {'Trades':<8} {'WR%':<8} {'PnL%':<10}")
    for m in sorted(monthly.keys()):
        mt = monthly[m]['trades']
        mw = sum(1 for t in mt if t['pnl'] > 0)
        mp = monthly[m]['pnl']
        marker = "++" if mp > 3 else ("+" if mp > 0 else "-")
        print(f"  {m:<10} {len(mt):<8} {mw/len(mt)*100:<8.1f} {mp:+.2f} {marker}")

    # ═══ SL Trades Detail ═══
    sl_trades = [t for t in trades if t['exit'] == 'SL']
    if sl_trades:
        print(f"\n  Stop Loss Trades ({len(sl_trades)}):")
        print(f"  -----------------------------------------------")
        for t in sl_trades:
            dir_label = 'LONG' if t['dir'] == 'BUY' else 'SHORT'
            day_name = DAY_NAMES[t['time'].weekday()]
            print(f"    {t['time'].strftime('%Y-%m-%d %H:%M')} ({day_name}) "
                  f"{dir_label} @ {t['entry']:.0f} -> {t['pnl']:+.3f}%")

    # ═══ Breakeven Trades Detail ═══
    be_trades = [t for t in trades if t['exit'] == 'breakeven']
    if be_trades:
        print(f"\n  Breakeven Trades ({len(be_trades)}):")
        print(f"  -----------------------------------------------")
        for t in be_trades:
            dir_label = 'LONG' if t['dir'] == 'BUY' else 'SHORT'
            day_name = DAY_NAMES[t['time'].weekday()]
            print(f"    {t['time'].strftime('%Y-%m-%d %H:%M')} ({day_name}) "
                  f"{dir_label} @ {t['entry']:.0f} | TPs: {t['tps']}/3 -> {t['pnl']:+.3f}%")

    # ═══ All Trades Detail ═══
    print(f"\n  All Trades Detail:")
    print(f"  -----------------------------------------------")
    print(f"  {'#':<4} {'Date':<17} {'Dir':<7} {'Entry':<10} {'TPs':<4} {'Exit':<12} {'PnL%':<8}")
    print(f"  {'-'*4} {'-'*17} {'-'*7} {'-'*10} {'-'*4} {'-'*12} {'-'*8}")
    for i, t in enumerate(trades):
        dir_label = 'LONG' if t['dir'] == 'BUY' else 'SHORT'
        exit_label = {
            'SL': 'Stop Loss',
            'breakeven': 'Breakeven',
            'all_TP': 'All TPs',
            'timeout': 'Timeout'
        }.get(t['exit'], t['exit'])
        print(f"  {i+1:<4} {t['time'].strftime('%Y-%m-%d %H:%M'):<17} "
              f"{dir_label:<7} {t['entry']:<10.0f} {t['tps']:<4} "
              f"{exit_label:<12} {t['pnl']:+.3f}")

    # ═══ Summary ═══
    print(f"\n{'=' * 64}")
    print(f"  SUMMARY: {total} trades | WR: {wr:.1f}% | "
          f"PF: {profit_factor:.2f} | PnL: {total_pnl:+.2f}% | "
          f"MaxDD: -{max_dd:.2f}%")
    print(f"{'=' * 64}")


# ═══════════════════════════════════════════════════════════════
#  Main — Run the Backtest
# ═══════════════════════════════════════════════════════════════
def main():
    # Determine test period
    months = DEFAULT_MONTHS
    if len(sys.argv) > 1:
        try:
            months = int(sys.argv[1])
        except ValueError:
            print("  Usage: python backtest_scalp.py [months]")
            print("  Example: python backtest_scalp.py 6")
            sys.exit(1)

    days = months * 30

    print(f"\n{'=' * 64}")
    print(f"  SCALP BOT BACKTEST — {months} Month(s)")
    print(f"  Symbol: {SYMBOL} | Timeframe: {TIMEFRAME}")
    print(f"  SL: {SL_PCT}% | TP: {TP1_PCT}/{TP2_PCT}/{TP3_PCT}% | CD: {COOLDOWN_BARS}h")
    print(f"{'=' * 64}")

    # Connect to exchange
    exchange = ccxt.mexc({'enableRateLimit': True})

    until_ms = int(time.time() * 1000)
    since_ms = until_ms - days * 24 * 3600 * 1000

    print(f"\n  Fetching historical data...")
    candles = fetch_ohlcv(exchange, SYMBOL, since_ms, until_ms)

    if len(candles) < 250:
        print("  Insufficient data!")
        sys.exit(1)

    # Convert to arrays
    close      = np.array([c[4] for c in candles], dtype=float)
    high       = np.array([c[2] for c in candles], dtype=float)
    low        = np.array([c[3] for c in candles], dtype=float)
    timestamps = [c[0] for c in candles]

    # Calculate indicators
    print(f"  Calculating indicators... ({len(candles)} candles, {len(candles)/24:.0f} days)")
    rsi    = calc_rsi(close, RSI_LENGTH)
    ema150 = calc_ema(close, EMA_FAST)
    ema200 = calc_ema(close, EMA_SLOW)

    # Detect signals
    print(f"  Detecting signals...")
    signals = find_signals(close, high, low, timestamps, rsi, ema150, ema200)
    print(f"  Found {len(signals)} signal(s)")

    # Simulate trades
    print(f"  Simulating trades...")
    trades = simulate_trades(close, high, low, timestamps, signals)

    # Print report
    print_report(trades, months)


if __name__ == "__main__":
    main()
