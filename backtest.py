"""
=================================================================
  Bulls Signals Bot Backtest 1 year— 100% Compatible with scalp_bot
=================================================================
  BTC/USDT | 1H | RSI(14) OB=65 OS=35 | EMA 150/200
  SL: 2.5% | TP1: 1%(50%) | TP2: 3%(25%) | TP3: 4%(25%)
  Breakeven after TP1 | Cooldown: 10 bars | Timeout: 168h (7 days)
  Sends results to Telegram channel after completion

  Usage:
      backtest_scalp.py          <- 1 year (365 days)
      backtest_scalp.py 6        <- 6 months
      backtest_scalp.py 3        <- 3 months
      backtest_scalp.py 12       <- 1 year
=================================================================
"""

import os
import ccxt
import numpy as np
import time
import sys
import requests
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
#  Telegram Configuration (same as scalp_bot.py)
# ═══════════════════════════════════════════════════════════════
BOT_TOKEN  = os.environ.get("BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "")

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
EMA_FAR_THRESHOLD = 1.5
MAX_EMA200_DIST   = 3.0

COOLDOWN_BARS = 10

SL_PCT  = 2.5
TP1_PCT = 1.0
TP2_PCT = 3.0
TP3_PCT = 4.0

TP1_ALLOC = 0.50
TP2_ALLOC = 0.25
TP3_ALLOC = 0.25

TIMEOUT_BARS = 168

DEFAULT_MONTHS = 12

DAY_NAMES = {
    0: 'Mon', 1: 'Tue', 2: 'Wed',
    3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}


# ═══════════════════════════════════════════════════════════════
#  Telegram Send Function
# ═══════════════════════════════════════════════════════════════
def send_telegram(text):
    """Send message to Telegram channel (splits if > 4096 chars)"""
    if not BOT_TOKEN or not CHANNEL_ID:
        print("  [TG] BOT_TOKEN or CHANNEL_ID not set — skipping Telegram send")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    chunks = []
    while len(text) > 4096:
        split_at = text.rfind('\n', 0, 4096)
        if split_at < 2000:
            split_at = 4096
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    chunks.append(text)

    success = True
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": CHANNEL_ID,
            "text": chunk,
            "parse_mode": "HTML"
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.json().get('ok'):
                print(f"  [TG] Part {i+1}/{len(chunks)} sent successfully")
            else:
                print(f"  [TG] Error: {resp.json().get('description')}")
                success = False
            if len(chunks) > 1:
                time.sleep(1)
        except Exception as e:
            print(f"  [TG] Network error: {e}")
            success = False

    return success


# ═══════════════════════════════════════════════════════════════
#  Indicator Calculations
# ═══════════════════════════════════════════════════════════════
def calc_ema(data, length):
    alpha = 2.0 / (length + 1)
    result = np.empty(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def calc_rsi(data, length):
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
    signals = []
    last_buy_idx = -COOLDOWN_BARS - 1
    last_sell_idx = -COOLDOWN_BARS - 1

    for i in range(EMA_SLOW + 1, len(close)):
        if (np.isnan(rsi[i]) or np.isnan(rsi[i - 1]) or
                np.isnan(ema150[i]) or np.isnan(ema200[i])):
            continue

        ema_dist = abs((ema150[i] - ema200[i]) / ema200[i]) * 100
        if ema_dist >= EMA_FAR_THRESHOLD:
            continue

        price_dist = abs((close[i] - ema200[i]) / ema200[i]) * 100
        if price_dist > MAX_EMA200_DIST:
            continue

        rsi_cross_up   = rsi[i] > RSI_OVERBOUGHT and rsi[i - 1] <= RSI_OVERBOUGHT
        rsi_cross_down  = rsi[i] < RSI_OVERSOLD   and rsi[i - 1] >= RSI_OVERSOLD

        if (rsi_cross_up and
                ema150[i] > ema200[i] and
                close[i] > ema150[i] and
                (i - last_buy_idx) > COOLDOWN_BARS):
            signals.append((i, 'BUY'))
            last_buy_idx = i

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

        end_bar = min(bar_idx + TIMEOUT_BARS, len(close))

        for j in range(bar_idx + 1, end_bar):
            if direction == 'BUY':
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

                if tp1_hit:
                    if low[j] <= entry_price:
                        be_hit = True
                        remaining = 0
                        exit_reason = 'breakeven'
                        break
                else:
                    if low[j] <= sl_price:
                        sl_hit = True
                        pnl += -SL_PCT * remaining
                        remaining = 0
                        exit_reason = 'SL'
                        break

            else:  # SELL
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

                if tp1_hit:
                    if high[j] >= entry_price:
                        be_hit = True
                        remaining = 0
                        exit_reason = 'breakeven'
                        break
                else:
                    if high[j] >= sl_price:
                        sl_hit = True
                        pnl += -SL_PCT * remaining
                        remaining = 0
                        exit_reason = 'SL'
                        break

            if remaining <= 0.01:
                exit_reason = 'all_TP'
                break

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
#  Build Telegram Report
# ═══════════════════════════════════════════════════════════════
def build_telegram_report(trades, months):
    """Build formatted Telegram message (split into parts if needed)"""
    total = len(trades)
    if total == 0:
        return ["<b>SCALP BOT BACKTEST</b>\n\nNo trades found!"]

    days = months * 30
    wins   = sum(1 for t in trades if t['pnl'] > 0)
    losses = total - wins
    wr     = wins / total * 100

    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl   = total_pnl / total

    gross_profit  = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss    = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    cum_pnl = np.cumsum([t['pnl'] for t in trades])
    max_dd = 0
    peak = cum_pnl[0]
    for v in cum_pnl:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    max_cw = max_cl = cw = cl = 0
    for t in trades:
        if t['pnl'] > 0:
            cw += 1; cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0
            max_cl = max(max_cl, cl)

    sl_exits  = sum(1 for t in trades if t['exit'] == 'SL')
    be_exits  = sum(1 for t in trades if t['exit'] == 'breakeven')
    tp_exits  = sum(1 for t in trades if t['exit'] == 'all_TP')
    to_exits  = sum(1 for t in trades if t['exit'] == 'timeout')
    avg_tps   = np.mean([t['tps'] for t in trades])

    buys  = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']

    monthly = {}
    for t in trades:
        m = t['time'].strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'trades': [], 'pnl': 0}
        monthly[m]['trades'].append(t)
        monthly[m]['pnl'] += t['pnl']

    pnl_icon = "📈" if total_pnl > 0 else "📉"
    wr_icon  = "🟢" if wr >= 70 else ("🟡" if wr >= 50 else "🔴")

    part1 = (
        f"<b>{pnl_icon} SCALP BOT BACKTEST</b>\n"
        f"<b>{SYMBOL}</b> | {months} Month(s) | 1H\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"<b>⚙️ Parameters:</b>\n"
        f"  ┆ SL: <code>{SL_PCT}%</code> | TP: <code>{TP1_PCT}/{TP2_PCT}/{TP3_PCT}%</code>\n"
        f"  ┆ RSI({RSI_LENGTH}) OB={RSI_OVERBOUGHT}/OS={RSI_OVERSOLD}\n"
        f"  ┆ EMA {EMA_FAST}/{EMA_SLOW} | CD={COOLDOWN_BARS}h\n\n"
        f"<b>{wr_icon} Performance:</b>\n"
        f"  ┆ Total Trades: <code>{total}</code>\n"
        f"  ┆ Win Rate: <code>{wr:.1f}%</code> ({wins}W/{losses}L)\n"
        f"  ┆ Total PnL: <code>{total_pnl:+.2f}%</code>\n"
        f"  ┆ Monthly Avg: <code>{total_pnl/(days/30):+.2f}%/mo</code>\n"
        f"  ┆ Avg PnL/Trade: <code>{avg_pnl:+.3f}%</code>\n"
        f"  ┆ Profit Factor: <code>{profit_factor:.2f}</code>\n"
        f"  ┆ Max Drawdown: <code>-{max_dd:.2f}%</code>\n"
        f"  ┆ Consec Win/Loss: <code>{max_cw}/{max_cl}</code>\n"
        f"  ┆ Avg TPs Hit: <code>{avg_tps:.1f}/3</code>\n\n"
        f"<b>🚪 Exit Reasons:</b>\n"
        f"  ┆ Stop Loss: <code>{sl_exits}</code> ({sl_exits/total*100:.1f}%)\n"
        f"  ┆ Breakeven: <code>{be_exits}</code> ({be_exits/total*100:.1f}%)\n"
        f"  ┆ All TPs: <code>{tp_exits}</code> ({tp_exits/total*100:.1f}%)\n"
        f"  ┆ Timeout: <code>{to_exits}</code> ({to_exits/total*100:.1f}%)\n"
    )

    if buys:
        b_pnl = sum(t['pnl'] for t in buys)
        b_wins = sum(1 for t in buys if t['pnl'] > 0)
        part1 += f"\n  🟢 LONG: <code>{len(buys)}</code> | WR: <code>{b_wins/len(buys)*100:.1f}%</code> | PnL: <code>{b_pnl:+.2f}%</code>\n"
    if sells:
        s_pnl = sum(t['pnl'] for t in sells)
        s_wins = sum(1 for t in sells if t['pnl'] > 0)
        part1 += f"  🔴 SHORT: <code>{len(sells)}</code> | WR: <code>{s_wins/len(sells)*100:.1f}%</code> | PnL: <code>{s_pnl:+.2f}%</code>\n"

    part2 = (
        f"<b>📅 Monthly Breakdown:</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    for m in sorted(monthly.keys()):
        mt = monthly[m]['trades']
        mw = sum(1 for t in mt if t['pnl'] > 0)
        mp = monthly[m]['pnl']
        marker = "🔥" if mp > 3 else ("✅" if mp > 0 else "❌")
        part2 += (
            f"  <code>{m}</code> | {len(mt)} trades | "
            f"WR: <code>{mw/len(mt)*100:.0f}%</code> | "
            f"PnL: <code>{mp:+.2f}%</code> {marker}\n"
        )

    sl_trades_list = [t for t in trades if t['exit'] == 'SL']
    if sl_trades_list:
        part3 = (
            f"\n<b>❌ Stop Loss Trades ({len(sl_trades_list)}):</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        for t in sl_trades_list:
            dir_label = 'LONG' if t['dir'] == 'BUY' else 'SHORT'
            day_name = DAY_NAMES[t['time'].weekday()]
            part3 += (
                f"  {t['time'].strftime('%m-%d %H:%M')} ({day_name}) "
                f"<code>{dir_label}</code> @ <code>{t['entry']:.0f}</code> "
                f"→ <code>{t['pnl']:+.2f}%</code>\n"
            )
    else:
        part3 = ""

    part4 = (
        f"\n<b>📋 All Trades:</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    for i, t in enumerate(trades):
        dir_label = 'L' if t['dir'] == 'BUY' else 'S'
        exit_short = {
            'SL': 'SL', 'breakeven': 'BE',
            'all_TP': 'TPs', 'timeout': 'TO'
        }.get(t['exit'], t['exit'])
        pnl_icon_t = "✅" if t['pnl'] > 0 else "❌"
        part4 += (
            f"  {i+1:>2}. {t['time'].strftime('%m-%d %H:%M')} "
            f"<code>{dir_label}</code> {t['entry']:.0f} "
            f"TPs:{t['tps']} {exit_short} "
            f"{pnl_icon_t} <code>{t['pnl']:+.2f}%</code>\n"
        )

    footer = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 Dr Python Scalp Bot | ⏱ 1H\n"
        f"<i>Backtest results — Not financial advice</i>"
    )

    parts = []
    current = part1 + part2 + part3
    parts.append(current)

    trade_lines = part4 + footer
    if len(trade_lines) <= 4096:
        parts.append(trade_lines)
    else:
        header = f"<b>📋 All Trades:</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        lines = part4.split('\n')
        trade_only_lines = []
        for line in lines:
            if line.strip() and not line.startswith('<b>📋') and not line.startswith('━━━'):
                trade_only_lines.append(line)

        chunk = header
        chunk_count = 1
        for line in trade_only_lines:
            if len(chunk) + len(line) + 1 > 4000:
                parts.append(chunk)
                chunk_count += 1
                chunk = f"<b>📋 All Trades (cont. {chunk_count}):</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            chunk += line + '\n'

        chunk += footer
        parts.append(chunk)

    return parts


# ═══════════════════════════════════════════════════════════════
#  Statistics & Reports (Console)
# ═══════════════════════════════════════════════════════════════
def print_report(trades, months):
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

    cum_pnl = np.cumsum([t['pnl'] for t in trades])
    max_dd = 0
    peak = cum_pnl[0]
    for v in cum_pnl:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    max_cw = max_cl = cw = cl = 0
    for t in trades:
        if t['pnl'] > 0:
            cw += 1; cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0
            max_cl = max(max_cl, cl)

    sl_exits  = sum(1 for t in trades if t['exit'] == 'SL')
    be_exits  = sum(1 for t in trades if t['exit'] == 'breakeven')
    tp_exits  = sum(1 for t in trades if t['exit'] == 'all_TP')
    to_exits  = sum(1 for t in trades if t['exit'] == 'timeout')
    avg_tps   = np.mean([t['tps'] for t in trades])

    buys  = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']

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

    sl_trades = [t for t in trades if t['exit'] == 'SL']
    if sl_trades:
        print(f"\n  Stop Loss Trades ({len(sl_trades)}):")
        print(f"  -----------------------------------------------")
        for t in sl_trades:
            dir_label = 'LONG' if t['dir'] == 'BUY' else 'SHORT'
            day_name = DAY_NAMES[t['time'].weekday()]
            print(f"    {t['time'].strftime('%Y-%m-%d %H:%M')} ({day_name}) "
                  f"{dir_label} @ {t['entry']:.0f} -> {t['pnl']:+.3f}%")

    be_trades = [t for t in trades if t['exit'] == 'breakeven']
    if be_trades:
        print(f"\n  Breakeven Trades ({len(be_trades)}):")
        print(f"  -----------------------------------------------")
        for t in be_trades:
            dir_label = 'LONG' if t['dir'] == 'BUY' else 'SHORT'
            day_name = DAY_NAMES[t['time'].weekday()]
            print(f"    {t['time'].strftime('%Y-%m-%d %H:%M')} ({day_name}) "
                  f"{dir_label} @ {t['entry']:.0f} | TPs: {t['tps']}/3 -> {t['pnl']:+.3f}%")

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

    print(f"\n{'=' * 64}")
    print(f"  SUMMARY: {total} trades | WR: {wr:.1f}% | "
          f"PF: {profit_factor:.2f} | PnL: {total_pnl:+.2f}% | "
          f"MaxDD: -{max_dd:.2f}%")
    print(f"{'=' * 64}")


# ═══════════════════════════════════════════════════════════════
#  Main — Run the Backtest
# ═══════════════════════════════════════════════════════════════
def main():
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
    print(f"  Telegram: {'ON' if BOT_TOKEN and CHANNEL_ID else 'OFF (no credentials)'}")
    print(f"{'=' * 64}")

    exchange = ccxt.mexc({'enableRateLimit': True})

    until_ms = int(time.time() * 1000)
    since_ms = until_ms - days * 24 * 3600 * 1000

    print(f"\n  Fetching historical data...")
    candles = fetch_ohlcv(exchange, SYMBOL, since_ms, until_ms)

    if len(candles) < 250:
        print("  Insufficient data!")
        sys.exit(1)

    close      = np.array([c[4] for c in candles], dtype=float)
    high       = np.array([c[2] for c in candles], dtype=float)
    low        = np.array([c[3] for c in candles], dtype=float)
    timestamps = [c[0] for c in candles]

    print(f"  Calculating indicators... ({len(candles)} candles, {len(candles)/24:.0f} days)")
    rsi    = calc_rsi(close, RSI_LENGTH)
    ema150 = calc_ema(close, EMA_FAST)
    ema200 = calc_ema(close, EMA_SLOW)

    print(f"  Detecting signals...")
    signals = find_signals(close, high, low, timestamps, rsi, ema150, ema200)
    print(f"  Found {len(signals)} signal(s)")

    print(f"  Simulating trades...")
    trades = simulate_trades(close, high, low, timestamps, signals)

    print_report(trades, months)

    # Send report to Telegram
    if BOT_TOKEN and CHANNEL_ID:
        print(f"\n  Sending results to Telegram...")
        tg_parts = build_telegram_report(trades, months)
        for i, part in enumerate(tg_parts):
            send_telegram(part)
            if i < len(tg_parts) - 1:
                time.sleep(1)
        print(f"  Done! Sent {len(tg_parts)} message(s) to Telegram")
    else:
        print(f"\n  [TG] Skipped — BOT_TOKEN or CHANNEL_ID not set")


if __name__ == "__main__":
    main()
