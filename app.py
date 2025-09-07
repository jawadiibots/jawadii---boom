#!/usr/bin/env python3
"""
app.py - TradingView-style Analyzer + Monitoring Telegram Bot
Features:
- /analyze SYMBOL -> computes entry, SL, up to 5 TPs, replies with text + annotated chart
- Persists "open orders" to trades.json and monitors them in background (checks every minute)
- When a TP or SL is hit, it sends a message + screenshot and marks the trade as closed
- Designed to be deployed on Railway/Heroku (Procfile provided)
Env vars required:
- BOT_TOKEN (Telegram Bot token)
Optional env vars:
- CCXT_EXCHANGE (default: binance)
- POLL_INTERVAL (seconds, default: 60)
- OHLCV_LIMIT (candles for plotting, default 200)
Notes: Use test/demo funds; this bot provides informational signals only.
"""

import os, io, math, json, asyncio, logging, traceback
from datetime import datetime
from typing import Dict, Any, List

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from tradingview_ta import TA_Handler, Interval
import ccxt

from telegram import InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# ---------- Config ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Please set BOT_TOKEN environment variable.")

CCXT_EXCHANGE_ID = os.getenv("CCXT_EXCHANGE", "binance")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

TRADES_FILE = "trades.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Persistence helpers ----------
def load_trades() -> dict:
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    return {"open": [], "closed": []}

def save_trades(data: dict):
    with open(TRADES_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

# ---------- Market data helpers ----------
def fetch_ohlcv_ccxt(symbol_ccxt: str, timeframe='1h', limit=OHLCV_LIMIT):
    exchange_cls = getattr(ccxt, CCXT_EXCHANGE_ID)
    ex = exchange_cls({"enableRateLimit": True})
    try:
        ex.load_markets()
    except Exception:
        pass
    ohlcv = ex.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

def compute_atr(df, period=ATR_PERIOD):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

# ---------- Analysis core ----------
def analyze_symbol(tv_symbol: str, timeframe='1h'):
    """
    Returns a dict with entry, sl, tps, side, price, atr and a simple ohlcv dataframe (if available)
    """
    if ':' in tv_symbol:
        exchange, symbol = tv_symbol.split(':', 1)
    else:
        exchange = 'BINANCE'
        symbol = tv_symbol

    handler = TA_Handler(
        symbol=symbol,
        screener="crypto",
        exchange=exchange.upper(),
        interval=Interval.INTERVAL_1_HOUR if timeframe == '1h' else Interval.INTERVAL_1_DAY
    )

    analysis = None
    try:
        analysis = handler.get_analysis()
    except Exception as e:
        logger.info("tradingview_ta analysis failed: %s", e)

    current_price = None
    summary = {}
    if analysis:
        summary = analysis.summary
        indicators = analysis.indicators
        current_price = indicators.get("close") or indicators.get("last") or indicators.get("open")

    # Build ccxt symbol (e.g., BTCUSDT -> BTC/USDT)
    ccxt_symbol = None
    symbol_upper = symbol.upper()
    for suf in ['USDT','USD','BTC','ETH','EUR']:
        if symbol_upper.endswith(suf):
            base = symbol_upper[:-len(suf)]
            ccxt_symbol = f"{base}/{suf}"
            break
    if ccxt_symbol is None:
        ccxt_symbol = f"{symbol}/USDT"

    ohlcv = None
    try:
        ohlcv = fetch_ohlcv_ccxt(ccxt_symbol, timeframe=timeframe, limit=OHLCV_LIMIT)
    except Exception as e:
        logger.info("CCXT OHLCV fetch failed for %s: %s", ccxt_symbol, e)

    if ohlcv is None and current_price is not None:
        now = pd.Timestamp.utcnow()
        ohlcv = pd.DataFrame({'open':[current_price],'high':[current_price],'low':[current_price],'close':[current_price],'vol':[0]}, index=[now])

    atr = None
    if ohlcv is not None:
        atr_series = compute_atr(ohlcv, ATR_PERIOD)
        atr = float(atr_series.iloc[-1])

    # Determine side
    side = "wait"
    if summary:
        sig = summary.get("RECOMMENDATION") or summary.get("RECOMMENDATION")
        if sig:
            s = sig.lower()
            if "buy" in s:
                side = "long"
            elif "sell" in s:
                side = "short"
            else:
                side = "wait"
    else:
        if ohlcv is not None and len(ohlcv) >= 2:
            side = "long" if ohlcv['close'].iloc[-1] > ohlcv['close'].iloc[-2] else "short"

    price = None
    if ohlcv is not None:
        price = float(ohlcv['close'].iloc[-1])
    if current_price is not None:
        price = float(current_price)

    if price is None:
        return {"error":"Could not fetch price."}

    # ATR-based SL
    atr_mult = 1.5 * atr if atr else 0.01 * price
    if side == 'long':
        entry = price
        sl = price - atr_mult
        risk = entry - sl
        tps = [entry + risk * r for r in [1,1.5,2,3,5]]
    elif side == 'short':
        entry = price
        sl = price + atr_mult
        risk = sl - entry
        tps = [entry - risk * r for r in [1,1.5,2,3,5]]
    else:
        entry = price
        sl = price - (atr_mult if atr else price*0.01)
        tps = [price + price * pct for pct in [0.005, 0.01, 0.02]]

    def nice_round(x):
        if x is None or (isinstance(x,float) and math.isnan(x)):
            return None
        if x >= 10:
            return round(x,2)
        elif x >= 1:
            return round(x,4)
        else:
            return round(x,6)

    result = {
        "symbol": tv_symbol,
        "side": side,
        "price": nice_round(price),
        "atr": nice_round(atr) if atr else None,
        "entry": nice_round(entry),
        "sl": nice_round(sl),
        "tps": [nice_round(x) for x in tps][:5],
        "ohlcv": ohlcv.reset_index().to_dict(orient="records") if ohlcv is not None else None,
        "ccxt_symbol": ccxt_symbol
    }
    return result

# ---------- Chart / image ----------
def plot_levels_image(analysis_result, width=1200, height=600):
    # Recreate df if present
    df = None
    if analysis_result.get("ohlcv"):
        df = pd.DataFrame(analysis_result["ohlcv"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    entry = analysis_result.get('entry')
    sl = analysis_result.get('sl')
    tps = analysis_result.get('tps', [])
    side = analysis_result.get('side')
    symbol = analysis_result.get('symbol')

    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    if df is not None and len(df) > 1:
        ax.plot(df.index, df['close'], linewidth=1)
        ax.set_xlim(df.index[-min(len(df), 200)], df.index[-1])
    else:
        now = pd.Timestamp.utcnow()
        ax.plot([now], [entry], marker='o')

    if entry is not None:
        ax.axhline(entry, linestyle='--', linewidth=1.2, label=f'Entry @ {entry}')

    if sl is not None:
        box_top = max(entry, sl) if side=='short' else entry
        box_bottom = min(entry, sl) if side=='short' else sl
        if box_top == box_bottom:
            box_top = entry + 0.002 * entry
            box_bottom = entry - 0.002 * entry
        ax.add_patch(plt.Rectangle((0.01, box_bottom), 0.98, box_top - box_bottom,
                                   transform=ax.get_xaxis_transform(), alpha=0.15, color='red', label='Stop Loss Area'))

    for i, tp in enumerate(tps):
        ax.axhline(tp, linestyle='-', linewidth=1, alpha=0.9, label=f'TP{i+1} @ {tp}')
        ax.add_patch(plt.Rectangle((0.98, tp - 0.0005*tp), 0.02, 0.001*tp, transform=ax.get_xaxis_transform(),
                                   alpha=0.25, color='green'))

    ax.set_title(f"{symbol} — {side.upper()} — Entry:{entry} SL:{sl}")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle=':', linewidth=0.4)
    ax.legend(loc='upper left', fontsize='small')

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------- Monitoring / background task ----------
async def monitor_loop(app):
    await app.wait_until_ready()
    logger.info("Monitor loop started, polling every %s seconds", POLL_INTERVAL)
    trades = load_trades()
    while True:
        try:
            trades = load_trades()  # reload in case external changes
            open_trades = trades.get("open", [])
            if not open_trades:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            for t in list(open_trades):  # iterate over copy
                try:
                    ccxt_symbol = t.get("ccxt_symbol")
                    side = t.get("side")
                    chat_id = t.get("chat_id")
                    # fetch latest price (use ccxt)
                    ex_cls = getattr(ccxt, CCXT_EXCHANGE_ID)
                    ex = ex_cls({"enableRateLimit": True})
                    price = None
                    try:
                        ticker = ex.fetch_ticker(ccxt_symbol)
                        price = float(ticker.get("last") or ticker.get("close") or ticker.get("lastPrice"))
                    except Exception as e:
                        # fallback to OHLCV
                        try:
                            df = fetch_ohlcv_ccxt(ccxt_symbol, timeframe='1m', limit=2)
                            price = float(df['close'].iloc[-1])
                        except Exception as ee:
                            logger.info("Price fetch failed for %s: %s / %s", ccxt_symbol, e, ee)

                    if price is None:
                        continue

                    # Check SL / TPs
                    sl = float(t.get("sl"))
                    tps = [float(x) for x in t.get("tps", [])]
                    hit_tp_index = None
                    hit_sl = False

                    if side == "long":
                        # check if any TP reached (lowest index first)
                        for idx, tp in enumerate(tps):
                            if price >= tp and not t.get("tp_hit", {}).get(str(idx)):
                                hit_tp_index = idx
                                break
                        if price <= sl and not t.get("sl_hit"):
                            hit_sl = True
                    else:  # short
                        for idx, tp in enumerate(tps):
                            if price <= tp and not t.get("tp_hit", {}).get(str(idx)):
                                hit_tp_index = idx
                                break
                        if price >= sl and not t.get("sl_hit"):
                            hit_sl = True

                    if hit_tp_index is not None:
                        # mark tp hit
                        if "tp_hit" not in t:
                            t["tp_hit"] = {}
                        t["tp_hit"][str(hit_tp_index)] = {"price": price, "time": str(datetime.utcnow())}
                        # create message and screenshot
                        text = f"🎉 CONGRATULATIONS — TP{hit_tp_index+1} hit for {t.get('symbol')}!\nPrice: {price}\nTP{hit_tp_index+1}: {t.get('tps')[hit_tp_index]}\n"
                        text += f"Order entry: {t.get('entry')} SL: {t.get('sl')}\n"
                        # send message & screenshot
                        img_buf = plot_levels_image(t)
                        img_buf.seek(0)
                        await app.bot.send_message(chat_id=chat_id, text=text)
                        await app.bot.send_photo(chat_id=chat_id, photo=InputFile(img_buf, filename=f\"{t.get('symbol')}_tp{hit_tp_index+1}.png\"))
                        # keep the trade open but record TP hit; optionally close on first TP:
                        # if you want to close the trade after first TP, move it to closed list:
                        # t['closed'] = True
                        logger.info("TP%d hit for %s at price %s", hit_tp_index+1, t.get("symbol"), price)

                    if hit_sl:
                        t["sl_hit"] = {"price": price, "time": str(datetime.utcnow())}
                        text = f"❌ STOP LOSS hit for {t.get('symbol')}. Trade cancelled.\nPrice: {price}\nEntry: {t.get('entry')} SL: {t.get('sl')}\n"
                        img_buf = plot_levels_image(t)
                        img_buf.seek(0)
                        await app.bot.send_message(chat_id=chat_id, text=text)
                        await app.bot.send_photo(chat_id=chat_id, photo=InputFile(img_buf, filename=f\"{t.get('symbol')}_sl.png\"))
                        # mark as closed
                        t['closed'] = True
                        logger.info("SL hit for %s at price %s", t.get("symbol"), price)

                except Exception as e:
                    logger.error("Error processing trade: %s", e)
                    logger.error(traceback.format_exc())

            # remove closed trades, move to closed list
            trades = load_trades()
            open_trades = trades.get("open", [])
            new_open = []
            for t in open_trades:
                if t.get("closed"):
                    trades.setdefault("closed", []).append(t)
                else:
                    new_open.append(t)
            trades["open"] = new_open
            save_trades(trades)

        except Exception as e:
            logger.error("Monitor loop top-level error: %s", e)
            logger.error(traceback.format_exc())
        await asyncio.sleep(POLL_INTERVAL)

# ---------- Telegram handlers ----------
async def start_cmd(update, context):
    await update.message.reply_text(\"\"\"Salaam! Send /analyze SYMBOL to get trade setup and start monitoring.
Examples:\n/analyze BINANCE:BTCUSDT\n/analyze BTCUSDT\n\nThis bot calculates ATR-based SL and multiple TPs, then monitors price to inform you when TP/SL triggers.\"\"\")

async def help_cmd(update, context):
    await start_cmd(update, context)

async def analyze_cmd(update, context):
    args = context.args
    if not args:
        await update.message.reply_text(\"Usage: /analyze SYMBOL  — e.g. /analyze BINANCE:BTCUSDT\")
        return
    tv_symbol = args[0].strip()
    msg = await update.message.reply_text(f\"Analyzing {tv_symbol} ...\")
    loop = asyncio.get_event_loop()

    # Run analysis sync function in executor
    analysis = await loop.run_in_executor(None, analyze_symbol, tv_symbol, '1h')
    if analysis.get('error'):
        await msg.edit_text(f\"Error: {analysis['error']}\")
        return

    # Build response text
    text = f\"🔎 Analysis for {analysis['symbol']}\\n\"
    text += f\"Current Price: {analysis['price']}\\n\"
    text += f\"Bias: {analysis['side'].upper()}\\n\"
    text += f\"Entry: {analysis['entry']}\\nStop Loss: {analysis['sl']}\\n\"
    for i, tp in enumerate(analysis['tps']):
        text += f\"TP{i+1}: {tp}\\n\"
    if analysis.get('atr'):
        text += f\"ATR ({ATR_PERIOD}): {analysis.get('atr')}\\n\"
    text += \"\\nTo start monitoring this trade, reply with: /monitor_here\"

    # Save temporary analysis in context.chat_data so user can confirm monitoring
    context.chat_data['last_analysis'] = analysis
    await update.message.reply_text(text)
    buf = await loop.run_in_executor(None, plot_levels_image, analysis)
    buf.seek(0)
    await update.message.reply_photo(photo=InputFile(buf, filename=\"analysis.png\"))
    await msg.delete()

async def monitor_here_cmd(update, context):
    # Start monitoring the last_analysis for this chat and persist to trades.json
    analysis = context.chat_data.get('last_analysis')
    if not analysis:
        await update.message.reply_text(\"No recent analysis found. First use /analyze SYMBOL then /monitor_here to monitor.\")
        return
    trades = load_trades()
    trade_obj = analysis.copy()
    trade_obj['chat_id'] = update.effective_chat.id
    trade_obj['added_at'] = str(datetime.utcnow())
    trade_obj['tp_hit'] = {}
    trade_obj['sl_hit'] = None
    trade_obj['closed'] = False
    trades.setdefault('open', []).append(trade_obj)
    save_trades(trades)
    await update.message.reply_text(f\"Started monitoring {trade_obj['symbol']}. I will notify in this chat when TPs or SL hit.\")


async def list_trades_cmd(update, context):
    trades = load_trades()
    open_trades = trades.get('open', [])
    if not open_trades:
        await update.message.reply_text(\"No open monitored trades.\")
        return
    text = \"Open monitored trades:\\n\"
    for i, t in enumerate(open_trades):
        text += f\"{i+1}. {t.get('symbol')} | Side: {t.get('side')} | Entry: {t.get('entry')} | SL: {t.get('sl')}\\n\"
    await update.message.reply_text(text)

async def stoptrade_cmd(update, context):
    # stop a monitored trade by index or symbol
    args = context.args
    trades = load_trades()
    open_trades = trades.get('open', [])
    if not args:
        await update.message.reply_text(\"Usage: /stoptrade <index_or_symbol> (use /list_trades to see indices)\")
        return
    key = args[0]
    removed = False
    if key.isdigit():
        idx = int(key)-1
        if 0 <= idx < len(open_trades):
            t = open_trades.pop(idx)
            t['closed'] = True
            trades.setdefault('closed', []).append(t)
            removed = True
    else:
        for t in open_trades:
            if t.get('symbol').lower() == key.lower():
                t['closed'] = True
                trades.setdefault('closed', []).append(t)
                open_trades.remove(t)
                removed = True
                break
    trades['open'] = open_trades
    save_trades(trades)
    if removed:
        await update.message.reply_text(\"Trade stopped and removed from monitoring.\")
    else:
        await update.message.reply_text(\"No matching open trade found.\")


def run_bot():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start_cmd))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('analyze', analyze_cmd))
    app.add_handler(CommandHandler('monitor_here', monitor_here_cmd))
    app.add_handler(CommandHandler('list_trades', list_trades_cmd))
    app.add_handler(CommandHandler('stoptrade', stoptrade_cmd))

    # Start background monitor
    async def start_background(_):
        app.create_task(monitor_loop(app))

    app.post_init = start_background

    logger.info(\"Bot running...\")
    app.run_polling()

if __name__ == '__main__':
    run_bot()
