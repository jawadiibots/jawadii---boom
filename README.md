# TradingView Analyzer & Telegram Monitor Bot

This project is a Telegram bot that analyzes a TradingView-style symbol, computes:
- Entry price (current market)
- Stop-loss (ATR-based)
- Up to 5 Take-Profit levels (1R,1.5R,2R,3R,5R)

It also **monitors** open trades and notifies the chat when TP or SL levels are hit, sending text + a generated screenshot showing levels.

## How to deploy (Railway / Heroku / locally)

1. Create a Telegram bot and get BOT_TOKEN from BotFather.
2. Set environment variables:
   - `BOT_TOKEN` (required)
   - `CCXT_EXCHANGE` (optional, default: binance)
   - `POLL_INTERVAL` (optional, seconds to poll; default 60)
3. Push to GitHub and connect repository to Railway/Heroku. Railway will install `requirements.txt` and run the `Procfile` command `web: python app.py`.
4. In Telegram chat, use:
   - `/analyze BINANCE:BTCUSDT` to perform analysis
   - `/monitor_here` after analyze to start monitoring that symbol in this chat
   - `/list_trades` to view open monitored trades
   - `/stoptrade <index_or_symbol>` to remove a monitored trade

## Notes & caveats
- This bot uses `tradingview_ta` and `ccxt` for market data; symbol formats vary across exchanges. Test symbols first.
- This is **informational** only — not financial advice. Backtest before trading live.
- `ta-lib` is NOT required (ATR is computed with pandas).

## Files
- `app.py` - main bot
- `requirements.txt` - deps
- `Procfile` - for Railway/Heroku
- `trades.json` - created at runtime to persist monitored trades