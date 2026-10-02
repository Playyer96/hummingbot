"""
Backtest the supertrend_v1 directional controller over a grid of settings and print a ranked table.

Runs on real Binance perpetual candles through the built-in BacktestingEngineBase, so the entry
logic is exactly what the live controller uses. Each row is one settings combination.

Usage (inside the hummingbot container):
    docker exec -it hummingbot python /home/hummingbot/scripts/backtest_supertrend_sweep.py
    docker exec -it hummingbot python /home/hummingbot/scripts/backtest_supertrend_sweep.py --days 60
    docker exec -it hummingbot python /home/hummingbot/scripts/backtest_supertrend_sweep.py --pairs BTC-USDT
"""
import argparse
import asyncio
import logging
import os
import sys
import time
import traceback

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch broken optional dependency (injective proto mismatch)
try:
    from pyinjective.proto.injective.stream.v2 import query_pb2
    if not hasattr(query_pb2, "OrderFailuresFilter"):
        query_pb2.OrderFailuresFilter = type("OrderFailuresFilter", (), {})
except ImportError:
    pass

from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory  # noqa: E402
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HistoricalCandlesConfig  # noqa: E402
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase  # noqa: E402

from controllers.directional_trading.supertrend_v1 import SuperTrend  # noqa: E402

# pandas_ta's supertrend leaves SUPERTl_* (long band) empty in downtrends and SUPERTs_* (short band) empty in
# uptrends. The backtesting engine drops every row containing a NaN, which wipes out all rows. The signal only
# needs SUPERT_* and SUPERTd_*, so drop the one-sided bands before the engine sees them.
_original_update_processed_data = SuperTrend.update_processed_data


async def _update_processed_data_without_bands(self):
    await _original_update_processed_data(self)
    df = self.processed_data["features"]
    self.processed_data["features"] = df.drop(
        columns=[c for c in df.columns if c.startswith(("SUPERTl_", "SUPERTs_"))])


SuperTrend.update_processed_data = _update_processed_data_without_bands

logging.basicConfig(level=logging.WARNING, format="    %(name)s - %(levelname)s - %(message)s")

# (label, interval, stop_loss, take_profit, time_limit_seconds, trailing_stop, percentage_threshold)
SETTINGS = [
    ("live config", "15m", 0.0075, 0.015, 2 * 3600, "0.01,0.003", 0.005),
    ("15m wide", "15m", 0.015, 0.03, 4 * 3600, "", 0.005),
    ("15m wider", "15m", 0.025, 0.05, 8 * 3600, "", 0.01),
    ("1h tight", "1h", 0.01, 0.02, 8 * 3600, "", 0.01),
    ("1h wide", "1h", 0.02, 0.04, 24 * 3600, "", 0.01),
    ("1h wider", "1h", 0.03, 0.06, 48 * 3600, "", 0.015),
    ("1h wide + trail", "1h", 0.02, 0.06, 48 * 3600, "0.02,0.008", 0.01),
]


def build_config(trading_pair: str, amount: int, leverage: int, cooldown: int, setting):
    label, interval, stop_loss, take_profit, time_limit, trailing_stop, threshold = setting
    config_data = {
        "id": f"bt_supertrend_{trading_pair}_{label}".replace(" ", "_"),
        "controller_name": "supertrend_v1",
        "controller_type": "directional_trading",
        "connector_name": "binance_perpetual",
        "trading_pair": trading_pair,
        "candles_connector": "binance_perpetual",
        "candles_trading_pair": trading_pair,
        "total_amount_quote": amount,
        "leverage": leverage,
        "max_executors_per_side": 1,
        "cooldown_time": cooldown,
        "position_mode": "ONEWAY",
        "stop_loss": str(stop_loss),
        "take_profit": str(take_profit),
        "time_limit": time_limit,
        "trailing_stop": trailing_stop,
        "interval": interval,
        "length": 20,
        "multiplier": 4.0,
        "percentage_threshold": threshold,
    }
    return BacktestingEngineBase.get_controller_config_instance_from_dict(
        config_data, controllers_module="controllers"
    )


async def main(days: int, pairs, amount: int, leverage: int, cooldown: int, trade_cost: float):
    end_ts = int(time.time())
    start_ts = end_ts - days * 24 * 3600
    margin = amount / leverage
    engine = BacktestingEngineBase()
    rows = []

    # Sanity check: make sure candles download before running the whole grid
    for interval in ("1m", "15m"):
        feed = CandlesFactory.get_candle(CandlesConfig(connector="binance_perpetual", trading_pair=pairs[0],
                                                       interval=interval, max_records=100))
        df = await feed.get_historical_candles(HistoricalCandlesConfig(
            connector_name="binance_perpetual", trading_pair=pairs[0], interval=interval,
            start_time=end_ts - 6 * 3600, end_time=end_ts))
        print(f"candle check: {pairs[0]} {interval} -> {len(df)} candles in the last 6h", flush=True)

    total_runs = len(pairs) * len(SETTINGS)
    run = 0
    for pair in pairs:
        for setting in SETTINGS:
            run += 1
            label = setting[0]
            print(f"[{run}/{total_runs}] {pair} | {label} ...", flush=True)
            try:
                config = build_config(pair, amount, leverage, cooldown, setting)
                result = await engine.run_backtesting(config, start_ts, end_ts,
                                                      backtesting_resolution="1m", trade_cost=trade_cost)
                r = result["results"]
                rows.append({
                    "pair": pair,
                    "label": label,
                    "interval": setting[1],
                    "sl_tp": f"{setting[2] * 100:.2f}%/{setting[3] * 100:.1f}%",
                    "trades": int(r["total_executors_with_position"]),
                    "win_rate": float(r["accuracy"]),
                    "pnl": float(r["net_pnl_quote"]),
                    "max_dd": float(r["max_drawdown_pct"]),
                    "close_types": r["close_types"],
                })
            except Exception as e:
                print(f"    failed: {e}", flush=True)
                if run == 1:
                    traceback.print_exc()

    rows.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  SuperTrend backtest | last {days} days | ${amount} position at {leverage}x "
          f"(= ${margin:.2f} margin) | fees {trade_cost * 100:.3f}% per side")
    print(f"{'=' * 100}")
    print(f"  {'pair':<10}{'settings':<18}{'candles':<9}{'SL/TP':<14}{'trades':>7}{'win%':>8}"
          f"{'PnL $':>10}{'PnL % of margin':>17}{'worst dip $':>13}")
    for x in rows:
        print(f"  {x['pair']:<10}{x['label']:<18}{x['interval']:<9}{x['sl_tp']:<14}{x['trades']:>7}"
              f"{x['win_rate'] * 100:>7.0f}%{x['pnl']:>10.2f}{x['pnl'] / margin * 100:>16.0f}%"
              f"{x['max_dd'] * amount:>13.2f}")
    print("\n  How each setting's trades closed:")
    for x in rows:
        print(f"  {x['pair']:<10}{x['label']:<18}{x['close_types']}")
    print("\n  PnL % of margin below -100% means that setting would have wiped out the margin.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest supertrend_v1 over a grid of settings")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--pairs", type=str, default="BTC-USDT,ETH-USDT", help="Comma-separated trading pairs")
    parser.add_argument("--amount", type=int, default=100, help="Position size in USDT")
    parser.add_argument("--leverage", type=int, default=20, help="Leverage (only changes the margin column)")
    parser.add_argument("--cooldown-time", type=int, default=900, help="Cooldown after each trade in seconds")
    parser.add_argument("--trade-cost", type=float, default=0.0005, help="Fee per side (0.0005 = Binance taker 0.05%%)")
    args = parser.parse_args()

    asyncio.run(main(args.days, [p.strip() for p in args.pairs.split(",") if p.strip()],
                     args.amount, args.leverage, args.cooldown_time, args.trade_cost))
