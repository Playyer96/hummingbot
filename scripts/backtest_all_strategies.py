"""
Backtest every single-exchange controller in this repo with its shipped default settings, plus a
funding-rate arbitrage calculation from Binance's real funding history, and print one ranked table.

- Directional controllers run on binance_perpetual with the futures taker fee (0.05% per side).
- Market making and grid controllers run twice: Binance spot fees (0.1%) and futures fees (0.05%).
- Total PnL includes unrealized PnL of inventory still held at the end of the test.

Controllers that need two exchanges, two pairs or an external signal feed (arbitrage_controller,
xemm_multiple_levels, stat_arb, hedge_asset, quantum_grid_allocator, ai_livestream) cannot run in
the single-pair backtesting engine and are listed as skipped.

Usage (inside the hummingbot container):
    docker exec -it hummingbot python /home/hummingbot/scripts/backtest_all_strategies.py
    docker exec -it hummingbot python /home/hummingbot/scripts/backtest_all_strategies.py --days 60 --pairs BTC-USDT
"""
import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import urllib.request

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch broken optional dependency (injective proto mismatch)
try:
    from pyinjective.proto.injective.stream.v2 import query_pb2
    if not hasattr(query_pb2, "OrderFailuresFilter"):
        query_pb2.OrderFailuresFilter = type("OrderFailuresFilter", (), {})
except ImportError:
    pass

from controllers.directional_trading.supertrend_v1 import SuperTrend  # noqa: E402
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory  # noqa: E402
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HistoricalCandlesConfig  # noqa: E402
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase  # noqa: E402

# pandas_ta's supertrend always leaves one of SUPERTl_*/SUPERTs_* NaN and the engine drops rows with NaN.
_original_supertrend_update = SuperTrend.update_processed_data


async def _supertrend_update_without_bands(self):
    await _original_supertrend_update(self)
    df = self.processed_data["features"]
    self.processed_data["features"] = df.drop(
        columns=[c for c in df.columns if c.startswith(("SUPERTl_", "SUPERTs_"))])


SuperTrend.update_processed_data = _supertrend_update_without_bands

SPOT_FEE = 0.001
FUTURES_FEE = 0.0005
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_all_errors.log")

DIRECTIONAL = ["bollinger_v1", "bollinger_v2", "bollingrid", "dman_v3", "macd_bb_v1", "supertrend_v1"]
MAKERS = [("pmm_simple", "market_making"), ("pmm_dynamic", "market_making"), ("dman_maker_v2", "market_making"),
          ("pmm_v1", "generic"), ("pmm_mister", "generic"), ("grid_strike", "generic")]
SKIPPED = ["arbitrage_controller (two exchanges)", "xemm_multiple_levels (two exchanges)",
           "stat_arb (two pairs)", "hedge_asset (hedging helper)", "quantum_grid_allocator (multi-pair)",
           "ai_livestream (external signal feed)", "multi_grid_strike (needs hand-placed grids)"]


def directional_config(name: str, pair: str, amount: int):
    return {
        "id": f"bt_{name}_{pair}",
        "controller_name": name,
        "controller_type": "directional_trading",
        "connector_name": "binance_perpetual",
        "trading_pair": pair,
        "candles_connector": "binance_perpetual",
        "candles_trading_pair": pair,
        "total_amount_quote": amount,
        "leverage": 1,
        "position_mode": "ONEWAY",
    }


def maker_config(name: str, controller_type: str, connector: str, pair: str, amount: int, price: float):
    config = {
        "id": f"bt_{name}_{connector}_{pair}",
        "controller_name": name,
        "controller_type": controller_type,
        "connector_name": connector,
        "trading_pair": pair,
        "total_amount_quote": amount,
        "leverage": 1,
        "position_mode": "ONEWAY",
    }
    if name == "pmm_dynamic":
        config.update({"candles_connector": connector, "candles_trading_pair": pair})
    if name == "pmm_v1":
        config["order_amount"] = str(round(amount / 2 / price, 6))
    if name == "pmm_mister":
        config["portfolio_allocation"] = "0.1"
    if name == "grid_strike":
        # BUY grid covering +/-3% around the starting price, stop 1% below the grid
        config.update({
            "side": "BUY",
            "start_price": str(round(price * 0.97, 6)),
            "end_price": str(round(price * 1.03, 6)),
            "limit_price": str(round(price * 0.97 * 0.99, 6)),
        })
    return config


async def start_price(connector: str, pair: str, start_ts: int) -> float:
    feed = CandlesFactory.get_candle(CandlesConfig(connector=connector, trading_pair=pair, interval="1m",
                                                   max_records=10))
    df = await feed.get_historical_candles(HistoricalCandlesConfig(
        connector_name=connector, trading_pair=pair, interval="1m",
        start_time=start_ts, end_time=start_ts + 3600))
    return float(df["close"].iloc[0])


def funding_arbitrage(pair: str, start_ts: int, end_ts: int, amount: int):
    """Short perp + long spot: collects funding when positive, pays when negative."""
    symbol = pair.replace("-", "")
    rates = []
    cursor = start_ts * 1000
    while cursor < end_ts * 1000:
        url = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}"
               f"&startTime={cursor}&endTime={end_ts * 1000}&limit=1000")
        with urllib.request.urlopen(url, timeout=30) as response:
            batch = json.loads(response.read().decode())
        if not batch:
            break
        rates.extend(float(x["fundingRate"]) for x in batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    # amount is split into a spot leg and a perp leg of equal size
    leg = amount / 2
    funding_income = sum(rates) * leg
    # open and close both legs as taker: spot 0.1% x2, perp 0.05% x2
    fees = leg * (2 * SPOT_FEE + 2 * FUTURES_FEE)
    return {"payments": len(rates), "negative": sum(1 for r in rates if r < 0),
            "income": funding_income, "fees": fees, "pnl": funding_income - fees}


async def main(days: int, pairs, amount: int):
    end_ts = int(time.time())
    start_ts = end_ts - days * 24 * 3600
    engine = BacktestingEngineBase()
    rows = []
    errors = []

    jobs = []
    for pair in pairs:
        for name in DIRECTIONAL:
            jobs.append((name, "futures", pair, "directional", FUTURES_FEE))
        for name, controller_type in MAKERS:
            jobs.append((name, "spot", pair, controller_type, SPOT_FEE))
            jobs.append((name, "futures", pair, controller_type, FUTURES_FEE))

    prices = {}
    for i, (name, market, pair, controller_type, fee) in enumerate(jobs, start=1):
        connector = "binance_perpetual" if market == "futures" else "binance"
        print(f"[{i}/{len(jobs)}] {name} | {market} | {pair} ...", flush=True)
        try:
            if controller_type == "directional":
                config_data = directional_config(name, pair, amount)
                controller_type = "directional_trading"
            else:
                if (connector, pair) not in prices:
                    prices[(connector, pair)] = await start_price(connector, pair, start_ts)
                config_data = maker_config(name, controller_type, connector, pair, amount,
                                           prices[(connector, pair)])
            config = BacktestingEngineBase.get_controller_config_instance_from_dict(
                config_data, controllers_module="controllers")
            result = await engine.run_backtesting(config, start_ts, end_ts,
                                                  backtesting_resolution="1m", trade_cost=fee)
            r = result["results"]
            unrealized = float(r.get("unrealized_pnl_quote", 0) or 0)
            rows.append({
                "name": name, "market": market, "pair": pair,
                "trades": int(r["total_executors_with_position"]),
                "fees": float(r.get("total_fees_quote", 0) or 0),
                "pnl": float(r["net_pnl_quote"]) + unrealized,
                "unrealized": unrealized,
            })
        except Exception as e:
            print(f"    failed: {str(e)[:150]}", flush=True)
            errors.append(f"===== {name} | {market} | {pair}\n{traceback.format_exc()}")

    funding_rows = []
    for pair in pairs:
        print(f"funding arbitrage | {pair} ...", flush=True)
        try:
            funding_rows.append((pair, funding_arbitrage(pair, start_ts, end_ts, amount)))
        except Exception as e:
            print(f"    failed: {str(e)[:150]}", flush=True)
            errors.append(f"===== funding arbitrage | {pair}\n{traceback.format_exc()}")

    rows.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n{'=' * 92}")
    print(f"  ALL STRATEGIES | last {days} days | ${amount} per strategy | default settings | "
          f"fees: spot {SPOT_FEE * 100:.2f}%, futures {FUTURES_FEE * 100:.2f}% per side")
    print(f"{'=' * 92}")
    print(f"  {'strategy':<16}{'market':<9}{'pair':<10}{'trades':>7}{'fees $':>9}{'unrealized $':>14}"
          f"{'TOTAL PnL $':>13}{'per month':>11}{'on your $14/mo':>16}")
    for x in rows:
        monthly = x["pnl"] / days * 30
        print(f"  {x['name']:<16}{x['market']:<9}{x['pair']:<10}{x['trades']:>7}{x['fees']:>9.2f}"
              f"{x['unrealized']:>14.2f}{x['pnl']:>13.2f}{monthly / amount * 100:>10.2f}%"
              f"{monthly / amount * 14:>15.2f}")

    print(f"\n  FUNDING-RATE ARBITRAGE (long spot + short perp, ${amount / 2:.0f} each leg, fees to open and close)")
    for pair, f in funding_rows:
        monthly = f["pnl"] / days * 30
        print(f"  {pair:<10} {f['payments']} payments ({f['negative']} negative) | funding ${f['income']:.2f} "
              f"- fees ${f['fees']:.2f} = ${f['pnl']:.2f} | {monthly / amount * 100:.2f}%/month | "
              f"on your $14: ${monthly / amount * 14:.2f}/month")

    print("\n  Not backtestable in this engine: " + ", ".join(SKIPPED))
    print("  'per month' is % of the strategy's capital. 'on your $14/mo' scales the result to your balance.")
    if errors:
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(errors))
        print(f"\n  {len(errors)} run(s) failed. Details saved to data/backtest_all_errors.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest every controller with default settings")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--pairs", type=str, default="BTC-USDT,ETH-USDT", help="Comma-separated trading pairs")
    parser.add_argument("--amount", type=int, default=100, help="Capital per strategy in USDT")
    args = parser.parse_args()

    asyncio.run(main(args.days, [p.strip() for p in args.pairs.split(",") if p.strip()], args.amount))
