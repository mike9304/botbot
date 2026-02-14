"""Main entry point for BotBot Trading Arena.

Usage:
    python main.py              # Run web dashboard (default)
    python main.py --headless   # Run simulation without web UI
    python main.py --candles 2000 --speed 50  # Custom settings
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("botbot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BotBot Crypto Trading Agent Arena")
    parser.add_argument("--headless", action="store_true", help="Run without web UI")
    parser.add_argument("--candles", type=int, default=1000, help="Number of candles to simulate")
    parser.add_argument("--speed", type=float, default=10.0, help="Simulation speed multiplier")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
        help="Comma-separated trading symbols",
    )
    return parser.parse_args()


async def run_headless(args: argparse.Namespace):
    """Run simulation in headless mode (no web UI)."""
    from src.core.simulation import SimulationConfig, SimulationEngine

    symbols = [s.strip() for s in args.symbols.split(",")]
    config = SimulationConfig(
        symbols=symbols,
        n_candles=args.candles,
        speed_multiplier=args.speed,
    )

    engine = SimulationEngine(config)
    logger.info("=" * 60)
    logger.info("  BOTBOT CRYPTO TRADING AGENT ARENA")
    logger.info("=" * 60)
    logger.info(f"  Symbols: {', '.join(symbols)}")
    logger.info(f"  Candles: {args.candles}")
    logger.info(f"  Speed:   {args.speed}x")
    logger.info("=" * 60)

    results = await engine.run()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL RESULTS")
    logger.info("=" * 60)

    logger.info("\n  TOP 20 GLOBAL LEADERBOARD:")
    logger.info("-" * 80)
    for i, agent in enumerate(results["global_leaderboard"][:20]):
        rank = i + 1
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank:2d}")
        pnl = agent["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        logger.info(
            f"  {medal} {agent['name']:30s} | {agent['group']:20s} | "
            f"PnL: {pnl_sign}${pnl:>10.2f} ({pnl_sign}{agent['pnl_percent']:.1f}%) | "
            f"WR: {agent['win_rate']}% | Trades: {agent['total_trades']:3d} | "
            f"DD: {agent['max_drawdown']}%"
        )

    logger.info("\n  GROUP RANKINGS:")
    logger.info("-" * 60)
    for i, group in enumerate(results["group_rankings"]):
        rank = i + 1
        pnl = group["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        logger.info(
            f"  #{rank} {group['name']:25s} | PnL: {pnl_sign}${pnl:>10.2f} | "
            f"WR: {group['avg_win_rate']}% | Agents: {group['total_agents']}"
        )

    logger.info(f"\n  Total Generations: {results['total_generations']}")
    logger.info("=" * 60)

    return results


def run_web(args: argparse.Namespace):
    """Run the web dashboard."""
    import uvicorn
    from src.visualization.api import app

    logger.info("=" * 60)
    logger.info("  BOTBOT CRYPTO TRADING AGENT ARENA")
    logger.info(f"  Dashboard: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main():
    args = parse_args()

    if args.headless:
        asyncio.run(run_headless(args))
    else:
        run_web(args)


if __name__ == "__main__":
    main()
