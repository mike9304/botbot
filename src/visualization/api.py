"""FastAPI server for the trading simulation dashboard.

Provides:
- REST API for simulation control and data
- WebSocket for real-time updates
- Static file serving for the frontend
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.simulation import SimulationConfig, SimulationEngine

logger = logging.getLogger(__name__)

app = FastAPI(title="BotBot - Crypto Trading Agent Arena", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation state
simulation: Optional[SimulationEngine] = None
simulation_task: Optional[asyncio.Task] = None


@app.get("/api/status")
async def get_status():
    if simulation is None:
        return {"status": "idle", "message": "No simulation running"}
    return {
        "status": "running" if simulation.running else "stopped",
        "candle_index": simulation.current_candle_idx,
        "total_candles": simulation.config.n_candles,
        "groups": len(simulation.groups),
        "total_agents": sum(len(g.agents) for g in simulation.groups),
    }


@app.post("/api/simulation/start")
async def start_simulation(
    n_candles: int = 1000,
    speed: float = 10.0,
    symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
):
    global simulation, simulation_task

    if simulation and simulation.running:
        return {"error": "Simulation already running"}

    symbol_list = [s.strip() for s in symbols.split(",")]
    config = SimulationConfig(
        symbols=symbol_list,
        n_candles=n_candles,
        speed_multiplier=speed,
    )
    simulation = SimulationEngine(config)
    simulation_task = asyncio.create_task(simulation.run())

    return {"status": "started", "config": {
        "symbols": symbol_list,
        "n_candles": n_candles,
        "speed": speed,
    }}


@app.post("/api/simulation/stop")
async def stop_simulation():
    global simulation
    if simulation:
        simulation.stop()
        return {"status": "stopped"}
    return {"error": "No simulation running"}


@app.get("/api/leaderboard")
async def get_leaderboard():
    if not simulation:
        return {"error": "No simulation running"}

    all_agents = []
    for group in simulation.groups:
        for agent in group.agents:
            summary = agent.get_summary()
            summary["fitness"] = round(agent.fitness, 2)
            all_agents.append(summary)

    all_agents.sort(key=lambda x: x.get("total_pnl", 0), reverse=True)
    return {"leaderboard": all_agents}


@app.get("/api/groups")
async def get_groups():
    if not simulation:
        return {"error": "No simulation running"}

    groups = []
    for group in simulation.groups:
        groups.append({
            "name": group.name,
            "category": group.category,
            "description": group.description,
            "total_agents": len(group.agents),
            "total_pnl": round(group.total_pnl, 2),
            "avg_win_rate": round(group.avg_win_rate * 100, 1),
            "rankings": group.get_rankings(),
        })
    return {"groups": groups}


@app.get("/api/groups/{group_index}")
async def get_group_detail(group_index: int):
    if not simulation:
        return {"error": "No simulation running"}
    if group_index >= len(simulation.groups):
        return {"error": "Group not found"}

    group = simulation.groups[group_index]
    return {
        "name": group.name,
        "category": group.category,
        "description": group.description,
        "rankings": group.get_rankings(),
    }


@app.get("/api/evolution")
async def get_evolution_history():
    if not simulation:
        return {"error": "No simulation running"}
    return {
        "generation": simulation.evolver.generation,
        "history": [r.to_dict() for r in simulation.evolver.history],
    }


@app.get("/api/prices")
async def get_current_prices():
    if not simulation:
        return {"error": "No simulation running"}
    return {"prices": {s: round(p, 2) for s, p in simulation.exchange.current_prices.items()}}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation updates."""
    await websocket.accept()

    if not simulation:
        await websocket.send_json({"error": "No simulation running"})
        await websocket.close()
        return

    queue = simulation.subscribe()

    try:
        while True:
            # Send updates from queue
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_text(message)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

            # Check for incoming messages (e.g., commands)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                cmd = json.loads(data)
                if cmd.get("type") == "stop":
                    simulation.stop()
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        simulation.unsubscribe(queue)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        simulation.unsubscribe(queue)


# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend" / "public"


@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>BotBot Trading Arena</h1><p>Frontend not built yet.</p>")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
