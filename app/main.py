import torch
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from agents.alphazero import AlphaZeroAgent
from state import Game
import os

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("Loading AlphaZero model... please wait.")
    bot = AlphaZeroAgent(model_path="models/weights/iter002.safetensors", train=False, random_select=False) 
    ml_models["bot"] = bot
    yield
    # --- SHUTDOWN ---
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get_ui():
    # Serving the HTML content directly for a single-file experience
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create a fresh game instance for this specific connection
    game = Game()
    bot = ml_models["bot"]
    
    # Helper to send state to UI
    async def send_update():
        # Quick forward pass to get "Priors" (Intuition) without MCTS
        if not game.over():
            with torch.no_grad():
                state_tensor = game.get_state_tensor()
                policy, value = bot.model.predict(state_tensor)
                priors = torch.softmax(policy, dim=0).numpy().tolist()
                win_prob = value.item() # -1 to 1
        else:
            priors = [0.0] * 7
            win_prob = 0.0

        await websocket.send_json({
            "type": "UPDATE",
            "board": game.board.tolist(),
            "priors": priors,
            "value": win_prob,
            "turn": game.turn,
            "winner": game.score() if game.over() else None
        })

    # Send initial board
    await send_update()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["action"] == "MOVE":
                col = int(msg["col"])
                if col in game.get_valid_moves():
                    game.make_move(col)
                    await send_update()

            elif msg["action"] == "STEP":
                if not game.over():
                    # Run MCTS in a separate thread so we don't block the WebSocket heartbeat
                    loop = asyncio.get_event_loop()
                    # You can dynamically adjust sims here if you want
                    bot.MCTS = int(msg.get("sims", 600)) 
                    
                    # Notify UI we are thinking
                    await websocket.send_json({"type": "THINKING"})
                    
                    best_move = await loop.run_in_executor(None, bot.get_best_move, game)
                    game.make_move(best_move)
                    await send_update()

            elif msg["action"] == "RESET":
                game = Game()
                await send_update()

    except WebSocketDisconnect:
        print("Client disconnected")