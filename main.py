from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from zero import AlphaZero
from state import Game

# This dictionary acts as a persistent "Global Store" for the app
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP: Runs when the server starts ---
    print("Loading AlphaZero model... please wait.")
    game = Game()
    bot = AlphaZero(noise=0.3, model_pth="models/iter002.safetensors") # Load your heavy weights here
    
    ml_models["game"] = game
    ml_models["bot"] = bot
    yield
    # --- SHUTDOWN: Runs when you stop the server ---
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get_ui():
    return FileResponse("index.html")

@app.get("/move")
async def handle_move(col: int):
    game = ml_models["game"]
    bot = ml_models["bot"]

    # 1. Player Move
    game.make_move(col)
    if game.over():
        return {
            "board": game.board.tolist(),
            "bot_move": -1,
            "winner": game.score()
        }

    # 2. Bot Move (AlphaZero logic)
    bot_col = bot.get_best_move(game)
    game.make_move(bot_col)
    return {
        "board": game.board.tolist(), 
        "bot_move": bot_col,
        "winner": game.score() if game.over() else None
    }

@app.get(path="/reset")
async def reset_all():
    g = ml_models["game"]
    g.reset()
    return {
        "board": g.board.tolist()
    }