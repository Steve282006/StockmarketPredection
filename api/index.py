import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure root folder is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Top-level FastAPI application instance for Vercel Serverless Functions
app = FastAPI(
    title="QuantAI Stock ML Platform",
    description="Machine Learning Stock Analysis, Forecasting & Backtesting API",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_DIR = os.path.join(root_dir, "public")
SYNOPSIS_FILE = os.path.join(root_dir, "SYNOPSIS.md")

POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "sector": "Automotive & Energy"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "sector": "Index ETF"},
    {"symbol": "BTC-USD", "name": "Bitcoin USD", "sector": "Cryptocurrency"}
]

CACHE = {}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "QuantAI Stock ML Platform"}

@app.get("/api/stocks")
def list_stocks():
    return {"stocks": POPULAR_STOCKS}

@app.get("/api/synopsis")
def get_synopsis():
    try:
        content = ""
        if os.path.exists(SYNOPSIS_FILE):
            with open(SYNOPSIS_FILE, "r", encoding="utf-8") as f:
                content = f.read()
        return {"synopsis": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze")
def analyze_stock(symbol: str = "AAPL", period: str = "2y"):
    symbol = symbol.upper().strip()
    cache_key = f"{symbol}_{period}"

    if cache_key in CACHE:
        logger.info(f"Serving cached pipeline results for {cache_key}")
        return CACHE[cache_key]

    try:
        logger.info(f"Running ML pipeline for request: symbol={symbol}, period={period}")
        # Lazy import to ensure instant top-level app initialization
        from ml_engine.pipeline import run_full_ml_pipeline
        pipeline_result = run_full_ml_pipeline(symbol=symbol, period=period)
        CACHE[cache_key] = pipeline_result
        return pipeline_result
    except Exception as e:
        logger.exception(f"Error running pipeline for {symbol}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze {symbol}: {str(e)}")

# Mount static frontend web files if directory exists
if os.path.exists(PUBLIC_DIR):
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
