import os
import json
import logging
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from ml_engine.pipeline import run_full_ml_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8000
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
SYNOPSIS_FILE = os.path.join(os.path.dirname(__file__), "SYNOPSIS.md")
README_FILE = os.path.join(os.path.dirname(__file__), "README.md")

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

# Simple in-memory cache for recent pipeline queries
CACHE = {}

class StockMLRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        if path == "/api/stocks":
            self.send_json_response(200, {"stocks": POPULAR_STOCKS})
            return

        elif path == "/api/synopsis":
            try:
                content = ""
                if os.path.exists(SYNOPSIS_FILE):
                    with open(SYNOPSIS_FILE, "r", encoding="utf-8") as f:
                        content = f.read()
                self.send_json_response(200, {"synopsis": content})
            except Exception as e:
                self.send_json_response(500, {"error": str(e)})
            return

        elif path == "/api/analyze":
            symbol = query.get("symbol", ["AAPL"])[0].upper().strip()
            period = query.get("period", ["2y"])[0]
            cache_key = f"{symbol}_{period}"

            if cache_key in CACHE:
                logger.info(f"Serving cached pipeline results for {cache_key}")
                self.send_json_response(200, CACHE[cache_key])
                return

            try:
                logger.info(f"Running ML pipeline for request: symbol={symbol}, period={period}")
                pipeline_result = run_full_ml_pipeline(symbol=symbol, period=period)
                CACHE[cache_key] = pipeline_result
                self.send_json_response(200, pipeline_result)
            except Exception as e:
                logger.exception(f"Error running pipeline for {symbol}")
                self.send_json_response(500, {"error": str(e), "message": f"Failed to analyze {symbol}"})
            return

        elif path == "/api/health":
            self.send_json_response(200, {"status": "ok", "service": "QuantAI Stock ML Platform"})
            return

        # Serve static frontend files
        super().do_GET()

    def send_json_response(self, status_code: int, data: dict):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, StockMLRequestHandler)
    print(f"[INFO] QuantAI Web App Server running at http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
