import os
import re
import json
import time
import math
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import pandas as pd
import numpy as np
from termcolor import colored

try:
    from ddgs import DDGS
except Exception:
    DDGS = None

try:
    from google import genai
except Exception:
    genai = None

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "agent.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log_info(msg: str):
    logging.info(msg)
    print(colored(msg, "cyan"))

def log_warn(msg: str):
    logging.warning(msg)
    print(colored(msg, "yellow"))

def log_error(msg: str):
    logging.error(msg)
    print(colored(msg, "red"))

METRICS = {"requests": 0, "model_calls": 0, "local_calls": 0}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_CANDIDATES = os.getenv("GEMINI_MODEL_CANDIDATES", "").strip()

client = None
chosen_model = None

def init_gemini_client_with_rotation():
    global client, chosen_model
    if genai is None:
        log_warn("genai SDK not available; Gemini integration disabled.")
        client = None
        chosen_model = None
        return None, None
    if not GOOGLE_API_KEY:
        log_warn("GOOGLE_API_KEY not found in environment; Gemini disabled.")
        client = None
        chosen_model = None
        return None, None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        log_warn(f"genai.configure failed: {e}")
    candidates = []
    if GEMINI_MODEL:
        candidates.append(GEMINI_MODEL)
    if GEMINI_CANDIDATES:
        candidates.extend([m.strip() for m in GEMINI_CANDIDATES.split(",") if m.strip()])
    if not candidates:
        candidates = ["gemini-1.5-pro", "gemini-1.0", "gemini-1.5-mini"]
    for model in candidates:
        try:
            tmp_client = genai.GenAI() if hasattr(genai, "GenAI") else genai
            prompt = [{"role":"user","content":"Ping"}]
            try:
                resp = tmp_client.generate(model=model, input="Ping")
            except Exception:
                resp = tmp_client.generate_text(model=model, text="Ping")
            chosen_model = model
            client = tmp_client
            os.environ["GEMINI_MODEL"] = model
            log_info(f"Gemini model initialized: {model}")
            return client, model
        except Exception as e:
            msg = str(e).lower()
            log_warn(f"Gemini model {model} failed: {e}")
            if "invalid" in msg or "unauthorized" in msg or "401" in msg:
                log_error("Authentication failure for Google API key.")
                return None, None
            time.sleep(0.1)
            continue
    log_warn("No Gemini model candidates succeeded; running local-only.")
    client = None
    chosen_model = None
    return None, None

init_gemini_client_with_rotation()

def safe_json_loads(s: Any) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s

def get_stock_fundamentals(ticker: str) -> str:
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info if hasattr(tk, "info") else {}
        except Exception:
            info = {}
        data = {
            "symbol": info.get("symbol", ticker),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "peg_ratio": info.get("pegRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "currency": info.get("currency")
        }
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": f"Error fetching fundamentals: {str(e)}"})

def get_technical_analysis(ticker: str, period: str = "1y") -> str:
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period)
        if hist is None or hist.empty:
            return json.dumps({"error": "no historical data"})
        close = hist["Close"].dropna()
        def sma(series, window):
            return series.rolling(window=window).mean()
        sma_5 = sma(close, 5).iloc[-1] if len(close) >= 5 else None
        sma_20 = sma(close, 20).iloc[-1] if len(close) >= 20 else None
        sma_50 = sma(close, 50).iloc[-1] if len(close) >= 50 else None
        sma_200 = sma(close, 200).iloc[-1] if len(close) >= 200 else None
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean().replace(0, np.nan)
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        rsi_latest = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd - signal).iloc[-1]) if not macd.empty else None
        latest_close = float(close.iloc[-1])
        trend_short = "BULLISH" if sma_5 and latest_close > sma_5 else "BEARISH"
        trend_medium = "BULLISH" if sma_50 and latest_close > sma_50 else "BEARISH"
        trend_long = "BULLISH" if sma_200 and latest_close > sma_200 else "BEARISH"
        res = {
            "ticker": ticker,
            "last_close": latest_close,
            "sma_5": sma_5,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi_14": rsi_latest,
            "macd_hist": macd_hist,
            "trend_short": trend_short,
            "trend_medium": trend_medium,
            "trend_long": trend_long
        }
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": f"Error in technical analysis: {str(e)}"})

def get_market_news(query: str, max_results: int = 3) -> str:
    try:
        if DDGS is None:
            return json.dumps([{"error": "ddgs package unavailable"}])
        with DDGS() as ddgs:
            results = ddgs.text(f"financial news {query}", max_results=max_results)
            summary = []
            for r in results:
                title = r.get("title") or r.get("href")
                body = r.get("body") or r.get("snippet") or ""
                summary.append({"title": title, "snippet": body})
            return json.dumps(summary)
    except Exception as e:
        return json.dumps([{"error": f"Error searching news: {str(e)}"}])

available_tools = {
    "get_stock_fundamentals": get_stock_fundamentals,
    "get_technical_analysis": get_technical_analysis,
    "get_market_news": get_market_news
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_fundamentals",
            "description": "Get current price, market cap, and analyst recommendations.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_analysis",
            "description": "Get technical trend indicators.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_news",
            "description": "Search for news to analyze sentiment.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]

class InMemorySession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow()
    def add_message(self, role: str, content: Any):
        self.history.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})

class SessionService:
    def __init__(self):
        self.sessions: Dict[str, InMemorySession] = {}
    def new_session(self, session_id: Optional[str] = None) -> InMemorySession:
        sid = session_id or f"session_{int(time.time())}"
        s = InMemorySession(sid)
        self.sessions[sid] = s
        return s
    def get_session(self, session_id: str) -> Optional[InMemorySession]:
        return self.sessions.get(session_id)

session_service = SessionService()

def extract_tickers(text: str) -> List[str]:
    tickers = re.findall(r"\(([A-Z]{1,5})\)", text)
    if not tickers:
        candidates = re.findall(r"\b([A-Z]{2,5})\b", text)
        tickers = list(dict.fromkeys(candidates))
    return tickers

def local_extended_analysis(tickers: List[str]) -> str:
    METRICS["local_calls"] += 1
    results = []
    for t in tickers:
        try:
            tech = json.loads(get_technical_analysis(t))
        except Exception:
            tech = {"error": "technical failed"}
        try:
            fund = json.loads(get_stock_fundamentals(t))
        except Exception:
            fund = {"error": "fundamentals failed"}
        q_revs = None
        try:
            tk = yf.Ticker(t)
            q_fin = tk.quarterly_financials if hasattr(tk, "quarterly_financials") else None
            if q_fin is not None and not q_fin.empty:
                rev_rows = [r for r in q_fin.index if "reven" in r.lower()]
                if rev_rows:
                    q_revs = q_fin.loc[rev_rows[0]].dropna().to_dict()
        except Exception:
            q_revs = None
        results.append({"ticker": t, "fundamentals": fund, "technical": tech, "quarterly_revenues": q_revs})
    lines = ["# Extended Local Analysis\n"]
    lines.append("| Ticker | Revenue Growth | Price | Trend (short/med/long) |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        t = r.get("ticker")
        f = r.get("fundamentals") or {}
        tech = r.get("technical") or {}
        rg = f.get("revenue_growth") if isinstance(f, dict) else "-"
        price = f.get("current_price") if isinstance(f, dict) else "-"
        ts = tech.get("trend_short", "-")
        tm = tech.get("trend_medium", "-")
        tl = tech.get("trend_long", "-")
        lines.append(f"| {t} | {rg} | {price} | {ts}/{tm}/{tl} |")
    lines.append("\n## Notes\n- Local analysis is best-effort based on yfinance.\n")
    return "\n".join(lines)

class GeminiAgent:
    def __init__(self, client_obj, model_name: str):
        self.client = client_obj
        self.model = model_name
    def generate(self, prompt: str) -> Optional[str]:
        METRICS["model_calls"] += 1
        if not self.client:
            return None
        try:
            if hasattr(self.client, "generate"):
                resp = self.client.generate(model=self.model, input=prompt)
                text = None
                try:
                    if isinstance(resp, dict):
                        text = resp.get("candidates",[{}])[0].get("content")
                    else:
                        text = getattr(resp, "text", None) or str(resp)
                except Exception:
                    text = str(resp)
                return text
            if hasattr(self.client, "generate_text"):
                resp = self.client.generate_text(model=self.model, text=prompt)
                return getattr(resp, "text", None) or str(resp)
            return None
        except Exception as e:
            log_warn(f"Gemini generate failed: {e}")
            return None

gemini_agent = None
if client and chosen_model:
    gemini_agent = GeminiAgent(client, chosen_model)

class OrchestratorAgent:
    def __init__(self, model_agent: Optional[GeminiAgent] = None):
        self.model_agent = model_agent
    def run(self, user_input: str, session: InMemorySession) -> str:
        METRICS["requests"] += 1
        session.add_message("user", user_input)
        tickers = extract_tickers(user_input)
        if self.model_agent and client:
            try:
                log_info("Attempting Gemini LLM analysis...")
                llm_prompt = f"You are a senior market analyst. User input: {user_input}"
                llm_out = self.model_agent.generate(llm_prompt)
                if llm_out:
                    session.add_message("assistant", llm_out)
                    return llm_out
            except Exception as e:
                log_warn(f"Gemini LLM error: {e}")
        log_info("Running local fallback analysis...")
        if tickers:
            lines = ["# Local Quick Comparative Analysis\n"]
            lines.append("| Ticker | Revenue Growth | Current Price | Technical Trend |")
            lines.append("|---|---:|---:|---:|")
            for t in tickers:
                fund = json.loads(get_stock_fundamentals(t))
                tech = json.loads(get_technical_analysis(t))
                rg = fund.get("revenue_growth") if isinstance(fund, dict) else "-"
                price = fund.get("current_price") if isinstance(fund, dict) else "-"
                trend = tech.get("trend_short") if isinstance(tech, dict) else "-"
                lines.append(f"| {t} | {rg} | {price} | {trend} |")
            lines.append("\nWould you like a deeper local extended analysis (SMA20/50/200, RSI, MACD, quarterly revenues)? Reply 'yes' to run it.")
            res = "\n".join(lines)
            session.add_message("assistant", res)
            return res
        else:
            help_text = "I couldn't detect ticker symbols in your request. Please include symbols like 'MSFT' or 'AAPL' or ask a general question."
            session.add_message("assistant", help_text)
            return help_text

orchestrator = OrchestratorAgent(gemini_agent)

class Evaluator:
    def __init__(self):
        self.records = []
    def record(self, session: InMemorySession, verdict: str, notes: str = ""):
        self.records.append({"session": session.session_id, "verdict": verdict, "notes": notes, "ts": datetime.utcnow().isoformat()})
    def summary(self):
        return {"count": len(self.records), "records": self.records}

evaluator = Evaluator()

def seed_example_session():
    s = session_service.new_session("demo_session")
    s.add_message("system", "Demo session started")
    return s

def print_welcome():
    print(colored("\n==================================================", "green"))
    print(colored("🚀 CAPSTONE: MULTI-AGENT ENTERPRISE MARKET ANALYST", "white", attrs=["bold"]))
    print(colored("  Tracks: Enterprise Agents — Multi-agent demonstration", "white"))
    print(colored("==================================================\n", "green"))
    print("Example: 'Compare Microsoft (MSFT) and Google (GOOGL). Which one has better revenue growth?'")

def run_cli():
    print_welcome()
    session = session_service.new_session()
    if client and chosen_model:
        log_info(f"Gemini initialized: {chosen_model}")
    else:
        log_warn("Gemini not initialized; operating in local-only mode.")
    while True:
        try:
            user_input = input(colored("\n👤 Analyst Request (or 'exit'): ", "blue"))
            if user_input.lower() in ("exit", "quit"):
                print("Shutting down secure session...")
                break
            if user_input.strip().lower() == "yes":
                last_user = None
                for m in reversed(session.history):
                    if m["role"] == "user":
                        last_user = m
                        break
                if last_user:
                    last_text = last_user["content"] if isinstance(last_user["content"], str) else json.dumps(last_user["content"])
                    tickers = extract_tickers(last_text)
                    if tickers:
                        print(colored("\n📈 Running extended local analysis...", "magenta"))
                        report = local_extended_analysis(tickers)
                        session.add_message("assistant", report)
                        print(colored("\n📝 EXTENDED REPORT:\n", "green", attrs=["bold"]))
                        print(report)
                    else:
                        print(colored("No tickers found in the previous query to run extended analysis.", "yellow"))
                else:
                    print(colored("No previous user request found to base extended analysis on.", "yellow"))
                continue
            print(colored("\n📊 Analyzing Market Data...", "magenta"))
            output = orchestrator.run(user_input, session)
            print(colored("\n📝 AGENT RESPONSE:\n", "green", attrs=["bold"]))
            print(output)
            if "Would you like a deeper local extended analysis" in output:
                print(colored("\nTip: Reply 'yes' to run the extended local analysis (SMA20/50/200, RSI, MACD, quarterly revenues).", "yellow"))
        except KeyboardInterrupt:
            break
        except Exception as e:
            log_error(f"System Error: {e}")
            try:
                tks = extract_tickers(user_input)
                print(local_extended_analysis(tks))
            except Exception:
                pass

if __name__ == "__main__":
    run_cli()
