# EnterpriseMarketAgent
Enterprise Market Analyst Agent
Project Overview

The Enterprise Market Analyst Agent is an autonomous financial intelligence system designed to democratize institutional-grade market research. By orchestrating real-time market data, technical analysis, and sentiment processing, the agent delivers comprehensive investment insights and detailed reports within seconds.

This project showcases a complete multi-agent architecture built for the Enterprise Agents Track of the Kaggle AI Agents Intensive Capstone.
Core Features

Real-Time Market Data
Fetches live pricing, valuation metrics, market capitalization, revenue growth, and more using yfinance.

Technical Analysis Engine
Computes SMA indicators, RSI, MACD, and short/mid/long-term trend strength to determine bullish or bearish momentum.

News Sentiment Processing
Scrapes the latest financial news using DuckDuckGo (ddgs) to infer sentiment and correlate with market action.

Automated Investment Reports
Generates structured Markdown reports including fundamentals, technicals, sentiment, and Buy/Hold/Sell thesis.

Multi-Agent Architecture
Includes:

Groq LLM Agent (financial reasoning)

Tool Agent (quantitative data fetcher)

Extended Local Analysis Agent (fallback agent)

Orchestrator Agent (coordinates all workflows)

Session Memory
Stores conversation history and supports follow-up analysis based on previous queries.

Secure API Management
Environment-based key loading through python-dotenv.

Architecture Summary

Language Model
Groq-powered Llama 3 (configurable) used for high-level reasoning and synthesis.

Tools Layer
Custom Python tools:

get_stock_fundamentals

get_technical_analysis

get_market_news

Session and Memory
In-memory session service tracks full analysis flow and supports continuity.

Long-Running Operations
Extended local analysis (SMA20/50/200, RSI, MACD, quarterly revenues) triggered on demand.

Observability
Logging and runtime metrics included for debugging and evaluation.



Install Dependencies

Create a virtual environment (recommended) and install required packages
pip install -r requirements.txt
