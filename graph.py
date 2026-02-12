


from typing import TypedDict,Dict,Optional,List
from langgraph.graph import StateGraph, START,END
import yfinance as yf
from langchain_groq import ChatGroq
from pprint import pprint
import numpy as np
import talib



class DataState(TypedDict):
    user_portfolio : List[Dict]
    user_needs : str
    intent : str
    stable_performers : Optional[List[Dict]]
    potential_reducers : Optional[List[Dict]]
    management_plan : Optional[str]
    allocation_plan : Optional[str]
    eval_output : Optional[str]
    output : str
    
    




  
def fetch_price_history(symbol, duration="1y"):
    history_data = yf.download(symbol, period=duration, progress=False)
    return history_data["Close"].squeeze()


def compute_signals(price_series,symbol):
    # returns_data = price_series.pct_change().dropna()
    
    signals = {}
    
    
    signals["1m_return"] = (price_series.iloc[-1] / price_series.iloc[-21]) - 1
    signals["3m_return"] = (price_series.iloc[-1] / price_series.iloc[-63]) - 1
    signals["6m_return"] = (price_series.iloc[-1] / price_series.iloc[-126]) - 1
    # signals["9m_return"] = (price_series.iloc[-1] / price_series.iloc[-189]) - 1
    signals["1y_return"] = (price_series.iloc[-1] / price_series.iloc[0]) - 1

    signals["1m_formatted"] = format_return(signals["1m_return"])
    signals["3m_formatted"] = format_return(signals["3m_return"])
    signals["6m_formatted"] = format_return(signals["6m_return"])
    # signals["9m_formatted"] = format_return(signals["9m_return"])
    signals["1y_formatted"] = format_return(signals["1y_return"])
    daily_returns = price_series.pct_change().dropna()
    signals["volatility"] = daily_returns.std() * (252 ** 0.5)
    if signals["3m_return"] > 0 and signals["6m_return"] > 0:
        signals["trend"] = "up"
    elif signals["3m_return"] < 0 and signals["6m_return"] < 0:
        signals["trend"] = "down"
    else:
        signals["trend"] = "sideways"

    signals["1m_return"] = float(signals["1m_return"])
    signals["3m_return"] = float(signals["3m_return"])
    signals["6m_return"] = float(signals["6m_return"])
    signals["1y_return"] = float(signals["1y_return"])
    signals["volatility"] = float(signals["volatility"])
    signals["TA_Signals"] = get_TA_signals(symbol)
    signals["fundamentals"] = get_fundamentals(symbol)
    del signals["1m_return"]
    del signals["3m_return"]
    del signals["6m_return"]
    del signals["1y_return"]

    return signals
    



def fundamental_risk_score(
    pe: float,
    debt_to_equity: float,
    beta: float
) -> float:
    #Stable defaults
    pe = 18 if pe is None or pe <= 0 else pe
    debt_to_equity = (
        debt_to_equity
        if debt_to_equity is not None and debt_to_equity >= 0
        else 0.5
    )

    beta = (
        beta
        if beta is not None and beta > 0
        else 0.8
    )

    #PE Scoremax 4
    if pe <= 15:
        pe_score = 4.0
    elif pe <= 25:
        pe_score = 3.0
    elif pe <= 40:
        pe_score = 2.0
    else:
        pe_score = 1.0

    #EquityScore max 3.5
    if debt_to_equity <= 0.5:
        debt_score = 3.5
    elif debt_to_equity <= 1:
        debt_score = 2.5
    elif debt_to_equity <= 2:
        debt_score = 1.5
    else:
        debt_score = 0.5

    #Beta Score max2.5
    if beta <= 0.8:
        beta_score = 2.5
    elif beta <= 1:
        beta_score = 2.0
    elif beta <= 1.3:
        beta_score = 1.5
    else:
        beta_score = 0.5

    total_score = pe_score + debt_score + beta_score
    return round(total_score, 2)



def get_fundamentals(symbol):
    if symbol.startswith("^"):
        return None  

    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return None

    d = {
        "pe_ratio": info.get("trailingPE"),
        # "pb_ratio": info.get("priceToBook"),
        # "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "beta": info.get("beta"),
        # "sector": info.get("sector")
    }
    return fundamental_risk_score(pe=d["pe_ratio"],debt_to_equity=d["debt_to_equity"],beta=d["beta"])


   
def format_return(return_value):
        """Convert Pandas Series/scalar to +15.32% format"""
        if hasattr(return_value, 'iloc'):
            value = return_value.iloc[0] 
        else:
            value = return_value
        
        return f"{value*100}"



def technical_stability_score(
    adx: float,
    rsi: float,
    atr_pct: float,   
    obv_trend: float 
) -> float:
    #Stable defaults
    adx = adx if adx > 0 else 20
    rsi = rsi if 0 < rsi < 100 else 50
    atr_pct = atr_pct if atr_pct > 0 else 0.02
    obv_trend = obv_trend if abs(obv_trend) < 1 else 0

    #ADX Scormax 2.5
    if adx < 20:
        adx_score = 2.5  # stable 
    elif adx < 30:
        adx_score = 2.0
    elif adx < 40:
        adx_score = 1.2
    else:
        adx_score = 0.5 # strong trend

    #RSI Scoremax 2.5
    if 45 <= rsi <= 55:
        rsi_score = 2.5
    elif 40 <= rsi <= 60:
        rsi_score = 2.0
    elif 30 <= rsi <= 70:
        rsi_score = 1.2
    else:
        rsi_score = 0.5                 #overbough

    #ATR Scoremax 3.0
    if atr_pct <= 0.015:
        atr_score = 3.0
    elif atr_pct <= 0.025:
        atr_score = 2.2
    elif atr_pct <= 0.04:
        atr_score = 1.2
    else:
        atr_score = 0.6 # high volatility

    #OBV Scoremax 2.0
    if abs(obv_trend) < 0.01:
        obv_score = 2.0     # stable
    elif abs(obv_trend) < 0.03:
        obv_score = 1.5
    elif abs(obv_trend) < 0.06:
        obv_score = 1.0
    else:
        obv_score = 0.5#aggressive flows

    total_score = adx_score + rsi_score + atr_score + obv_score
    return round(total_score, 2)


def get_TA_signals(symbol: str, period: str = "1y") -> dict:

    df = yf.download(symbol, period=period, progress=False)

    if df.empty:
        return {"error": "No data fetched"}

    df = df.dropna()

    if len(df) < 100:
        return {"error": "Insufficient data"}

    high = np.asarray(df["High"], dtype=np.float64).reshape(-1)
    low = np.asarray(df["Low"], dtype=np.float64).reshape(-1)
    close = np.asarray(df["Close"], dtype=np.float64).reshape(-1)
    volume = np.asarray(df["Volume"], dtype=np.float64).reshape(-1)

    adx = talib.ADX(high, low, close, 14)
    atr = talib.ATR(high, low, close, 14)
 #   macd, macd_signal, macd_hist = talib.MACD(close)
    obv = talib.OBV(close, volume)
    rsi = talib.RSI(close, 14)

    # close_series = df["Close"]

    # high_52w = float(close_series.rolling(252, min_periods=100).max().iloc[-1])
    # low_52w = float(close_series.rolling(252, min_periods=100).min().iloc[-1])
    # current_price = float(close_series.iloc[-1])

    # pct_from_high = (
    #     (current_price / high_52w) * 100 if high_52w > 0 else None
    # )
    # pct_from_low = (
    #     (current_price / low_52w) * 100 if low_52w > 0 else None
    # )

    d = {
        "symbol": symbol,
        "adx": float(adx[-1]),
        "rsi": float(rsi[-1]),
        # "macd_hist": float(macd_hist[-1]),
        "atr": float(atr[-1]),
        "obv": float(obv[-1]),
        # "current_price": current_price,
        # "52w_high": high_52w,
        # "52w_low": low_52w,
        # "pct_from_52w_high": round(pct_from_high, 2) if pct_from_high else None,
        # "pct_from_52w_low": round(pct_from_low, 2) if pct_from_low else None,
        # "data_points": len(df)
    }
    
    return technical_stability_score(adx=d["adx"],rsi=d["rsi"],atr_pct=d["atr"],obv_trend=d["obv"])


   
   
   
   

def classify_stock(metrics: Dict) -> Dict:
    score = 0

    if float(metrics["1m_formatted"]) > 0:
        score += 1
    else:
        score -= 1
        
    if float(metrics["3m_formatted"]) > 0:
        score += 1
    else:
        score -= 1

    if float(metrics["6m_formatted"]) > 0:
        score += 1
        

    if float(metrics["1y_formatted"]) < 0:
        score -= 1

    if metrics["TA_Signals"] >= 6.5:
        score += 1
        
    elif metrics["TA_Signals"] < 4:
        score -= 1
        

    if metrics["fundamentals"] >= 6.5:
        score += 1
         
    elif metrics["fundamentals"] < 4:
        score -= 1
        

    if metrics["trend"] == "up":
        score += 1
         
    elif metrics["trend"] == "down":
        score -= 1
         

    v = metrics["volatility"]

    if v > 0.6:
        score -= 2
         
    elif v > 0.3:
        score -= 1
         

    return score




    
    
    
def common_metrics_node(current_state: DataState):
    
    portfolio_with_metrics = [asset for asset in current_state["user_portfolio"]]
    
    #fetch current price of the asset
    def fetch_current_price(asset_symbol):
        asset_data = yf.Ticker(asset_symbol)
        asset_data = asset_data.info
        return asset_data.get('currentPrice')
        

    #basic calculation of the asset 
    total_value_of_portfolio = 0
    for asset in portfolio_with_metrics:
        asset['current_price'] = fetch_current_price(asset["symbol"])
        price = asset.get("current_price")
        if price is None:
            asset["value"] = asset['buy_price'] * asset["quantity"]
        else:
            asset["value"] = price * asset["quantity"]
        total_value_of_portfolio += asset['value']

    
    #portfolio weightage calcluation
    for asset in portfolio_with_metrics:
        asset['weightage'] = round((asset['value'] / total_value_of_portfolio) * 100,2)

    return {
        "user_portfolio": portfolio_with_metrics,
        "total_value_of_portfolio": total_value_of_portfolio
    }





























    
    
def common_signals_node(state : DataState) -> DataState:
    
    stable_performers = []
    potential_reducers = []
    portfolio_with_signals = [asset for asset in state["user_portfolio"]]
    for asset in portfolio_with_signals:
        asset_price_history = fetch_price_history(asset["symbol"])
    # print(asset_price_history)
        signals = compute_signals(asset_price_history,asset["symbol"])
        asset["metrics"] = signals
       # pprint(asset['metrics'])
        score  = classify_stock(asset["metrics"])
        if score >= 2:
            stable_performers.append({asset['symbol'] : asset['weightage']})
        else:
            potential_reducers.append({asset['symbol'] : asset['weightage']})
    return {
        **state,
        'user_portfolio' : portfolio_with_signals,
        'stable_performers' : stable_performers,
        'potential_reducers' : potential_reducers
    }




















def llm_portfolio_evaluator_prompt(stable_performers, potential_reducers):
    prompt = f"""
SYSTEM:
You are a professional portfolio evaluation and allocation-quality assessor.
Your role is to evaluate portfolio structure, allocation balance, and risk discipline.
You do NOT predict prices or returns.
You do NOT give buy/sell recommendations.
You do NOT use external market data.

USER:
You are given the outcome of a prior quantitative screening process.
The portfolio assets are already grouped based on their stability and risk contribution.

1. Stable Performers:
These stocks contribute positively to portfolio stability due to
controlled volatility, acceptable trends, and balanced signals.

{stable_performers}

2. Potential Risk Contributors:
These stocks introduce higher volatility, instability, or structural risk
to the portfolio.

{potential_reducers}

Your task is to **evaluate the overall portfolio quality** from an
allocation and risk-management perspective.

========================
EVALUATION OBJECTIVES
========================

You must assess the portfolio strictly on the following dimensions:
- Allocation balance
- Risk concentration
- Stability under uncertain market conditions
- Dependence on high-risk assets
- Overall portfolio discipline

========================
REQUIRED ANALYSIS
========================

You MUST address ALL sections clearly and objectively:

1. Portfolio Structure Evaluation
- Evaluate how well-balanced the current portfolio is
- Identify whether stability or risk dominates the structure
- Explain in simple, non-technical language

2. Key Strengths of the Portfolio
- List up to 2 structural strengths
- Focus on diversification, stability, or disciplined exposure

3. Key Weaknesses of the Portfolio
- List up to 2 structural weaknesses
- Focus on risk concentration, instability, or over-reliance on risky stocks

4. Allocation Quality Assessment
- Evaluate whether the portfolio is:
  - Well-allocated
  - Moderately imbalanced
  - Poorly allocated
- Justify the assessment briefly

5. Conservative Improvement Guidance
- Provide high-level, non-actionable guidance to improve portfolio quality
- You may suggest:
  - Rebalancing emphasis toward stable performers
  - Limiting exposure to risk contributors
  - Improving diversification discipline
- Do NOT give buy/sell or timing instructions

6. Portfolio Quality Score
- Assign an overall portfolio quality score (0-100%)
- Base the score strictly on:
  - Stability
  - Risk balance
  - Structural discipline
- Justify the score briefly

========================
STRICT CONSTRAINTS
========================
- Do NOT calculate or assume returns
- Do NOT forecast performance
- Do NOT introduce new assets or data
- Base reasoning ONLY on the provided stock groups
- Keep reasoning conservative and professional

========================
REQUIRED OUTPUT FORMAT
========================

1. Portfolio Structure Evaluation

2. Key Strengths
   - Strength 1:
   - Strength 2:
   
   Key Weaknesses
   - Weakness 1:
   - Weakness 2:

3. Trim or Exclude or Reduce stocks:
    -STOCK A(Trim|Exclude|Reduce) - <reason>
    
4. Allocation Quality Assessment
   - Assessment:
   - Brief Reasoning:

5. Portfolio Quality Score
   - Score:
   - Justification:
"""
    return prompt








def llm_portfolio_eval_agent(state : DataState) -> DataState:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        #add api key
        temperature=0.3
    )
    prompt = ""
    if 'eval_output' not in state:
        prompt = llm_portfolio_evaluator_prompt(stable_performers=state["stable_performers"],potential_reducers=state['potential_reducers'])
        llm_response = llm.invoke(prompt)
        return {
            **state,
            'eval_output' : llm_response.content
        }
    # else:
    #     prompt = prompt_nifty_metrics(nifty_details=state['nifty_details'])
    #     llm_response = llm.invoke(prompt)
    #     return {
    #         **state,
    #         'benchmark_analysis_NIFTY50' : llm_response.content
    #     }
    

def allocation_plan_agent(state: DataState) -> DataState:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
       #add api key
        temperature=0.3
    )

    # stable_performers = state["stable_performers"]
    # potential_reducers = state["potential_reducers"]
    user_needs = state.get("user_needs", "No specific constraints provided")

    prompt = f"""
SYSTEM:
You are a portfolio allocation engine with risk-monitoring capability.
You output ONLY structured allocation and future monitoring notes.
You do NOT explain reasoning.
You do NOT predict returns.
You do NOT give buy/sell advice.

INPUT:
Portfolio data:
{state['user_portfolio']}

User needs (if any):
{user_needs}

TASK:
Generate ONLY the following two sections.
No additional text, headers, or explanations are allowed.

========================
OUTPUT FORMAT (STRICT)
========================

allocation:
Current Stock Allocation:
- STOCK_A: XX%
- STOCK_B: XX%
- STOCK_C: XX%

AI Agent Plan :
-STOCK_A:XX%
-STOCK_B:XX%

potential metrics:
- STOCK_A: <future observation note>
- STOCK_B: <future observation note>
- STOCK_C: <future observation note>

Your Question:
    - Response to User Question(User Needs) if None leave it and also reason 

========================
RULES:
- Allocation MUST sum to exactly 100%
- Stable stocks must have higher allocation
- Risky stocks must be capped
- Potential metrics must be conditional and future-focused
- Do NOT forecast prices or returns
- Do NOT add any other sections or text
"""


    llm_response = llm.invoke(prompt)

    return {
        **state,
        "allocation_plan": llm_response.content
    }






































def intent_classifier_agent(state: DataState) -> DataState:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        #add api key
        temperature=0.0
    )

    user_input = state["user_needs"]

    prompt = f"""
SYSTEM:
You are a STRICT intent classification engine.
Your job is to classify user intent into EXACTLY ONE category.

ALLOWED OUTPUT VALUES (ONLY ONE):
- allocation_plan
- reasoning

INTENT RULES (MANDATORY):
- Choose allocation_plan ONLY if the user is asking for actions, plans, allocation, or investment ideas
- Choose reasoning ONLY if the user is asking for explanation, understanding, or analysis
- If unclear, ALWAYS choose reasoning

USER INPUT:
{user_input}

STRICT OUTPUT RULES:
- Output ONLY one of the two labels
- No explanation
- No punctuation
- No extra words
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "intent": response.content.strip()
    }





def decision_node_1(state : DataState) -> str:
    if state['intent'] == 'allocation_plan': 
        return "allocation_plan"  
    else:
        return "eval"


# def decision_node_2(state : DataState) -> str:
#     if int(state['risk_rate']) > 6:
#         return 'benchmark'
#     else:
#         return 'report'

# def decision_node_3(state : DataState) -> str:
#     if 'benchmark_analysis_NIFTY50' not in state:
#         return 'reasoning'
#     else:
#         return 'report'




graph = StateGraph(DataState)
graph.add_node('metrics_fetcher',common_metrics_node)
graph.add_node('signals_getters',common_signals_node)
graph.add_node('intent_agent',intent_classifier_agent)
graph.add_node('eval_agent',llm_portfolio_eval_agent)
graph.add_node('allocation_plan_agent',allocation_plan_agent)
# graph.add_node('benchmark_agent',benchmark_comparison_node)
# graph.add_node('final_analysis_with_bench_mark',final_analysis_agent)

graph.add_edge(START,'metrics_fetcher')
graph.add_edge('metrics_fetcher','signals_getters')
graph.add_edge('signals_getters','intent_agent')
# graph.add_edge('risk_analysis_agent',END)

graph.add_conditional_edges(
    "intent_agent",
    decision_node_1,
    {
        "eval" : 'eval_agent',
        'allocation_plan' : 'allocation_plan_agent'
    }
)



graph.add_edge('eval_agent',END)
graph.add_edge('allocation_plan_agent','eval_agent')



graph = graph.compile()


# llm = ChatGroq(
#     model="llama3-70b-8192",
#     temperature=0.3
# )


user_portfolio = [
    {"symbol": "INFY.NS", "type": "stock", "quantity": 10},
    {"symbol": "HDFCBANK.NS", "type": "stock", "quantity": 20},
    {"symbol": "TATASTEEL.NS", "type": "stock", "quantity": 25},
    {"symbol": "IDFCFIRSTB.NS", "type": "stock", "quantity": 50},
    {"symbol": "ITC.NS", "type": "stock", "quantity": 4},
    {"symbol": "OLAELEC.NS", "type": "stock", "quantity": 1500},
    {"symbol": "BHARTIARTL.NS", "type": "stock", "quantity": 10},
    # {"symbol": "GOLD", "type": "commodity", "quantity": 30}  
]


# result = graph.invoke({
#     "user_portfolio": user_portfolio,  
#     "user_needs": {
#     'risk_tolerance' : "low",
# }                   
# })



# # for val in result:
# #     pprint(result[val])
    
    


