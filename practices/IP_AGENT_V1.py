from typing import TypedDict, Optional, List, Dict
import feedparser
from dotenv import load_dotenv
import json
import os

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq  


class NewsState(TypedDict):
    news_texts: Optional[List[str]]
    merged_news: Optional[str]

    relevance: Optional[bool]
    sentiment: Optional[str]
    impact: Optional[str]
    action: Optional[List[Dict]]


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("API_KEY_GROQ")
)


def multi_website_news_loader(state: NewsState):
    feeds = {
        "ECONOMIC_TIMES": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    }

    articles = []

    for source, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            title = entry.get("title", "No Title")
            summary = (
                entry.get("summary")
                or entry.get("description")
                or (entry.get("content")[0].get("value") if entry.get("content") else "")
                or "No summary available"
            )
            articles.append(f"[{source}] {title} - {summary}")

    return {"news_texts": articles}


def merge_news_node(state: NewsState):
    combined_news = "\n\n".join(state["news_texts"])[:8000]
    return {"merged_news": combined_news}


def analysis_node(state: NewsState):
    prompt = f"""
You are a senior financial analyst.

Analyze the following aggregated financial news.

Return STRICT JSON ONLY in this format:
{{
  "relevance": true | false,
  "sentiment": "Positive | Negative | Neutral",
  "impact": "Short paragraph describing short-term and long-term market impact",
  "action": [
    {{
      "company": "Company name",
      "trend": "Bullish | Bearish | Neutral",
      "reason": "Why",
      "decision_logic": "How decision is made",
      "decision": "Buy | Sell | Hold | Ignore",
      "duration": "Short-term | Long-term"
    }}
  ]
}}

Rules:
- Be conservative
- No assumptions beyond the news
- Financial risk awareness is mandatory

News:
{state['merged_news']}
"""

    response = llm.invoke(prompt).content.strip()

    if not response:
        raise RuntimeError("Empty response from LLM")

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        print("RAW LLM OUTPUT:\n", response)
        raise ValueError("LLM did not return valid JSON")

    return {
        "relevance": parsed["relevance"],
        "sentiment": parsed["sentiment"],
        "impact": parsed["impact"],
        "action": parsed["action"]
    }


graph = StateGraph(NewsState)

graph.add_node("load_news", multi_website_news_loader)
graph.add_node("merge_news", merge_news_node)
graph.add_node("analysis", analysis_node)

graph.set_entry_point("load_news")
graph.add_edge("load_news", "merge_news")
graph.add_edge("merge_news", "analysis")
graph.add_edge("analysis", END)

graph = graph.compile()



# if __name__ == "__main__":
#     print("I am in ")
#     result = app.invoke({})
#     print("\n================ FINAL ANALYSIS ================\n")
#     print("RELEVANCE :", result["relevance"])
#     print("SENTIMENT :", result["sentiment"])
#     print("\nIMPACT ANALYSIS:\n", result["impact"])
#     print("\nACTIONS:")
#     for a in result["action"]:
#         print(a)
