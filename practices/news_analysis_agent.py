from typing import TypedDict, Optional, List
import feedparser
from dotenv import load_dotenv
import os
import json

from langchain_groq import ChatGroq


class NewsState(TypedDict):
    news_texts: Optional[List[str]]
    merged_news: Optional[str]
    analysis_result: Optional[dict]


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("API_KEY_GROQ")
)
def load_and_merge_news() -> str:
    feeds = {
        "ECONOMIC_TIMES": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        
    }
    
    feeds = {
    # "ECONOMIC_TIMES_MARKETS": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    # "ECONOMIC_TIMES_ECONOMY": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    # "ECONOMIC_TIMES_PERSONAL_FINANCE": "https://economictimes.indiatimes.com/wealth/rssfeeds/837555174.cms",

    "MONEYCONTROL_NEWS": "https://www.moneycontrol.com/rss/latestnews.xml",
    "MONEYCONTROL_BUSINESS": "https://www.moneycontrol.com/rss/business.xml",
    "MONEYCONTROL_MARKETS": "https://www.moneycontrol.com/rss/marketreports.xml",
    "MONEYCONTROL_PERSONAL_FINANCE": "https://www.moneycontrol.com/rss/personalfinance.xml",

    # "FINANCIAL_EXPRESS_MARKET": "https://www.financialexpress.com/market/feed/",
    # "FINANCIAL_EXPRESS_ECONOMY": "https://www.financialexpress.com/economy/feed/"
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

    combined_news = "\n\n".join(articles)
    return combined_news

def full_analysis(news_text: str) -> dict:
    prompt = f"""
You are a senior financial analyst and market strategist.

Analyze the following aggregated financial news and provide all results in **STRICT JSON ONLY**:

1. Relevance: true or false
2. Sentiment: Positive, Negative, Neutral
3. Impact: Short paragraph describing short-term and long-term market impact
4. Actions: For each company mentioned, provide:
   - company
   - trend: Bullish, Bearish, Neutral
   - reason: why
   - decision_logic: how the decision is made
   - decision: Buy, Sell, Hold, Ignore 
   - duration: Short-term, Long-term 
   - duration_reason : whey we go for this duration

JSON format:
{{
  "relevance": true | false,
  "sentiment": "Positive | Negative | Neutral",
  "impact": "Short paragraph",
  "action": [
    {{
      "company": "Company name",
      "trend": "Bullish | Bearish | Neutral",
      "reason": "Why? with respect to this company and provide the proof",
      "decision": "Buy | Sell | Hold | Ignore",
      "decision_logic": "How decision is made",
      "duration": "Short-term | Long-term",
      "duration_reason" : reason for the duration
    }}
  ]
}}

Rules:
- Be conservative
- Do NOT assume anything beyond the news
- Financial risk awareness is mandatory
- The JSON formats should be valid
- Don't compare any stocks with any other
- Don't assume unnecessary

MAKE THE LLM TO RETURN VALID JSON, REVERIFY IT

News:
{news_text}
"""

    response = llm.invoke(prompt).content.strip()

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        print("RAW LLM OUTPUT:\n", response)
        raise ValueError("LLM did not return valid JSON")

    return parsed


if __name__ == "__main__":
    news_text = load_and_merge_news()
    result = full_analysis(news_text)

    print("\n FIAL ANALYSIS ================\n")
    print("RELEVANCE :", result.get("relevance"))
    print("SENTMENT :", result.get("sentiment"))
    print("\nIMPACT ANALYSIS:\n", result.get("impact"))
    print("\nRECMMENDED ACTIONS:")
    for a in result.get("action", []):
        print(a)
