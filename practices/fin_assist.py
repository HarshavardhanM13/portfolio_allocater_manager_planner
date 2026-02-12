from typing import TypedDict, List, Optional
import feedparser
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("API_KEY_GROQ"),
    temperature=0
)

class AgentState(TypedDict):
    conversation_history: List[dict]
    market_memory: Optional[str]
    last_analysis: Optional[str]


def fetch_initial_market_snapshot() -> str:
    feeds = {
        "ECONOMIC_TIMES": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "MONEYCONTROL": "https://www.moneycontrol.com/rss/marketreports.xml"
    }

    news = []
    for src, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:4]:
            title = entry.get("title", "")
            summary = entry.get("summary") or entry.get("description") or ""
            news.append(f"[{src}] {title} - {summary}")

    return "\n".join(news)


def decide_mode(user_input: str) -> str:
    prompt = f"""
Classify the user message into ONE category only:
- ANALYSIS (explicit request for report, analysis, brief, outlook)
- CHAT (normal finance conversation, opinion, follow-up)
- NON_FINANCE (not related to finance)

Return ONLY one word.

User message:
"{user_input}"
"""
    result = llm.invoke(prompt).content.strip()
    if result not in ["ANALYSIS", "CHAT", "NON_FINANCE"]:
        return "CHAT"
    return result


def fetch_targeted_news(keyword: str) -> str:
    feeds = {
        "ECONOMIC_TIMES": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "MONEYCONTROL": "https://www.moneycontrol.com/rss/marketreports.xml"
    }

    articles = []
    for src, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary") or entry.get("description") or ""
            if keyword.lower() in title.lower() or keyword.lower() in summary.lower():
                articles.append(f"[{src}] {title} - {summary}")

    return "\n".join(articles[:4])


def is_memory_sufficient(memory: str, question: str) -> bool:
    prompt = f"""
Answer YES or NO.

Does the following market memory contain enough information
to answer the user question reliably?

Memory:
{memory}

Question:
{question}
"""
    response = llm.invoke(prompt).content.strip()
    return response == "YES"

# 
def chat_response(state: AgentState, user_input: str) -> str:
    memory_ok = is_memory_sufficient(state["market_memory"], user_input)

    context = state["market_memory"]
    if not memory_ok:
        context += "\n\nAdditional info:\n" + fetch_targeted_news(user_input)

    prompt = f"""
You are a simple, helpful Indian stock market assistant.

STRICT RULES:
- Talk ONLY about Indian stock market (NSE/BSE)
- Ignore crypto, US stocks, forex, global markets
- Use simple language
- Short answers (2–4 lines)
- Answer ONLY what is asked
- No reports, no lists unless required

Indian Market Context:
{context}

User Question:
{user_input}
"""
    return llm.invoke(prompt).content.strip()

# =
def analysis_response(state: AgentState, user_input: str) -> str:
    prompt = f"""
You are an Indian equity market expert.

Rules:
- Indian stocks only
- Simple words
- Clear buy/hold/sell view
- No unnecessary jargon
- If referring to stocks the consider it will be within the provided amount if not don't consider it 
- always consider a complete stock (ex : 1, 3, 300) not like (ex : 0.3, 0.6)

Structure:
Sentiment:
Impact:
Decision:
Reason:

Indian Market Data:
{state["market_memory"]}

User Request:
{user_input}
"""
    analysis = llm.invoke(prompt).content.strip()
    state["last_analysis"] = analysis
    return analysis


if __name__ == "__main__":
    state: AgentState = {
        "conversation_history": [],
        "market_memory": fetch_initial_market_snapshot(),
        "last_analysis": None
    }

    print("Financial Assistant Ready\nType 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        mode = decide_mode(user_input)

        if mode == "NON_FINANCE":
            response = "I don’t have reliable information on that topic."
        elif mode == "ANALYSIS":
            response = analysis_response(state, user_input)
        else:
            response = chat_response(state, user_input)

        state["conversation_history"].append({
            "user": user_input,
            "ai": response
        })

        print("\nAI:", response)
        print("-" * 80)
