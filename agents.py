from crewai import Agent, Task, Crew, LLM
import streamlit as st

llm = LLM(
    model="openai/llama-3.3-70b-versatile", 
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

def run_crew(query, stock_context):
    researcher=Agent(
        role="Stock Data Analyst",
        goal="Structure raw stock metrics into a clear data summary",
        backstory="Quantitative analyst who organises market data for decision making.",
        llm=llm
    )
    news_agent = Agent(
        role="News Sentiment Analyst",
        goal="Identify key headlines and interpret market sentiment.",
        backstory="Media analyst who tracks which headlines actually move markets.",
        llm=llm
    )

    strategist = Agent(
        role="Chief Financial Strategist",
        goal="Synthesize data and news into a concise actionable analysis.",
        backstory="Investment strategist who produces clear market outlooks clients trust.",
        llm=llm
    )

    research_task = Task(
        description = f"Analyze this stock data and summarize key metrics:\n{stock_context}",
        expected_output="Bullet point summary of price, trend, volatility, and forecast.",
        agent=researcher
    )

    news_task = Task(
        description=f"Analyze news sentiment for this stock:\n{stock_context}",
        expected_output="Summary of sentiment trend and 2-3 key headlines driving it",
        agent=news_agent
    )

    analysis_task = Task(
        description= f"Using the research and news analysis above, answer this user question:\n{query}",
        expected_output="3-5 sentence professional analysis with clear market outlook",
        agent=strategist
    )

    crew = Crew(
        agents = [researcher, news_agent, strategist],
        tasks = [research_task, news_task, analysis_task],
        verbose=False
    )
    result=crew.kickoff()
    return str(result)