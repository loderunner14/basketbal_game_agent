import os
from typing import Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from ddgs import DDGS


def _init_env() -> None:
    """Load environment variables from a local .env if present.

    Currently we don't require any keys; this is just here if you want
    to add config later.
    """
    load_dotenv(override=False)


def _get_date_range(days: int = 7) -> tuple[str, str]:
    """Get the current date and calculate the date range for the next N days.
    
    Returns:
        Tuple of (current_date_str, end_date_str) in format "Month Day, Year"
    """
    today = datetime.now()
    end_date = today + timedelta(days=days)
    
    current_date_str = today.strftime("%B %d, %Y")
    end_date_str = end_date.strftime("%B %d, %Y")
    
    return current_date_str, end_date_str


def _get_llm(model: str = "llama3.2:3b") -> ChatOllama:
    """Create the Ollama chat model used by the agent.

    Expects a local Ollama server with the given model pulled, e.g.:
    `ollama pull llama3.2:3b`
    """
    return ChatOllama(model=model, temperature=0.1)


def _make_ddg_search_fn():
    """Return a function that searches the web for college basketball games via DuckDuckGo.

    Uses LangChain's built-in DuckDuckGo search tool, which is free and requires no API key.
    Returns both search results text and a list of source URLs.
    """
    search_tool = DuckDuckGoSearchRun()
    ddgs = DDGS()

    def search_college_basketball(query: str) -> tuple[str, list[str]]:
        """Search the web for information about college basketball games.

        The query should describe:
        - specific teams (e.g. 'Duke vs UNC')
        - date range (e.g. 'this week', 'March 2026')
        - type of info (scores, schedule, betting lines, TV info, etc.)
        """
        # Extract key terms from the query - focus on team names and dates
        import re
        
        # Remove question marks and punctuation
        query_clean = re.sub(r'[?.,!]', '', query)
        words = query_clean.split()
        
        # Question words to remove
        question_words = {'what', 'when', 'where', 'who', 'how', 'which', 'are', 'is', 'the', 'a', 'an', 
                         'this', 'that', 'these', 'those', 'involving', 'happening', 'games', 'game'}
        
        # Extract team names (capitalized words, likely proper nouns)
        team_names = []
        date_terms = []
        
        date_keywords = {'week', 'today', 'tonight', 'tomorrow', 'schedule', 'scores', 'upcoming', 'next', 'tonight', 
                        'days', '7', 'seven'}
        
        # Look for "next X days" pattern
        query_lower = query.lower()
        if 'next' in query_lower and 'days' in query_lower:
            # Extract number before "days"
            import re
            days_match = re.search(r'next\s+(\d+)\s+days?', query_lower)
            if days_match:
                num_days = days_match.group(1)
                date_terms.extend(['next', num_days, 'days'])
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            # Skip question words
            if word_lower in question_words:
                continue
            # Keep capitalized words (likely team names like "Duke", "UNC", "Kansas")
            if word and word[0].isupper() and len(word) > 1:
                team_names.append(word)
            # Keep date-related terms (but skip if we already got "next X days")
            elif word_lower in date_keywords and not (word_lower == 'days' and 'next' in date_terms):
                date_terms.append(word_lower)
        
        # Check if query asks for ALL games (not specific teams)
        query_lower = query.lower()
        wants_all_games = 'all' in query_lower or ('every' in query_lower and 'game' in query_lower)
        
        # Build search query from extracted terms
        if team_names and not wants_all_games:
            # Use team names as primary focus (specific team search)
            team_part = ' '.join(team_names[:2])  # Max 2 team names
            if date_terms:
                # Include all date terms to capture "next 7 days"
                date_part = ' '.join(date_terms[:3])  # Include up to 3 date terms
                clean_query = f"{team_part} college basketball schedule {date_part}"
            else:
                clean_query = f"{team_part} college basketball schedule"
        else:
            # Search for ALL college basketball games (no specific team)
            if date_terms:
                date_part = ' '.join(date_terms[:3])  # Include up to 3 date terms
                clean_query = f"college basketball schedule {date_part}"
            else:
                clean_query = "college basketball schedule"
        
        print(f"\n[DEBUG] Original query: {query}")
        print(f"[DEBUG] Wants all games: {wants_all_games}")
        print(f"[DEBUG] Search query: {clean_query}\n")
        
        # Try multiple search queries for comprehensive results
        search_queries = [clean_query]
        
        if team_names and not wants_all_games:
            # Add variations for specific team searches
            team_part = ' '.join(team_names[:2])
            search_queries.extend([
                f"{team_part} basketball schedule next 7 days",
                f"{team_part} basketball games this week",
                f"{team_part} upcoming games",
            ])
        else:
            # Add variations for ALL games searches
            if date_terms:
                date_part = ' '.join(date_terms[:3])
                search_queries.extend([
                    f"NCAA basketball schedule {date_part}",
                    f"college basketball games {date_part}",
                    f"all college basketball games {date_part}",
                ])
            else:
                search_queries.extend([
                    "NCAA basketball schedule",
                    "college basketball games this week",
                    "all college basketball games",
                ])
        
        all_results = []
        all_urls = set()  # Use set to avoid duplicates
        
        for search_q in search_queries[:3]:  # Try up to 3 queries
            try:
                print(f"[DEBUG] Searching: '{search_q}'")
                result = search_tool.run(search_q)
                if result and len(result) > 100:  # Only add if substantial results
                    all_results.append(result)
                    print(f"[DEBUG] Found {len(result)} characters")
                
                # Also get URLs using ddgs directly
                try:
                    ddgs_results = ddgs.text(search_q, max_results=5)
                    if ddgs_results:
                        for item in ddgs_results:
                            url = item.get("href", "") or item.get("url", "")
                            if url:
                                all_urls.add(url)
                except Exception as url_error:
                    print(f"[DEBUG] URL extraction error: {url_error}")
                    continue
            except Exception as e:
                print(f"[DEBUG] Search error for '{search_q}': {e}")
                continue
        
        if all_results:
            # Combine results, removing duplicates
            combined = "\n\n--- Additional Search Results ---\n\n".join(all_results)
            print(f"[DEBUG] Combined search returned {len(combined)} characters")
            print(f"[DEBUG] Found {len(all_urls)} unique source URLs\n")
            return combined, list(all_urls)
        elif all_results == []:
            # Fallback to single query
            try:
                result = search_tool.run(clean_query)
                print(f"[DEBUG] Fallback search returned {len(result)} characters")
                
                # Get URLs for fallback query
                try:
                    ddgs_results = ddgs.text(clean_query, max_results=5)
                    if ddgs_results:
                        for item in ddgs_results:
                            url = item.get("href", "") or item.get("url", "")
                            if url:
                                all_urls.add(url)
                except Exception:
                    pass
                
                print(f"[DEBUG] Found {len(all_urls)} unique source URLs\n")
                return result, list(all_urls)
            except Exception as e:
                print(f"[DEBUG] Search error: {e}\n")
                return f"Search error: {e}", []
        else:
            return "No search results found.", []

    return search_college_basketball


def build_college_basketball_agent(
    model: str = "llama3.2:3b",
):
    """Create a callable 'agent' that does web search + LLM reasoning.

    It will:
    - Use DuckDuckGo to search for college basketball game info.
    - Feed the search results plus the user's question into the LLM.
    """
    _init_env()
    llm = _get_llm(model=model)
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
                (
                    "system",
                    (
                        "You are an assistant that helps users find detailed, up-to-date "
                        "information about **college basketball games** using web search. "
                        "Use the provided web search results as your primary source of truth. "
                        "When asked about ALL college basketball games for a specific time period "
                        "(like 'next 7 days'), provide a COMPREHENSIVE list of ALL games across "
                        "ALL teams and conferences in that timeframe. Do not limit to specific teams. "
                        "Include games from all divisions (Division I, II, III) and all conferences. "
                        "Organize the list by date and time. Always include dates, times, teams, "
                        "TV channels (if available), tournaments (if any), and specify how current "
                        "the information appears to be.\n\n"
                        "IMPORTANT: Today's date is {current_date}. When the user asks about "
                        "'next 7 days', this means from {current_date} to {end_date}. "
                        "Use these actual dates when organizing and presenting game information."
                    ),
                ),
            (
                "user",
                (
                    "User question: {question}\n\n"
                    "Web search results:\n{web_results}\n\n"
                    "IMPORTANT: At the end of your response, you MUST include a section titled "
                    "'Sources:' or 'Data Sources:' that lists all the URLs provided below. "
                    "Format each URL on a new line, numbered if there are multiple sources.\n\n"
                    "Source URLs:\n{sources}"
                ),
            ),
        ]
    )

    chain = prompt | llm | parser
    search_fn = _make_ddg_search_fn()

    def agent_call(question: str) -> str:
        # Get current date range for next 7 days
        current_date, end_date = _get_date_range(7)
        print(f"[DEBUG] Current date: {current_date}")
        print(f"[DEBUG] Date range (next 7 days): {current_date} to {end_date}\n")
        web_results, source_urls = search_fn(question)
        
        # Format source URLs for the prompt
        if source_urls:
            sources_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(source_urls)])
        else:
            sources_text = "No source URLs available."
        
        return chain.invoke({
            "question": question, 
            "web_results": web_results,
            "current_date": current_date,
            "end_date": end_date,
            "sources": sources_text
        })

    return agent_call


def run_example(query: Optional[str] = None) -> None:
    """Simple CLI entry point to test the agent."""
    agent = build_college_basketball_agent()

    user_query = query or "What are all the college basketball games for the next 7 days?"
    result = agent(user_query)
    print("\n=== Agent Answer ===\n")
    print(result)


if __name__ == "__main__":
    run_example()

