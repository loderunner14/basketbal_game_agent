import os
from typing import Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
# Note: AgentExecutor not available in this LangChain version, using manual tool calling
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


def _create_college_basketball_search_tool():
    """Create a LangChain tool for searching college basketball games.
    
    This tool can be used by the agent autonomously to search for game information.
    """
    search_tool = DuckDuckGoSearchRun()
    ddgs = DDGS()

    @tool
    def search_college_basketball_games(query: str) -> str:
        """Search the web for information about college basketball games.
        
        Use this tool to find:
        - Game schedules for specific teams or all teams
        - Game dates, times, and TV channels
        - Scores and results
        - Tournament information
        
        Args:
            query: Search query describing what you're looking for. Examples:
                - "college basketball schedule next 7 days"
                - "Duke basketball games this week"
                - "NCAA basketball schedule"
                - "all college basketball games"
        
        Returns:
            Search results with game information and source URLs.
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
            
            # Format URLs for inclusion in results
            if all_urls:
                urls_text = "\n\nSource URLs:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(list(all_urls))])
            else:
                urls_text = ""
            
            return combined + urls_text
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
                
                # Format URLs for inclusion in results
                if all_urls:
                    urls_text = "\n\nSource URLs:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(list(all_urls))])
                else:
                    urls_text = ""
                
                return result + urls_text
            except Exception as e:
                print(f"[DEBUG] Search error: {e}\n")
                return f"Search error: {e}"
        else:
            return "No search results found."

    return search_college_basketball_games


def build_college_basketball_agent(
    model: str = "llama3.2:3b",
):
    """Create an autonomous LangChain agent that can use web search tools.
    
    The agent will:
    - Analyze user input autonomously
    - Decide when to use the search tool
    - Extract necessary information from queries
    - Use tools to find college basketball game information
    - Provide comprehensive answers with sources
    """
    _init_env()
    llm = _get_llm(model=model)
    
    # Create the search tool
    search_tool = _create_college_basketball_search_tool()
    tools = [search_tool]
    
    # Get current date information
    current_date, end_date = _get_date_range(7)
    print(f"[DEBUG] Current date: {current_date}")
    print(f"[DEBUG] Date range (next 7 days): {current_date} to {end_date}\n")
    
    # Create a ReAct-style agent that can reason and use tools
    # Since ChatOllama doesn't support bind_tools, we'll use a manual approach
    
    def agent_call(question: str) -> str:
        """Autonomous ReAct-style agent that reasons and uses tools."""
        # Create a ReAct prompt that guides the agent
        react_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                f"You are an intelligent assistant that helps users find college basketball game information.\n\n"
                f"Today's date is {current_date}. The next 7 days are from {current_date} to {end_date}.\n\n"
                f"You have access to a search tool. Follow these steps:\n"
                f"1. THOUGHT: Analyze what information you need\n"
                f"2. If you need current information, use: ACTION: SEARCH: <search query>\n"
                f"3. After getting search results, immediately provide: ACTION: ANSWER: <your comprehensive answer>\n\n"
                f"IMPORTANT RULES:\n"
                f"- Use the search tool ONCE to get current information\n"
                f"- After receiving search results, ALWAYS provide your final answer\n"
                f"- Do NOT search URLs or try to search multiple times\n"
                f"- Do NOT search again after you have search results\n\n"
                f"Available tool:\n"
                f"- search_college_basketball_games(query): Search for college basketball games\n\n"
                f"Your answer must include:\n"
                f"- All relevant games organized by date\n"
                f"- Dates, times, teams, and TV channels\n"
                f"- A 'Sources:' section at the end with all URLs from search results\n\n"
                f"Format:\n"
                f"THOUGHT: <your reasoning>\n"
                f"ACTION: SEARCH: <query> (use once)\n"
                f"OR\n"
                f"ACTION: ANSWER: <your comprehensive answer> (use after search)\n"
            )),
            ("human", "{input}"),
        ])
        
        search_tool = tools[0]  # Get the search tool
        max_iterations = 5
        iteration = 0
        search_results = []
        source_urls = []
        
        enhanced_question = question
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n[DEBUG] Agent iteration {iteration}")
            
            # Get agent's reasoning and action
            response = llm.invoke(react_prompt.format(input=enhanced_question))
            response_text = response.content
            print(f"[DEBUG] Agent response: {response_text[:200]}...")
            
            # If we already have search results and agent tries to search again, force answer
            if len(search_results) > 0 and ("ACTION: SEARCH:" in response_text or "SEARCH:" in response_text):
                print(f"[DEBUG] Agent tried to search again, but we already have results. Forcing answer.")
                enhanced_question = (
                    f"Original question: {question}\n\n"
                    f"You already have search results from {len(search_results)} search(es). "
                    f"Please provide your final answer NOW using: ACTION: ANSWER: <your comprehensive answer>\n\n"
                    f"Combine all search results and provide:\n"
                    f"- All games organized by date\n"
                    f"- Dates, times, teams, TV channels\n"
                    f"- Sources section with all URLs\n\n"
                    f"Latest search results:\n{search_results[-1]}"
                )
            # Check if agent wants to search (and we don't have results yet)
            elif "ACTION: SEARCH:" in response_text or ("SEARCH:" in response_text and "ANSWER:" not in response_text):
                # Extract search query
                import re
                search_match = re.search(r'SEARCH:\s*(.+)', response_text, re.IGNORECASE)
                if search_match:
                    search_query = search_match.group(1).strip()
                    # Remove any trailing ACTION or ANSWER text
                    search_query = re.split(r'\s+ACTION:|ANSWER:', search_query)[0].strip()
                    
                    # Don't search if it's a URL (agent is confused)
                    if search_query.startswith("http://") or search_query.startswith("https://"):
                        print(f"[DEBUG] Agent tried to search a URL, redirecting to provide answer")
                        enhanced_question = (
                            f"Original question: {question}\n\n"
                            f"You have search results available. Please provide your final answer now using: "
                            f"ANSWER: <your comprehensive answer with games, dates, times, teams, TV channels, and Sources section>"
                        )
                    else:
                        print(f"[DEBUG] Agent decided to search: '{search_query}'")
                        
                        # Execute search
                        try:
                            result = search_tool.invoke({"query": search_query})
                            search_results.append(result)
                            print(f"[DEBUG] Search returned {len(result)} characters")
                            
                            # Extract URLs from result if present
                            if "Source URLs:" in result:
                                url_section = result.split("Source URLs:")[1]
                                urls = re.findall(r'https?://[^\s\n]+', url_section)
                                source_urls.extend(urls)
                            
                            # If we have search results, guide agent to provide answer
                            if len(search_results) >= 1:
                                enhanced_question = (
                                    f"Original question: {question}\n\n"
                                    f"Search results from {len(search_results)} search(es):\n{result}\n\n"
                                    f"You now have search results. Please provide your final comprehensive answer using: "
                                    f"ANSWER: <your answer>\n\n"
                                    f"Your answer should:\n"
                                    f"- List all relevant games organized by date\n"
                                    f"- Include dates, times, teams, and TV channels\n"
                                    f"- End with a 'Sources:' section listing all URLs from the search results\n"
                                    f"- Be comprehensive and well-organized"
                                )
                            else:
                                enhanced_question = (
                                    f"Original question: {question}\n\n"
                                    f"Search results:\n{result}\n\n"
                                    f"Based on the search results above, provide a comprehensive answer to the original question. "
                                    f"Include all relevant games with dates, times, teams, and TV channels. "
                                    f"At the end, include a 'Sources:' section with all URLs from the search results."
                                )
                        except Exception as e:
                            print(f"[DEBUG] Search error: {e}")
                            enhanced_question = (
                                f"Original question: {question}\n\n"
                                f"Search failed with error: {e}. Please provide an answer based on your knowledge, "
                                f"but note that the information may not be current."
                            )
                else:
                    # Couldn't extract search query, ask agent to provide answer
                    enhanced_question = (
                        f"Original question: {question}\n\n"
                        f"You have search results available. Please provide your final answer now using: "
                        f"ANSWER: <your comprehensive answer>"
                    )
            elif "ANSWER:" in response_text or (len(search_results) > 0 and iteration >= 2) or iteration >= max_iterations:
                # Agent is providing final answer
                print(f"[DEBUG] Agent providing final answer")
                
                # Extract answer if marked with ANSWER:
                if "ANSWER:" in response_text:
                    # Get everything after ANSWER:
                    answer_parts = response_text.split("ANSWER:")
                    if len(answer_parts) > 1:
                        answer = answer_parts[-1].strip()
                        # Remove any remaining ACTION or SEARCH lines
                        lines = answer.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            if not line.strip().startswith("ACTION:") and not line.strip().startswith("SEARCH:"):
                                cleaned_lines.append(line)
                        answer = '\n'.join(cleaned_lines).strip()
                    else:
                        answer = response_text
                else:
                    # No ANSWER: marker, try to extract useful content
                    # Remove THOUGHT and ACTION lines
                    lines = response_text.split('\n')
                    cleaned_lines = []
                    skip_next = False
                    for line in lines:
                        if line.strip().startswith("THOUGHT:") or line.strip().startswith("ACTION:"):
                            skip_next = True
                            continue
                        if skip_next and line.strip() == "":
                            skip_next = False
                            continue
                        if not skip_next:
                            cleaned_lines.append(line)
                    answer = '\n'.join(cleaned_lines).strip()
                    if not answer or len(answer) < 50:
                        # Fallback: use full response
                        answer = response_text
                
                # Ensure sources are included
                if source_urls and "Sources:" not in answer and "Source URLs:" not in answer:
                    unique_urls = list(set(source_urls))
                    answer += f"\n\nSources:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(unique_urls)])
                elif not source_urls and search_results:
                    # Try to extract URLs from search results
                    import re
                    for result in search_results:
                        if "Source URLs:" in result:
                            url_section = result.split("Source URLs:")[1]
                            urls = re.findall(r'https?://[^\s\n]+', url_section)
                            source_urls.extend(urls)
                    if source_urls and "Sources:" not in answer:
                        unique_urls = list(set(source_urls))
                        answer += f"\n\nSources:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(unique_urls)])
                
                return answer
            else:
                # Agent is still thinking, continue
                enhanced_question = (
                    f"Original question: {question}\n\n"
                    f"Your previous response: {response_text}\n\n"
                    f"Please decide: either use SEARCH: <query> to search for information, "
                    f"or ANSWER: <your answer> to provide the final response."
                )
        
        # Max iterations reached
        final_response = response_text if 'response_text' in locals() else "Unable to complete request."
        if source_urls and "Sources:" not in final_response:
            unique_urls = list(set(source_urls))
            final_response += f"\n\nSources:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(unique_urls)])
        return final_response
    
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

