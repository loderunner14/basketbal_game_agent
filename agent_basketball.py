import os
from typing import Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Using bind_tools directly since create_react_agent not available in this version
from langchain_ollama import ChatOllama
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
    """Create an autonomous LangChain agent that can use web search tools with bind_tools support.
    
    The agent will:
    - Analyze user input autonomously
    - Decide when to use the search tool using bind_tools
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
    
    # Create agent prompt with date context
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an intelligent assistant that helps users find detailed, up-to-date "
                "information about **college basketball games** using web search tools.\n\n"
                f"IMPORTANT CONTEXT:\n"
                f"- Today's date is {current_date}\n"
                f"- When users ask about 'next 7 days', this means from {current_date} to {end_date}\n"
                f"- Always use the actual current date when interpreting date ranges\n\n"
                "You have access to a search tool. When a user asks about college basketball games:\n"
                "1. Analyze their query to understand what they need (specific teams, date range, etc.)\n"
                "2. Use the search_college_basketball_games tool to find current information\n"
                "3. Provide a comprehensive answer based on the search results\n\n"
                "When providing answers:\n"
                "- For 'all games' queries, provide a COMPREHENSIVE list of ALL games across ALL teams and conferences\n"
                "- Organize games by date and time\n"
                "- Include dates, times, teams, TV channels (if available), and tournaments\n"
                "- At the end of your response, include a 'Sources:' section listing all source URLs from the search results\n"
                "- Be specific about dates and use the actual current date provided above"
            ),
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    # Bind tools to the LLM (this works with langchain-ollama ChatOllama)
    llm_with_tools = llm.bind_tools(tools)
        
    def agent_call(question: str) -> str:
        """Autonomous agent using bind_tools to call search tool."""
        # Add date context
        enhanced_question = (
            f"Today's date is {current_date}. The next 7 days are from {current_date} to {end_date}. "
            f"{question}"
        )
        
        # Create messages list using the prompt template
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
        
        # Format the prompt to get messages (provide empty scratchpad initially)
        formatted_messages = prompt.format_messages(
            input=enhanced_question,
            agent_scratchpad=[],
            chat_history=[]
        )
        messages = list(formatted_messages)
        
        # Set maximum iterations to prevent infinite loops in the agent's reasoning cycle.
        # The agent operates in a loop: it can call tools, receive results, and decide to call
        # more tools or provide a final answer. Without this limit, the agent could get stuck
        # in an infinite loop if it:
        # - Keeps trying to call tools repeatedly without making progress
        # - Can't decide on a final answer and keeps requesting more information
        # - Gets confused and oscillates between tool calls and responses
        # A limit of 5 iterations is typically sufficient for most queries (1-2 tool calls + final answer)
        # while preventing runaway execution that could consume excessive resources or time.
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n[DEBUG] Agent iteration {iteration}")
            
            # Get LLM response (may include tool calls)
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if LLM wants to call tools
            if hasattr(response, 'tool_calls') and response.tool_calls and len(response.tool_calls) > 0:
                print(f"[DEBUG] Agent decided to use tools: {len(response.tool_calls)} tool call(s)")
                
                # Execute all tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                    tool_query = tool_args.get("query", str(tool_args))
                    
                    print(f"[DEBUG] Executing tool: {tool_name}")
                    print(f"[DEBUG] Tool query: {tool_query}")
                    
                    # Find and execute the tool
                    tool_executed = False
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                result = tool.invoke(tool_args)
                                # Add tool result as ToolMessage
                                messages.append(ToolMessage(
                                    content=result,
                                    tool_call_id=tool_id
                                ))
                                print(f"[DEBUG] Tool returned {len(result)} characters")
                                tool_executed = True
                                break
                            except Exception as e:
                                print(f"[DEBUG] Tool execution error: {e}")
                                messages.append(ToolMessage(
                                    content=f"Error: {e}",
                                    tool_call_id=tool_id
                                ))
                                tool_executed = True
                                break
                    
                    if not tool_executed:
                        print(f"[DEBUG] Warning: Tool {tool_name} not found")
                        messages.append(ToolMessage(
                            content=f"Tool {tool_name} not found",
                            tool_call_id=tool_id
                        ))
            else:
                # No tool calls, return the final answer
                print(f"[DEBUG] Agent completed in {iteration} iteration(s)")
                answer = response.content
                
                # Ensure sources are included
                if "Sources:" not in answer and "Source URLs:" not in answer:
                    import re
                    # Extract URLs from tool messages
                    urls = []
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            msg_urls = re.findall(r'https?://[^\s\n]+', msg.content)
                            urls.extend(msg_urls)
                    # Also check answer
                    answer_urls = re.findall(r'https?://[^\s\n]+', answer)
                    urls.extend(answer_urls)
                    
                    if urls:
                        unique_urls = list(set(urls))
                        answer += f"\n\nSources:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(unique_urls)])
                
                return answer
        
        # Max iterations reached: The agent has exceeded the maximum number of iterations.
        # This safety mechanism prevents infinite loops. When this happens, we return the
        # last response from the agent (if available) rather than continuing indefinitely.
        # In practice, this should rarely occur if the agent is working correctly, but it
        # protects against edge cases where the agent might get stuck in a reasoning loop.
        final_answer = response.content if 'response' in locals() else "Agent reached maximum iterations."
        
        # Ensure sources
        if "Sources:" not in final_answer:
            import re
            urls = re.findall(r'https?://[^\s\n]+', final_answer)
            if urls:
                unique_urls = list(set(urls))
                final_answer += f"\n\nSources:\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(unique_urls)])
        
        return final_answer
    
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

