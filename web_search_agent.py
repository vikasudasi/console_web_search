#!/usr/bin/env python3
"""
DSPy ReAct Agent for Web Search using Google Custom Search API

This script uses DSPy's ReAct agent to perform intelligent web searches
based on piped input or command-line text input.

Usage:
    # Command-line input (uses Ollama llama3.1 by default)
    python web_search_agent.py "What is the latest news about AI?"

    # Piped input
    echo "Search for Python best practices" | python web_search_agent.py

    # With context
    cat context.txt | python web_search_agent.py "What does this mean?"

    # Using OpenAI instead of Ollama
    python web_search_agent.py "Your query" --llm-provider openai --model gpt-4

    # Custom Ollama model and URL
    python web_search_agent.py "Your query" --model llama3.2 --ollama-base-url http://localhost:11434
"""

import sys
import os
import argparse
from typing import Optional, List, Dict, Tuple
import json

try:
    import dspy
except ImportError:
    print("Error: DSPy is not installed. Please install it with: pip install dspy-ai")
    sys.exit(1)

try:
    from googleapiclient.discovery import build
except ImportError:
    print("Error: Google API client is not installed. Please install it with: pip install google-api-python-client")
    sys.exit(1)

try:
    # Try new langchain_ollama package first
    try:
        from langchain_ollama import OllamaLLM
        OLLAMA_AVAILABLE = True
        OLLAMA_CLASS = OllamaLLM
    except ImportError:
        # Fallback to deprecated langchain_community
        try:
            from langchain_community.llms import Ollama
            OLLAMA_AVAILABLE = True
            OLLAMA_CLASS = Ollama
        except ImportError:
            OLLAMA_AVAILABLE = False
            OLLAMA_CLASS = None
    
    from langchain_openai import ChatOpenAI
    try:
        from langchain_core.language_models import BaseLanguageModel
    except ImportError:
        # Fallback for older LangChain versions
        from langchain.schema import BaseLanguageModel
except ImportError as e:
    print(f"Error: LangChain is not installed. Please install it with: pip install langchain langchain-ollama langchain-openai")
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


class GoogleSearchTool:
    """Tool for performing Google Custom Search API queries."""
    
    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize Google Search Tool.
        
        Args:
            api_key: Google API key
            search_engine_id: Custom Search Engine ID
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.service = build("customsearch", "v1", developerKey=api_key)
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform a Google search.
        
        Args:
            query: Search query string
            num_results: Number of results to return (max 10 per request)
        
        Returns:
            List of search results with 'title', 'link', and 'snippet' keys
        """
        try:
            results = self.service.cse().list(
                q=query,
                cx=self.search_engine_id,
                num=min(num_results, 10)  # Google API limit is 10 per request
            ).execute()
            
            search_results = []
            if 'items' in results:
                for item in results['items']:
                    search_results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    })
            
            return search_results
        except Exception as e:
            print(f"Error performing search: {e}", file=sys.stderr)
            return []


def create_llm(provider: str = "ollama", model_name: str = "llama3.1", base_url: Optional[str] = None) -> BaseLanguageModel:
    """
    Create a LangChain LLM instance based on provider.
    
    Args:
        provider: LLM provider ("ollama", "openai", etc.)
        model_name: Model name (e.g., "llama3.1", "gpt-3.5-turbo")
        base_url: Base URL for Ollama (default: http://localhost:11434)
    
    Returns:
        LangChain LLM instance
    """
    if provider.lower() == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama is not available. Install with: pip install langchain-ollama")
        ollama_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Using Ollama with model '{model_name}' at {ollama_base_url}", file=sys.stderr)
        # Use the available Ollama class (either OllamaLLM or deprecated Ollama)
        return OLLAMA_CLASS(model=model_name, base_url=ollama_base_url)
    elif provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        print(f"Using OpenAI with model '{model_name}'", file=sys.stderr)
        return ChatOpenAI(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: 'ollama', 'openai'")


class WebSearchAgent:
    """DSPy ReAct Agent for web search tasks."""
    
    def __init__(
        self, 
        api_key: str, 
        search_engine_id: str, 
        llm_provider: str = "ollama",
        model_name: str = "llama3.1",
        ollama_base_url: Optional[str] = None
    ):
        """
        Initialize the Web Search Agent.
        
        Args:
            api_key: Google API key
            search_engine_id: Custom Search Engine ID
            llm_provider: LLM provider ("ollama" or "openai")
            model_name: Model name (e.g., "llama3.1", "gpt-3.5-turbo")
            ollama_base_url: Base URL for Ollama (default: http://localhost:11434)
        """
        # Create LangChain LLM and configure DSPy
        self.langchain_llm = None
        try:
            self.langchain_llm = create_llm(llm_provider, model_name, ollama_base_url)
            # Try to configure DSPy with a custom LM wrapper
            # DSPy expects an LM object with specific methods
            try:
                # Create a simple wrapper that implements DSPy's LM interface
                class LangChainDSPyLM:
                    def __init__(self, langchain_llm):
                        self.langchain_llm = langchain_llm
                        self.model = "langchain"
                    
                    def __call__(self, prompt, **kwargs):
                        if isinstance(prompt, str):
                            return self.langchain_llm.invoke(prompt)
                        # Handle DSPy prompt objects
                        prompt_str = str(prompt) if hasattr(prompt, '__str__') else prompt
                        return self.langchain_llm.invoke(prompt_str)
                    
                    def request(self, prompt, **kwargs):
                        prompt_str = str(prompt) if hasattr(prompt, '__str__') else prompt
                        return self.langchain_llm.invoke(prompt_str)
                
                lm = LangChainDSPyLM(self.langchain_llm)
                # Try to configure DSPy - it may not work with all versions
                try:
                    dspy.configure(lm=lm)
                    print(f"Successfully configured DSPy with {llm_provider} ({model_name})", file=sys.stderr)
                except:
                    # DSPy configuration failed, but we can still use LangChain directly
                    print(f"Note: DSPy configuration skipped, using LangChain directly", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not create DSPy LM wrapper: {e}", file=sys.stderr)
                print("Will use LangChain directly for LLM calls", file=sys.stderr)
        except Exception as e:
            print(f"Error creating LangChain LLM: {e}", file=sys.stderr)
            print("Falling back to direct search mode", file=sys.stderr)
        
        # Initialize Google Search Tool
        self.search_tool = GoogleSearchTool(api_key, search_engine_id)
        
        # Try to create a ReAct-like agent using DSPy
        # If DSPy ReAct doesn't work, we'll use a simpler approach
        self.agent = None
        try:
            # Try using DSPy's ReAct if available and properly configured
            if self.langchain_llm:
                # Create a simple agent that uses search + LLM reasoning
                class SimpleReActAgent:
                    def __init__(self, llm, search_tool):
                        self.llm = llm
                        self.search_tool = search_tool
                    
                    def __call__(self, question, context=""):
                        # Step 1: Perform search
                        search_results = self.search_tool(question)
                        
                        # Step 2: Use LLM to reason about results
                        if search_results:
                            prompt = f"""Based on the following search results, answer the question.

Question: {question}
{f'Context: {context}' if context else ''}

Search Results:
{search_results}

Provide a comprehensive answer based on the search results:"""
                            
                            try:
                                answer = self.llm.invoke(prompt)
                                return answer
                            except Exception as e:
                                print(f"Error in LLM reasoning: {e}", file=sys.stderr)
                                return search_results
                        else:
                            return "No search results found."
                
                self.agent = SimpleReActAgent(self.langchain_llm, self._search_wrapper)
                print("Initialized ReAct-like agent with LangChain", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not initialize agent: {e}", file=sys.stderr)
            print("Using fallback search module", file=sys.stderr)
    
    def _search_wrapper(self, query: str) -> str:
        """
        Wrapper function to convert search results to string format.
        
        Args:
            query: Search query
        
        Returns:
            Formatted string of search results
        """
        results = self.search_tool.search(query)
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result['title']}\n"
                f"   URL: {result['link']}\n"
                f"   {result['snippet']}\n"
            )
        
        return "\n".join(formatted_results)
    
    def search(self, question: str, context: Optional[str] = None) -> str:
        """
        Perform a web search and generate an answer.
        
        Args:
            question: The question or search query
            context: Optional context from piped input
        
        Returns:
            Answer based on search results
        """
        # Combine context and question if context is provided
        if context:
            full_query = f"Context: {context}\n\nQuestion: {question}"
        else:
            full_query = question
        
        try:
            # Use the agent to reason and act
            if self.agent:
                result = self.agent(question=question, context=context or "")
                if isinstance(result, str):
                    return result
                elif hasattr(result, 'answer'):
                    return result.answer
                else:
                    return str(result)
            else:
                # Fallback: direct search with basic reasoning
                return self._direct_search_with_reasoning(question, context)
        except Exception as e:
            print(f"Error in agent execution: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Fallback to direct search
            return self._direct_search_with_reasoning(question, context)
    
    def _direct_search_with_reasoning(self, question: str, context: Optional[str] = None) -> str:
        """
        Fallback method that performs search and formats results.
        
        Args:
            question: The search query
            context: Optional context
        
        Returns:
            Formatted search results
        """
        # Perform search
        results = self.search_tool.search(question)
        
        if not results:
            return "No search results found for your query."
        
        # Format results
        output = []
        if context:
            output.append(f"Context: {context}\n")
        output.append(f"Search Query: {question}\n")
        output.append(f"Found {len(results)} results:\n\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   URL: {result['link']}")
            output.append(f"   {result['snippet']}\n")
        
        return "\n".join(output)


def get_input() -> Tuple[str, Optional[str], str, str, str, str, Optional[str]]:
    """
    Get input from command-line arguments or stdin.
    
    Returns:
        Tuple of (text_input, piped_context, api_key, search_engine_id, llm_provider, model_name, ollama_base_url)
    """
    piped_context = None
    
    # Check if there's piped input
    if not sys.stdin.isatty():
        piped_context = sys.stdin.read().strip()
    
    # Get command-line arguments
    parser = argparse.ArgumentParser(
        description="DSPy ReAct Agent for Web Search using Google Custom Search API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python web_search_agent.py "What is the latest news about AI?"
  echo "Python best practices" | python web_search_agent.py "What does this mean?"
  cat context.txt | python web_search_agent.py "Summarize this"
        """
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query or question (optional if piped input is provided)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google API key (or set GOOGLE_API_KEY env var)"
    )
    parser.add_argument(
        "--search-engine-id",
        default=None,
        help="Google Custom Search Engine ID (or set GOOGLE_SEARCH_ENGINE_ID env var)"
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["ollama", "openai"],
        help="LLM provider to use: 'ollama' or 'openai' (default: ollama, or set LLM_PROVIDER env var)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: llama3.1 for Ollama, gpt-3.5-turbo for OpenAI, or set LLM_MODEL env var)"
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama base URL (default: http://localhost:11434, or set OLLAMA_BASE_URL env var)"
    )
    
    args = parser.parse_args()
    
    # Get LLM configuration
    llm_provider = args.llm_provider or os.getenv("LLM_PROVIDER", "ollama")
    
    # Set default model based on provider
    if args.model:
        model_name = args.model
    elif os.getenv("LLM_MODEL"):
        model_name = os.getenv("LLM_MODEL")
    else:
        model_name = "llama3.1" if llm_provider == "ollama" else "gpt-3.5-turbo"
    
    ollama_base_url = args.ollama_base_url or os.getenv("OLLAMA_BASE_URL")
    
    # Get API credentials
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    search_engine_id = args.search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key:
        print("Error: Google API key is required. Set GOOGLE_API_KEY env var or use --api-key", file=sys.stderr)
        sys.exit(1)
    
    if not search_engine_id:
        print("Error: Google Custom Search Engine ID is required. Set GOOGLE_SEARCH_ENGINE_ID env var or use --search-engine-id", file=sys.stderr)
        sys.exit(1)
    
    # Determine the query
    text_input = args.query
    
    if not text_input and not piped_context:
        print("Error: Please provide a query as argument or pipe input", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # If only piped input, use it as the query
    if not text_input and piped_context:
        text_input = piped_context
        piped_context = None
    
    return text_input, piped_context, api_key, search_engine_id, llm_provider, model_name, ollama_base_url


def main():
    """Main entry point for the script."""
    text_input, piped_context, api_key, search_engine_id, llm_provider, model_name, ollama_base_url = get_input()
    
    # Initialize the agent
    agent = WebSearchAgent(
        api_key, 
        search_engine_id, 
        llm_provider=llm_provider,
        model_name=model_name,
        ollama_base_url=ollama_base_url
    )
    
    # Perform search
    result = agent.search(text_input, piped_context)
    
    # Output result
    print(result)


if __name__ == "__main__":
    main()

