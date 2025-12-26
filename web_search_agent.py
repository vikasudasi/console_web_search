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
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Please install it with: pip install pyyaml")
    sys.exit(1)

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


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file, with environment variable support.
    
    Priority order: Environment variables > Config file > Defaults
    
    Args:
        config_path: Path to config file. If None, looks for:
                    1. CONFIG_FILE env var
                    2. config.yaml in current directory
                    3. ~/.web_search_agent/config.yaml
    
    Returns:
        Dictionary with configuration values
    """
    config = {}
    
    # Default config file locations
    if config_path is None:
        config_path = os.getenv("CONFIG_FILE")
        if config_path is None:
            # Try current directory
            if Path("config.yaml").exists():
                config_path = "config.yaml"
            # Try home directory
            elif Path.home().joinpath(".web_search_agent", "config.yaml").exists():
                config_path = str(Path.home().joinpath(".web_search_agent", "config.yaml"))
    
    # Load from file if it exists
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}", file=sys.stderr)
    
    # Override with environment variables (higher priority)
    env_mappings = {
        'GOOGLE_API_KEY': 'google_api_key',
        'GOOGLE_SEARCH_ENGINE_ID': 'google_search_engine_id',
        'LLM_PROVIDER': 'llm_provider',
        'LLM_MODEL': 'llm_model',
        'OLLAMA_BASE_URL': 'ollama_base_url',
        'OPENAI_API_KEY': 'openai_api_key',
        'REACT_MAX_ITERS': 'react_max_iters',
        'SEARCH_NUM_RESULTS': 'search_num_results',
        'LLM_TEMPERATURE': 'llm_temperature',
        'LLM_MAX_TOKENS': 'llm_max_tokens',
        'LLM_CACHE': 'llm_cache',
    }
    
    for env_var, config_key in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Convert string booleans and numbers
            if env_value.lower() in ('true', '1', 'yes'):
                config[config_key] = True
            elif env_value.lower() in ('false', '0', 'no'):
                config[config_key] = False
            elif env_value.isdigit():
                config[config_key] = int(env_value)
            elif env_value.replace('.', '', 1).isdigit():
                config[config_key] = float(env_value)
            else:
                config[config_key] = env_value
    
    return config


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


def create_llm(provider: str = "ollama", model_name: str = "llama3.1", base_url: Optional[str] = None, openai_api_key: Optional[str] = None) -> BaseLanguageModel:
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
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider. Set it in config file, env var, or pass as parameter")
        print(f"Using OpenAI with model '{model_name}'", file=sys.stderr)
        return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: 'ollama', 'openai'")


class LangChainDSPyLM(dspy.LM):
    """
    Wrapper to make LangChain LLMs compatible with DSPy's LM interface.
    This allows DSPy to work with LangChain LLMs (Ollama, OpenAI, etc.).
    Inherits from dspy.LM to satisfy isinstance checks.
    """
    
    def __init__(self, langchain_llm, model_name: str = "langchain", **kwargs):
        """
        Initialize the DSPy-LangChain bridge.
        
        Args:
            langchain_llm: LangChain LLM instance (OllamaLLM, ChatOpenAI, etc.)
            model_name: Name identifier for the model
            **kwargs: Additional arguments passed to dspy.LM (temperature, max_tokens, etc.)
        """
        # Initialize parent class with required parameters
        # dspy.LM expects: model, model_type='chat', temperature=0.0, max_tokens=1000, cache=True
        super().__init__(
            model=model_name,
            model_type='chat',
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 1000),
            cache=kwargs.get('cache', True),
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens', 'cache']}
        )
        
        self.langchain_llm = langchain_llm
        self.name = model_name
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Call the LangChain LLM with a prompt or messages.
        Matches BaseLM's abstract method signature and returns outputs in DSPy format.
        """
        # Build messages from prompt if needed (matching DSPy's LM format)
        if messages is None:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            messages = [{"role": "user", "content": str(prompt)}]
        
        # Convert messages to a single prompt string for LangChain
        # (LangChain LLMs typically work with strings or message objects)
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        prompt_str = "\n".join(prompt_parts) if prompt_parts else str(prompt) if prompt else ""
        
        # Get response from LangChain
        outputs = self._invoke_langchain(prompt_str)
        
        # Update history (matching DSPy's LM behavior)
        if not dspy.settings.disable_history:
            from datetime import datetime
            import uuid
            entry = dict(
                prompt=prompt,
                messages=messages,
                kwargs=kwargs,
                outputs=outputs,
                timestamp=datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                model=self.model,
                model_type=self.model_type,
            )
            self.history.append(entry)
            self.update_global_history(entry)
        
        # Return outputs in DSPy format (list of strings)
        return outputs
    
    def request(self, prompt, **kwargs):
        """Request method for DSPy compatibility."""
        prompt_str = str(prompt) if hasattr(prompt, '__str__') else prompt
        return self._invoke_langchain(prompt_str)
    
    def generate(self, prompt, **kwargs):
        """Generate method for DSPy compatibility."""
        return self.request(prompt, **kwargs)
    
    def __repr__(self):
        """String representation for debugging."""
        return f"LangChainDSPyLM(model={self.model})"
    
    def _invoke_langchain(self, prompt: str) -> list:
        """
        Invoke the LangChain LLM and return the response in DSPy format.
        DSPy expects a list of output strings (like LiteLLM's response format).
        """
        try:
            response = self.langchain_llm.invoke(prompt)
            # Handle different response types and convert to list format
            if isinstance(response, str):
                text = response
            elif hasattr(response, 'content'):
                text = response.content
            elif hasattr(response, 'text'):
                text = response.text
            else:
                text = str(response)
            
            # DSPy expects a list of strings (outputs), similar to LiteLLM's choices format
            # Return as a list to match DSPy's expected format
            return [text]
        except Exception as e:
            print(f"Error invoking LangChain LLM: {e}", file=sys.stderr)
            raise


class WebSearchSignature(dspy.Signature):
    """Signature for web search question-answering task."""
    question = dspy.InputField(desc="The question or query to answer")
    context = dspy.InputField(desc="Optional context information", default="")
    answer = dspy.OutputField(desc="The comprehensive answer based on web search results")


class WebSearchAgent:
    """DSPy ReAct Agent for web search tasks using native DSPy ReAct module."""
    
    def __init__(
        self, 
        api_key: str, 
        search_engine_id: str, 
        llm_provider: str = "ollama",
        model_name: str = "llama3.1",
        ollama_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        max_iters: int = 5,
        search_num_results: int = 5,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 1000,
        llm_cache: bool = True
    ):
        """
        Initialize the Web Search Agent with DSPy's native ReAct module.
        
        Args:
            api_key: Google API key
            search_engine_id: Custom Search Engine ID
            llm_provider: LLM provider ("ollama" or "openai")
            model_name: Model name (e.g., "llama3.1", "gpt-3.5-turbo")
            ollama_base_url: Base URL for Ollama (default: http://localhost:11434)
            openai_api_key: OpenAI API key (required if using OpenAI provider)
            max_iters: Maximum number of reasoning iterations for ReAct agent
            search_num_results: Number of search results to return (default: 5, max: 10)
            llm_temperature: LLM temperature for generation (default: 0.0)
            llm_max_tokens: Maximum tokens for LLM response (default: 1000)
            llm_cache: Whether to cache LLM responses (default: True)
        """
        self.search_num_results = search_num_results
        # Initialize Google Search Tool
        self.search_tool = GoogleSearchTool(api_key, search_engine_id)
        
        # Create LangChain LLM and configure DSPy
        self.langchain_llm = None
        self.dspy_lm = None
        self.agent = None
        
        try:
            # Create LangChain LLM
            self.langchain_llm = create_llm(llm_provider, model_name, ollama_base_url, openai_api_key)
            
            # Create DSPy-compatible LM wrapper with configurable parameters
            self.dspy_lm = LangChainDSPyLM(
                self.langchain_llm, 
                model_name=f"{llm_provider}/{model_name}",
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                cache=llm_cache
            )
            
            # Configure DSPy with the LM
            dspy.configure(lm=self.dspy_lm)
            
            # Verify the LM is properly configured
            try:
                # Test that DSPy recognizes the LM
                current_lm = dspy.settings.lm
                if current_lm is None:
                    raise ValueError("DSPy LM configuration failed - LM is None")
                print(f"Successfully configured DSPy with {llm_provider} ({model_name})", file=sys.stderr)
            except AttributeError:
                # DSPy might use a different attribute name
                print(f"Configured DSPy with {llm_provider} ({model_name})", file=sys.stderr)
                print("Note: Could not verify LM configuration, but proceeding...", file=sys.stderr)
            
            # Create the Google search tool function for DSPy
            def google_search(query: str, num_results: Optional[int] = None) -> str:
                """
                Search the web using Google Custom Search API.
                
                Args:
                    query: The search query string
                    num_results: Number of results to return (default: from config, max: 10)
                
                Returns:
                    Formatted string containing search results with titles, URLs, and snippets
                """
                if num_results is None:
                    num_results = self.search_num_results
                results = self.search_tool.search(query, num_results)
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
            
            # Initialize DSPy's native ReAct agent
            try:
                # Ensure LM is configured before creating ReAct agent
                # ReAct should use the globally configured LM
                self.agent = dspy.ReAct(
                    signature=WebSearchSignature,
                    tools=[google_search],
                    max_iters=max_iters
                )
                print(f"Initialized DSPy ReAct agent with max_iters={max_iters}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not initialize DSPy ReAct agent: {e}", file=sys.stderr)
                print("Falling back to direct search mode", file=sys.stderr)
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error setting up DSPy ReAct agent: {e}", file=sys.stderr)
            print("Falling back to direct search mode", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    def search(self, question: str, context: Optional[str] = None) -> str:
        """
        Perform a web search and generate an answer using DSPy's native ReAct agent.
        
        Args:
            question: The question or search query
            context: Optional context from piped input
        
        Returns:
            Answer based on search results
        """
        try:
            # Use DSPy's native ReAct agent
            if self.agent and self.dspy_lm:
                # Ensure LM is configured (reconfigure if needed)
                dspy.configure(lm=self.dspy_lm)
                
                # Try using dspy.context if available (thread-safe)
                try:
                    context_manager = dspy.context(lm=self.dspy_lm)
                    with context_manager:
                        result = self.agent(question=question, context=context or "")
                except AttributeError:
                    # dspy.context might not be available in all versions
                    # Just use the globally configured LM
                    result = self.agent(question=question, context=context or "")
                
                # Extract the answer from the result
                if hasattr(result, 'answer'):
                    return result.answer
                elif isinstance(result, str):
                    return result
                else:
                    # Try to get answer from result object
                    return str(result)
            else:
                # Fallback: direct search with basic reasoning
                return self._direct_search_with_reasoning(question, context)
        except Exception as e:
            print(f"Error in DSPy ReAct agent execution: {e}", file=sys.stderr)
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


def get_input() -> Dict:
    """
    Get input from command-line arguments, environment variables, or config file.
    Priority: CLI arguments > Environment variables > Config file > Defaults
    
    Returns:
        Dictionary with all configuration values
    """
    piped_context = None
    
    # Check if there's piped input
    if not sys.stdin.isatty():
        piped_context = sys.stdin.read().strip()
    
    # Load config file first (lowest priority)
    config = load_config()
    
    # Get command-line arguments
    parser = argparse.ArgumentParser(
        description="DSPy ReAct Agent for Web Search using Google Custom Search API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python web_search_agent.py "What is the latest news about AI?"
  echo "Python best practices" | python web_search_agent.py "What does this mean?"
  cat context.txt | python web_search_agent.py "Summarize this"
  python web_search_agent.py "Query" --config config.yaml
        """
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query or question (optional if piped input is provided)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (YAML format). Default: config.yaml or ~/.web_search_agent/config.yaml"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google API key (overrides config file and env var)"
    )
    parser.add_argument(
        "--search-engine-id",
        default=None,
        help="Google Custom Search Engine ID (overrides config file and env var)"
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["ollama", "openai"],
        help="LLM provider: 'ollama' or 'openai' (overrides config file and env var)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (overrides config file and env var)"
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama base URL (overrides config file and env var)"
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (overrides config file and env var)"
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Maximum ReAct reasoning iterations (overrides config file and env var)"
    )
    parser.add_argument(
        "--search-num-results",
        type=int,
        default=None,
        help="Number of search results to return (overrides config file and env var)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature (overrides config file and env var)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for LLM response (overrides config file and env var)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching (overrides config file and env var)"
    )
    
    args = parser.parse_args()
    
    # Reload config if custom path provided
    if args.config:
        config = load_config(args.config)
    
    # Apply CLI arguments (highest priority)
    if args.api_key:
        config['google_api_key'] = args.api_key
    if args.search_engine_id:
        config['google_search_engine_id'] = args.search_engine_id
    if args.llm_provider:
        config['llm_provider'] = args.llm_provider
    if args.model:
        config['llm_model'] = args.model
    if args.ollama_base_url:
        config['ollama_base_url'] = args.ollama_base_url
    if args.openai_api_key:
        config['openai_api_key'] = args.openai_api_key
    if args.max_iters is not None:
        config['react_max_iters'] = args.max_iters
    if args.search_num_results is not None:
        config['search_num_results'] = args.search_num_results
    if args.temperature is not None:
        config['llm_temperature'] = args.temperature
    if args.max_tokens is not None:
        config['llm_max_tokens'] = args.max_tokens
    if args.no_cache:
        config['llm_cache'] = False
    
    # Set defaults for missing values
    llm_provider = config.get('llm_provider', 'ollama')
    model_name = config.get('llm_model')
    if not model_name:
        model_name = "llama3.1" if llm_provider == "ollama" else "gpt-3.5-turbo"
        config['llm_model'] = model_name
    
    # Get required values
    api_key = config.get('google_api_key')
    search_engine_id = config.get('google_search_engine_id')
    
    if not api_key:
        print("Error: Google API key is required. Set in config file, GOOGLE_API_KEY env var, or use --api-key", file=sys.stderr)
        sys.exit(1)
    
    if not search_engine_id:
        print("Error: Google Custom Search Engine ID is required. Set in config file, GOOGLE_SEARCH_ENGINE_ID env var, or use --search-engine-id", file=sys.stderr)
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
    
    # Build final config dict
    final_config = {
        'query': text_input,
        'piped_context': piped_context,
        'google_api_key': api_key,
        'google_search_engine_id': search_engine_id,
        'llm_provider': llm_provider,
        'llm_model': model_name,
        'ollama_base_url': config.get('ollama_base_url'),
        'openai_api_key': config.get('openai_api_key'),
        'react_max_iters': config.get('react_max_iters', 5),
        'search_num_results': config.get('search_num_results', 5),
        'llm_temperature': config.get('llm_temperature', 0.0),
        'llm_max_tokens': config.get('llm_max_tokens', 1000),
        'llm_cache': config.get('llm_cache', True),
    }
    
    return final_config


def main():
    """Main entry point for the script."""
    config = get_input()
    
    # Initialize the agent with all configurable parameters
    agent = WebSearchAgent(
        api_key=config['google_api_key'],
        search_engine_id=config['google_search_engine_id'],
        llm_provider=config['llm_provider'],
        model_name=config['llm_model'],
        ollama_base_url=config.get('ollama_base_url'),
        openai_api_key=config.get('openai_api_key'),
        max_iters=config['react_max_iters'],
        search_num_results=config['search_num_results'],
        llm_temperature=config['llm_temperature'],
        llm_max_tokens=config['llm_max_tokens'],
        llm_cache=config['llm_cache']
    )
    
    # Perform search
    result = agent.search(config['query'], config.get('piped_context'))
    
    # Output result
    print(result)


if __name__ == "__main__":
    main()

