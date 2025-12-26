# DSPy ReAct Web Search Agent

A smart Python script that uses DSPy's ReAct (Reasoning + Acting) agent to perform intelligent web searches using Google Custom Search API.

## Features

- 🤖 **DSPy ReAct Agent**: Uses reasoning and acting capabilities for intelligent search
- 🔍 **Google Custom Search**: Integrates with Google Custom Search API
- 📥 **Piped Input Support**: Can accept context from stdin
- 💬 **Command-line Input**: Accepts queries as command-line arguments
- 🧠 **Context-aware**: Combines piped context with search queries
- 🦙 **Ollama Integration**: Uses local Ollama 3.1 by default via LangChain
- ⚙️ **Configurable LLM**: Switch between Ollama and OpenAI via LangChain

## Prerequisites

1. **Google Custom Search API Setup**:
   - Get a Google API key from [Google Cloud Console](https://console.cloud.google.com/)
   - Create a Custom Search Engine at [Google Custom Search](https://programmablesearchengine.google.com/)
   - Note your Search Engine ID

2. **Ollama Setup** (Default - Recommended):
   - Install [Ollama](https://ollama.ai/)
   - Pull the llama3.1 model: `ollama pull llama3.1`
   - Ensure Ollama is running (default: http://localhost:11434)

3. **OpenAI API Key** (Optional - if using OpenAI instead of Ollama):
   - Get an API key from [OpenAI](https://platform.openai.com/)
   - Set it as an environment variable: `export OPENAI_API_KEY="your-key"`

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set environment variables for Google API credentials and LLM configuration:

```bash
# Required: Google API credentials
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_SEARCH_ENGINE_ID="your-search-engine-id"

# Optional: LLM configuration (defaults to Ollama llama3.1)
export LLM_PROVIDER="ollama"  # or "openai"
export LLM_MODEL="llama3.1"   # or "gpt-3.5-turbo", "gpt-4", etc.
export OLLAMA_BASE_URL="http://localhost:11434"  # Ollama server URL

# Optional: Only needed if using OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

Or pass them as command-line arguments (see Usage below).

## Usage

### Basic Search (Uses Ollama llama3.1 by default)

```bash
python web_search_agent.py "What is the latest news about AI?"
```

### With Piped Context

```bash
# Use piped input as context
echo "Python best practices" | python web_search_agent.py "What does this mean?"

# Or from a file
cat context.txt | python web_search_agent.py "Summarize this information"
```

### Using Different Ollama Models

```bash
# Use a different Ollama model
python web_search_agent.py "Your query" --model "llama3.2"

# Use custom Ollama server URL
python web_search_agent.py "Your query" --ollama-base-url "http://localhost:11434"
```

### Using OpenAI Instead of Ollama

```bash
# Switch to OpenAI provider
python web_search_agent.py "Your query" --llm-provider openai --model "gpt-4"

# Or with API credentials via command-line
python web_search_agent.py "Search query" \
  --api-key "your-google-key" \
  --search-engine-id "your-engine-id" \
  --llm-provider openai \
  --model "gpt-3.5-turbo"
```

## How It Works

1. **Input Processing**: The script accepts input from:
   - Command-line arguments (query)
   - Standard input (piped context)

2. **LLM Configuration**: 
   - Uses LangChain to configure the LLM (Ollama by default)
   - Supports both Ollama (local) and OpenAI (cloud) providers
   - Wraps LangChain LLM for DSPy compatibility using `LangChainDSPyLM` bridge
   - Configures DSPy with the wrapped LM using `dspy.configure(lm=...)`

3. **DSPy ReAct Agent**: Uses DSPy's native `dspy.ReAct` module:
   - **Signature Definition**: `WebSearchSignature` defines input (question, context) and output (answer) fields
   - **Tool Integration**: Google Search is registered as a tool function with proper type annotations and docstrings
   - **Iterative Reasoning**: The ReAct agent performs iterative reasoning loops:
     - **Think**: Analyzes the question and determines what information is needed
     - **Act**: Calls the Google Search tool to retrieve information
     - **Observe**: Processes the search results
     - **Think**: Reasons about the results and determines if more information is needed
     - Repeats until a satisfactory answer is found or `max_iters` is reached (default: 5)
   - **Answer Synthesis**: Synthesizes all gathered information into a comprehensive answer

4. **Google Search Tool**: Performs actual web searches using Google Custom Search API
   - Formatted as a tool function with clear documentation for the ReAct agent
   - Returns structured search results (title, URL, snippet) as formatted strings

5. **Output**: Returns intelligent answers based on iterative reasoning and search results

## Example Workflow

```bash
# Example 1: Simple search with Ollama (default)
python web_search_agent.py "What are the best practices for Python error handling?"

# Example 2: Search with context
echo "The user is working on a machine learning project" | \
  python web_search_agent.py "What libraries should they use?"

# Example 3: Using in a pipeline
cat requirements.txt | python web_search_agent.py "Are there any security vulnerabilities in these packages?"

# Example 4: Using OpenAI instead
python web_search_agent.py "Your query" --llm-provider openai --model "gpt-4"

# Example 5: Custom Ollama model
python web_search_agent.py "Your query" --model "llama3.2" --ollama-base-url "http://localhost:11434"
```

## Environment Variables

### Required
- `GOOGLE_API_KEY`: Your Google API key
- `GOOGLE_SEARCH_ENGINE_ID`: Your Custom Search Engine ID

### Optional (LLM Configuration)
- `LLM_PROVIDER`: LLM provider to use - `"ollama"` (default) or `"openai"`
- `LLM_MODEL`: Model name - `"llama3.1"` (default for Ollama) or `"gpt-3.5-turbo"` (default for OpenAI)
- `OLLAMA_BASE_URL`: Ollama server URL - `"http://localhost:11434"` (default)
- `OPENAI_API_KEY`: Your OpenAI API key (required only if using OpenAI provider)

## Technical Details: DSPy ReAct Configuration

### Architecture Overview

The implementation uses DSPy's native `ReAct` module with the following components:

1. **LangChainDSPyLM Wrapper**: Bridges LangChain LLMs (Ollama/OpenAI) to DSPy's LM interface
   - Implements `__call__()` and `request()` methods required by DSPy
   - Handles different response types from LangChain LLMs
   - Enables DSPy to work with any LangChain-compatible LLM

2. **WebSearchSignature**: DSPy signature defining the task structure
   ```python
   class WebSearchSignature(dspy.Signature):
       question = dspy.InputField(desc="The question or query to answer")
       context = dspy.InputField(desc="Optional context information", default="")
       answer = dspy.OutputField(desc="The comprehensive answer based on web search results")
   ```

3. **Google Search Tool Function**: Registered tool for the ReAct agent
   - Properly annotated with type hints (`query: str`, `num_results: int = 5`)
   - Includes comprehensive docstring for the agent to understand its purpose
   - Returns formatted string of search results

4. **ReAct Agent Initialization**:
   ```python
   self.agent = dspy.ReAct(
       signature=WebSearchSignature,
       tools=[google_search],
       max_iters=5  # Configurable reasoning iterations
   )
   ```

### ReAct Reasoning Loop

The native DSPy ReAct agent performs iterative reasoning:

1. **Initial Thought**: Agent analyzes the question and context
2. **Action Selection**: Decides to call `google_search` tool
3. **Tool Execution**: Performs web search and receives results
4. **Observation**: Processes the search results
5. **Reasoning**: Determines if the answer is complete or if more searches are needed
6. **Iteration**: Repeats steps 2-5 until satisfied or `max_iters` reached
7. **Final Answer**: Synthesizes all information into a comprehensive answer

This iterative approach allows the agent to:
- Perform multiple searches if needed
- Refine queries based on initial results
- Synthesize information from multiple sources
- Provide more accurate and complete answers

## Troubleshooting

1. **"DSPy is not installed"**: Run `pip install dspy-ai`
2. **"Google API client is not installed"**: Run `pip install google-api-python-client`
3. **"LangChain is not installed"**: Run `pip install langchain langchain-ollama langchain-openai langchain-core`
4. **"Ollama deprecation warning"**: Install the new package: `pip install langchain-ollama` (this replaces the deprecated `langchain-community.llms.Ollama`)
5. **Ollama connection errors**: 
   - Ensure Ollama is running: `ollama serve`
   - Verify the model is installed: `ollama list`
   - Check the base URL matches your Ollama server
6. **API Key errors**: Verify your API keys are set correctly
7. **Search Engine ID errors**: Make sure your Custom Search Engine is properly configured
8. **Model not found**: For Ollama, ensure the model is pulled: `ollama pull llama3.1`
9. **DSPy ReAct initialization errors**: 
   - Check that DSPy is properly configured with `dspy.configure(lm=...)`
   - Verify the tool function has proper type annotations and docstrings
   - Ensure the signature fields match the tool inputs/outputs

## License

MIT License - feel free to use and modify as needed.

