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

Configuration can be set via **config file**, **environment variables**, or **command-line arguments** with the following priority:
1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Config file** (`config.yaml` or `~/.web_search_agent/config.yaml`)
4. **Defaults** (lowest priority)

### Config File Setup

1. Copy the example config file:
   ```bash
   cp config.yaml.example config.yaml
   ```

2. Edit `config.yaml` with your settings:
   ```yaml
   google_api_key: "your-google-api-key"
   google_search_engine_id: "your-search-engine-id"
   llm_provider: "ollama"
   llm_model: "llama3.1"
   # ... see config.yaml.example for all options
   ```

### Environment Variables

```bash
# Required: Google API credentials
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_SEARCH_ENGINE_ID="your-search-engine-id"

# Optional: LLM configuration (defaults to Ollama llama3.1)
export LLM_PROVIDER="ollama"  # or "openai"
export LLM_MODEL="llama3.1"   # or "gpt-3.5-turbo", "gpt-4", etc.
export OLLAMA_BASE_URL="http://localhost:11434"  # Ollama server URL
export OPENAI_API_KEY="your-openai-api-key"  # Only needed if using OpenAI

# Optional: ReAct and Search configuration
export REACT_MAX_ITERS="5"           # Max reasoning iterations
export SEARCH_NUM_RESULTS="5"        # Number of search results (max: 10)

# Optional: LLM generation parameters
export LLM_TEMPERATURE="0.0"         # Temperature (0.0-2.0)
export LLM_MAX_TOKENS="1000"         # Max tokens per response
export LLM_CACHE="true"              # Enable caching (true/false)
```

### All Configurable Parameters

| Parameter | Config Key | Env Var | CLI Arg | Default | Description |
|-----------|-----------|---------|---------|---------|-------------|
| Google API Key | `google_api_key` | `GOOGLE_API_KEY` | `--api-key` | *Required* | Google Custom Search API key |
| Search Engine ID | `google_search_engine_id` | `GOOGLE_SEARCH_ENGINE_ID` | `--search-engine-id` | *Required* | Google Custom Search Engine ID |
| LLM Provider | `llm_provider` | `LLM_PROVIDER` | `--llm-provider` | `ollama` | `ollama` or `openai` |
| LLM Model | `llm_model` | `LLM_MODEL` | `--model` | `llama3.1` (Ollama) or `gpt-3.5-turbo` (OpenAI) | Model name |
| Ollama Base URL | `ollama_base_url` | `OLLAMA_BASE_URL` | `--ollama-base-url` | `http://localhost:11434` | Ollama server URL |
| OpenAI API Key | `openai_api_key` | `OPENAI_API_KEY` | `--openai-api-key` | *Required if using OpenAI* | OpenAI API key |
| Max Iterations | `react_max_iters` | `REACT_MAX_ITERS` | `--max-iters` | `5` | Max ReAct reasoning iterations |
| Search Results | `search_num_results` | `SEARCH_NUM_RESULTS` | `--search-num-results` | `5` | Number of search results (max: 10) |
| Temperature | `llm_temperature` | `LLM_TEMPERATURE` | `--temperature` | `0.0` | LLM temperature (0.0-2.0) |
| Max Tokens | `llm_max_tokens` | `LLM_MAX_TOKENS` | `--max-tokens` | `1000` | Max tokens per response |
| Cache | `llm_cache` | `LLM_CACHE` | `--no-cache` | `true` | Enable LLM response caching |
| Config File | `config` | `CONFIG_FILE` | `--config` | `config.yaml` | Path to config file |

## Usage

### Basic Search (Uses config file or defaults)

```bash
# Using config file (recommended)
python web_search_agent.py "What is the latest news about AI?"

# Using environment variables
export GOOGLE_API_KEY="your-key"
export GOOGLE_SEARCH_ENGINE_ID="your-id"
python web_search_agent.py "What is the latest news about AI?"

# Using command-line arguments
python web_search_agent.py "What is the latest news about AI?" \
  --api-key "your-google-key" \
  --search-engine-id "your-engine-id"
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
# Via config file: set llm_model: "llama3.2" in config.yaml
python web_search_agent.py "Your query"

# Via environment variable
export LLM_MODEL="llama3.2"
python web_search_agent.py "Your query"

# Via command-line
python web_search_agent.py "Your query" --model "llama3.2"

# Custom Ollama server URL
python web_search_agent.py "Your query" \
  --ollama-base-url "http://localhost:11434" \
  --model "llama3.2"
```

### Using OpenAI Instead of Ollama

```bash
# Via config file: set llm_provider: "openai" and openai_api_key in config.yaml
python web_search_agent.py "Your query"

# Via environment variables
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-key"
python web_search_agent.py "Your query"

# Via command-line
python web_search_agent.py "Your query" \
  --llm-provider openai \
  --openai-api-key "your-openai-key" \
  --model "gpt-4"
```

### Advanced Configuration Examples

```bash
# Custom ReAct iterations and search results
python web_search_agent.py "Complex query" \
  --max-iters 10 \
  --search-num-results 8

# Custom LLM parameters
python web_search_agent.py "Creative query" \
  --temperature 0.7 \
  --max-tokens 2000

# Disable caching
python web_search_agent.py "Query" --no-cache

# Use custom config file
python web_search_agent.py "Query" --config /path/to/custom-config.yaml

# Full example with all parameters
python web_search_agent.py "What are the latest AI developments?" \
  --api-key "your-google-key" \
  --search-engine-id "your-engine-id" \
  --llm-provider openai \
  --openai-api-key "your-openai-key" \
  --model "gpt-4" \
  --max-iters 7 \
  --search-num-results 6 \
  --temperature 0.3 \
  --max-tokens 1500
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
# Example 1: Simple search with config file (recommended)
# Set up config.yaml first, then:
python web_search_agent.py "What are the best practices for Python error handling?"

# Example 2: Search with context
echo "The user is working on a machine learning project" | \
  python web_search_agent.py "What libraries should they use?"

# Example 3: Using in a pipeline
cat requirements.txt | python web_search_agent.py "Are there any security vulnerabilities in these packages?"

# Example 4: Using OpenAI with custom parameters
python web_search_agent.py "Your query" \
  --llm-provider openai \
  --openai-api-key "your-key" \
  --model "gpt-4" \
  --max-iters 7 \
  --temperature 0.3

# Example 5: Custom Ollama model with advanced settings
python web_search_agent.py "Your query" \
  --model "llama3.2" \
  --ollama-base-url "http://localhost:11434" \
  --max-iters 10 \
  --search-num-results 8 \
  --max-tokens 2000

# Example 6: Using environment variables
export GOOGLE_API_KEY="your-key"
export GOOGLE_SEARCH_ENGINE_ID="your-id"
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-key"
export LLM_MODEL="gpt-4"
python web_search_agent.py "Your query"
```

## Environment Variables Reference

### Required
- `GOOGLE_API_KEY`: Your Google API key
- `GOOGLE_SEARCH_ENGINE_ID`: Your Custom Search Engine ID

### Optional - LLM Configuration
- `LLM_PROVIDER`: LLM provider to use - `"ollama"` (default) or `"openai"`
- `LLM_MODEL`: Model name - `"llama3.1"` (default for Ollama) or `"gpt-3.5-turbo"` (default for OpenAI)
- `OLLAMA_BASE_URL`: Ollama server URL - `"http://localhost:11434"` (default)
- `OPENAI_API_KEY`: Your OpenAI API key (required only if using OpenAI provider)

### Optional - ReAct Agent Configuration
- `REACT_MAX_ITERS`: Maximum number of reasoning iterations (default: `5`)

### Optional - Search Configuration
- `SEARCH_NUM_RESULTS`: Number of search results to return (default: `5`, max: `10`)

### Optional - LLM Generation Parameters
- `LLM_TEMPERATURE`: Temperature for LLM generation (default: `0.0`, range: `0.0-2.0`)
- `LLM_MAX_TOKENS`: Maximum tokens for LLM response (default: `1000`)
- `LLM_CACHE`: Enable LLM response caching (default: `true`, values: `true`/`false`)

### Optional - Config File
- `CONFIG_FILE`: Path to custom config file (default: `config.yaml` or `~/.web_search_agent/config.yaml`)

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

