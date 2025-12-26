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
   - Wraps LangChain LLM for DSPy compatibility

3. **ReAct Agent**: DSPy's ReAct agent:
   - **Reasons** about the query and context using the configured LLM
   - **Acts** by calling the Google Search tool
   - Synthesizes results into a coherent answer

4. **Google Search**: Performs actual web searches using Google Custom Search API

5. **Output**: Returns intelligent answers based on search results

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

## Troubleshooting

1. **"DSPy is not installed"**: Run `pip install dspy-ai`
2. **"Google API client is not installed"**: Run `pip install google-api-python-client`
3. **"LangChain is not installed"**: Run `pip install langchain langchain-ollama langchain-openai langchain-core`
4. **"Ollama deprecation warning"**: Install the new package: `pip install langchain-ollama` (this replaces the deprecated `langchain-community.llms.Ollama`)
4. **Ollama connection errors**: 
   - Ensure Ollama is running: `ollama serve`
   - Verify the model is installed: `ollama list`
   - Check the base URL matches your Ollama server
5. **API Key errors**: Verify your API keys are set correctly
6. **Search Engine ID errors**: Make sure your Custom Search Engine is properly configured
7. **Model not found**: For Ollama, ensure the model is pulled: `ollama pull llama3.1`

## License

MIT License - feel free to use and modify as needed.

