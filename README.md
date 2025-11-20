# Agentic RAG System with Google Gemini File Search + Mem0 Memory

A sophisticated Retrieval Augmented Generation (RAG) architecture built with Google's Gemini File Search API and Mem0 memory layer. This system features multiple specialized agents working together to provide intelligent document retrieval, personalized responses, and long-term memory.

## Features

### Multi-Agent Architecture with Memory

The system implements a true agentic architecture with specialized agents and persistent memory:

- **FileSearchManager**: Handles file uploads, indexing, and store management
- **MemoryManager**: Manages user memory, preferences, and behavioral learning (🆕 Mem0)
- **QueryAgent**: Analyzes and classifies user queries with memory context
- **RetrievalAgent**: Performs semantic search using Gemini File Search
- **ResponseAgent**: Generates personalized responses with citations and memory
- **AgentOrchestrator**: Coordinates all agents and manages workflow

### Key Capabilities

- **Long-Term Memory (NEW)**: Remembers user preferences, query patterns, and learns from interactions using Mem0
- **Personalized Responses (NEW)**: Adapts answers based on user history and behavior (26% better accuracy)
- **Intelligent Query Classification**: Automatically categorizes queries to optimize retrieval and response generation
- **Semantic Search**: Uses Google's advanced embeddings for accurate document retrieval
- **Citation Support**: Provides source citations for transparency and verification
- **Conversation Context**: Maintains conversation history for contextual responses
- **Flexible Chunking**: Configurable document chunking strategies
- **Metadata Filtering**: Filter searches by custom metadata (author, date, category, etc.)
- **Web Interface**: User-friendly Flask-based web UI
- **Programmatic API**: Use as a Python library in your projects

### Memory Benefits (Powered by Mem0)

- **26% Better Accuracy**: Outperforms standard RAG and OpenAI's memory implementation
- **91% Faster Responses**: Compared to full-context approaches
- **90% Token Reduction**: More efficient than context-heavy alternatives
- **Persistent Learning**: Remembers across sessions and improves over time

## Architecture Overview

```
User Query
    ↓
[AgentOrchestrator]
    ↓
[QueryAgent] ──→ Analyze & Classify Query
    ↓
[ResponseAgent] ──→ [RetrievalAgent] ──→ Gemini File Search
    ↓                                           ↓
Generate Response ←───────── Retrieved Context ─┘
    ↓
Response with Citations
```

## Installation

### Prerequisites

- Python 3.8 or higher
- **Google Gemini API key** ([Get one here](https://ai.google.dev/))
  - **Note**: Only ONE API key needed! The system uses Google Gemini for both RAG (File Search) and Memory (embeddings)
  - **No OpenAI API key required!** Memory uses Google's text-embedding-004 model
- Supabase account for authentication ([Get one here](https://supabase.com))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd up
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add:
# - GEMINI_API_KEY (for both RAG and memory)
# - SUPABASE_URL and SUPABASE_KEY (for authentication)
# - SECRET_KEY (for Flask sessions)
```

## Usage

### Web Interface

1. Start the Flask application:
```bash
export GEMINI_API_KEY='your-api-key-here'
python app.py
```

2. Open your browser to `http://localhost:5000`

3. Upload documents to create a knowledge base

4. Ask questions and get intelligent responses with citations

### Programmatic Usage

```python
from agentic_rag import create_agentic_rag

# Initialize the system
rag = create_agentic_rag(api_key='your-api-key')

# Create a knowledge base
store_id = rag.create_knowledge_base(
    store_name='my_docs',
    file_paths=['doc1.pdf', 'doc2.txt', 'doc3.md'],
    chunking_config={
        'white_space_config': {
            'max_tokens_per_chunk': 500,
            'max_overlap_tokens': 50
        }
    }
)

# Query the system
result = rag.query(
    question="What is the main topic discussed?",
    include_citations=True
)

print(result['text'])
print(f"Query Type: {result['query_type']}")

# Display citations
for citation in result['citations']:
    print(f"Source: {citation['source']}")
    print(f"Snippet: {citation['snippet']}")
```

### Example Script

Run the included example to see the system in action:

```bash
export GEMINI_API_KEY='your-api-key-here'
python example_usage.py
```

## Supported File Types

The system supports 100+ file formats including:

- **Documents**: PDF, DOCX, TXT, MD, RTF, ODT
- **Spreadsheets**: XLSX, CSV
- **Presentations**: PPTX
- **Code**: Python, JavaScript, Java, C++, Go, Rust, TypeScript
- **Data**: JSON, XML, HTML
- **Notebooks**: Jupyter (.ipynb)

## Query Types

The system automatically classifies queries into different types:

- **FACTUAL**: Looking for specific facts or information
- **ANALYTICAL**: Requires analysis or deep understanding
- **COMPARISON**: Comparing multiple concepts or items
- **SUMMARIZATION**: Requesting a summary
- **CREATIVE**: Open-ended or creative questions
- **GENERAL**: General conversation

Each type is processed with optimized prompts and retrieval strategies.

## API Reference

### AgentOrchestrator

Main interface for the agentic RAG system.

#### `create_knowledge_base(store_name, file_paths, chunking_config=None)`

Creates a knowledge base by uploading and indexing files.

**Parameters:**
- `store_name` (str): Name for the knowledge base
- `file_paths` (list): List of file paths to index
- `chunking_config` (dict, optional): Document chunking configuration

**Returns:** Store ID/name

#### `query(question, store_name=None, metadata_filter=None, include_citations=True)`

Process a query through the agentic RAG pipeline.

**Parameters:**
- `question` (str): User's question
- `store_name` (str, optional): Store to search (uses current store if None)
- `metadata_filter` (str, optional): Metadata filter expression
- `include_citations` (bool): Whether to include citations

**Returns:** Dictionary with response text, query type, and citations

#### `clear_conversation()`

Clear conversation history.

#### `get_stats()`

Get system statistics (stores, documents, conversation length).

## Configuration

### Chunking Configuration

Control how documents are split during indexing:

```python
chunking_config = {
    'white_space_config': {
        'max_tokens_per_chunk': 500,  # Maximum tokens per chunk
        'max_overlap_tokens': 50       # Overlap between chunks
    }
}
```

### Metadata Filtering

Add custom metadata to files and filter searches:

```python
# Add metadata during upload
custom_metadata = [
    {"key": "author", "string_value": "John Doe"},
    {"key": "year", "numeric_value": 2024},
    {"key": "category", "string_value": "research"}
]

# Filter queries by metadata
result = rag.query(
    question="What did John Doe write about?",
    metadata_filter="author=JohnDoe"
)
```

## Web Interface Features

- **Drag-and-drop file upload**: Easily upload multiple documents
- **Real-time statistics**: Monitor knowledge bases, documents, and messages
- **Conversation history**: Track queries and responses
- **Citation viewer**: See which documents supported each response
- **Query type badges**: Visual indication of query classification
- **Responsive design**: Works on desktop and mobile

## Pricing

Google Gemini File Search pricing:

- **Indexing**: $0.15 per 1M tokens (one-time)
- **Storage**: Free
- **Query embeddings**: Free
- **Retrieved tokens**: Standard context token rates

## Limitations

- Maximum file size: 100 MB per document
- Storage limits vary by tier (Free: 1 GB, Tier 1: 10 GB, etc.)
- Recommended: Keep stores under 20 GB for optimal performance

## Advanced Features

### Conversation Context

The system maintains conversation history to provide contextual responses:

```python
# First query
result1 = rag.query("What is machine learning?")

# Follow-up query with context
result2 = rag.query("How does it differ from deep learning?")
```

### Multiple Knowledge Bases

Create and manage multiple knowledge bases:

```python
# Create specialized knowledge bases
tech_store = rag.create_knowledge_base('tech_docs', tech_files)
legal_store = rag.create_knowledge_base('legal_docs', legal_files)

# Query specific knowledge base
result = rag.query("...", store_name=tech_store)
```

## Troubleshooting

### "RAG system not initialized"

Make sure your GEMINI_API_KEY is set:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

### "File too large" error

Files must be under 100 MB. Split large files or compress them.

### Slow indexing

Large documents take time to index. The system waits for completion before returning.

## Examples

See `example_usage.py` for a complete working example that demonstrates:
- System initialization
- Knowledge base creation
- Multiple query types
- Citation handling
- Statistics tracking

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided as-is for educational and development purposes.

## Resources

- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs/file-search)
- [File Search Blog Post](https://blog.google/technology/developers/file-search-gemini-api/)
- [Gemini API Pricing](https://ai.google.dev/pricing)

## Architecture Details

### Agent Responsibilities

**FileSearchManager**
- Creates and manages file search stores
- Handles file uploads (direct upload or import)
- Configures chunking strategies
- Tracks indexed documents

**QueryAgent**
- Analyzes query semantics
- Classifies query intent
- Maintains conversation context
- Optimizes query formulation

**RetrievalAgent**
- Performs semantic search
- Builds optimized search queries
- Extracts relevant chunks
- Manages metadata filtering

**ResponseAgent**
- Generates contextual responses
- Synthesizes multiple sources
- Provides citations and sources
- Adapts tone based on query type

**AgentOrchestrator**
- Coordinates agent workflow
- Manages system state
- Routes queries between agents
- Tracks statistics and metrics

## Performance Considerations

- **Indexing**: One-time cost when files are uploaded
- **Query Time**: Fast semantic search (typically < 2 seconds)
- **Caching**: File Search uses internal caching for frequently accessed documents
- **Scalability**: Supports large document collections (recommended < 20 GB per store)

## Security Notes

- Never commit your `.env` file with API keys
- Use environment variables for sensitive configuration
- Implement proper access controls in production
- Validate and sanitize file uploads
- Use HTTPS in production deployments

---

Built with Google Gemini File Search API - Making RAG accessible and powerful.
