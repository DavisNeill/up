"""
Agentic RAG System with Google Gemini File Search
==================================================

This module implements an agentic Retrieval Augmented Generation (RAG) architecture
using Google's Gemini File Search API. The system features multiple specialized agents
that work together to provide intelligent document retrieval and response generation.

Architecture:
- FileSearchManager: Manages file uploads, indexing, and store operations
- QueryAgent: Analyzes and routes user queries
- RetrievalAgent: Performs semantic search using Gemini File Search
- ResponseAgent: Generates contextual responses
- AgentOrchestrator: Coordinates all agents and manages workflow
"""

import os
import time
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Warning: google-genai package not installed. Install with: pip install google-genai")
    genai = None
    types = None


class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    CREATIVE = "creative"
    GENERAL = "general"


@dataclass
class Document:
    """Represents a document in the system"""
    name: str
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_id: Optional[str] = None
    indexed_at: Optional[str] = None


@dataclass
class QueryContext:
    """Context for a user query"""
    query: str
    query_type: QueryType
    metadata_filter: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    retrieved_chunks: List[Dict] = field(default_factory=list)


class FileSearchManager:
    """
    Manages file search stores, file uploads, and indexing operations.
    Handles all interactions with Gemini's File Search API.
    """

    def __init__(self, client: Any):
        self.client = client
        self.stores: Dict[str, Any] = {}
        self.documents: Dict[str, Document] = {}

    def create_store(self, display_name: str) -> str:
        """
        Create a new file search store.

        Args:
            display_name: Name for the store

        Returns:
            Store name/ID
        """
        print(f"[FileSearchManager] Creating store: {display_name}")

        file_search_store = self.client.file_search_stores.create(
            config={'display_name': display_name}
        )

        store_name = file_search_store.name
        self.stores[display_name] = store_name

        print(f"[FileSearchManager] Store created: {store_name}")
        return store_name

    def list_stores(self) -> List[Any]:
        """List all file search stores"""
        stores = self.client.file_search_stores.list()
        return list(stores)

    def upload_and_index(
        self,
        file_path: str,
        store_name: str,
        display_name: Optional[str] = None,
        custom_metadata: Optional[List[Dict]] = None,
        chunking_config: Optional[Dict] = None
    ) -> Document:
        """
        Upload a file and index it in the store.

        Args:
            file_path: Path to the file to upload
            store_name: Name of the store
            display_name: Display name for the file
            custom_metadata: Custom metadata for filtering
            chunking_config: Configuration for document chunking

        Returns:
            Document object
        """
        print(f"[FileSearchManager] Uploading and indexing: {file_path}")

        if not display_name:
            display_name = Path(file_path).name

        # Prepare config
        config = {'display_name': display_name}

        if chunking_config:
            config['chunking_config'] = chunking_config

        # Upload and index
        operation = self.client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_name,
            config=config
        )

        # Wait for completion
        print(f"[FileSearchManager] Waiting for indexing to complete...")
        while not operation.done():
            time.sleep(2)
            operation = self.client.file_search_stores.get_operation(
                operation_name=operation.name
            )

        # Create document record
        doc = Document(
            name=display_name,
            file_path=file_path,
            metadata=custom_metadata or {},
            indexed_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        self.documents[display_name] = doc
        print(f"[FileSearchManager] File indexed successfully: {display_name}")

        return doc

    def import_existing_file(
        self,
        file_name: str,
        store_name: str,
        custom_metadata: Optional[List[Dict]] = None
    ):
        """Import an already uploaded file to the store"""
        print(f"[FileSearchManager] Importing file: {file_name}")

        operation = self.client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=file_name,
            custom_metadata=custom_metadata
        )

        # Wait for completion
        while not operation.done():
            time.sleep(2)

        print(f"[FileSearchManager] File imported successfully")


class QueryAgent:
    """
    Analyzes user queries to determine intent, type, and routing strategy.
    Decides how queries should be processed by other agents.
    """

    def __init__(self, client: Any):
        self.client = client

    def analyze_query(self, query: str, conversation_history: List[Dict] = None) -> QueryContext:
        """
        Analyze a user query to determine its type and processing requirements.

        Args:
            query: User's query string
            conversation_history: Previous conversation context

        Returns:
            QueryContext with analysis results
        """
        print(f"[QueryAgent] Analyzing query: {query[:100]}...")

        # Use Gemini to classify the query type
        analysis_prompt = f"""
        Analyze this user query and classify it into one of these categories:
        - FACTUAL: Looking for specific facts or information
        - ANALYTICAL: Requires analysis or deep understanding
        - COMPARISON: Comparing multiple concepts or items
        - SUMMARIZATION: Requesting a summary
        - CREATIVE: Open-ended or creative question
        - GENERAL: General conversation

        Query: {query}

        Respond with just the category name.
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=analysis_prompt
            )

            category = response.text.strip().upper()

            # Map to QueryType
            query_type = QueryType.GENERAL
            for qt in QueryType:
                if qt.name == category:
                    query_type = qt
                    break

            print(f"[QueryAgent] Query classified as: {query_type.name}")

        except Exception as e:
            print(f"[QueryAgent] Error classifying query: {e}")
            query_type = QueryType.GENERAL

        return QueryContext(
            query=query,
            query_type=query_type,
            conversation_history=conversation_history or []
        )


class RetrievalAgent:
    """
    Performs semantic search using Gemini File Search.
    Retrieves relevant document chunks based on query context.
    """

    def __init__(self, client: Any):
        self.client = client

    def retrieve(
        self,
        query_context: QueryContext,
        store_names: List[str],
        metadata_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents using File Search.

        Args:
            query_context: Context about the query
            store_names: Names of stores to search
            metadata_filter: Optional metadata filter

        Returns:
            List of retrieved document chunks
        """
        print(f"[RetrievalAgent] Retrieving documents for query type: {query_context.query_type.name}")

        # Build the search query with context
        search_query = self._build_search_query(query_context)

        # Configure file search tool
        file_search_config = types.FileSearch(
            file_search_store_names=store_names
        )

        if metadata_filter:
            file_search_config.metadata_filter = metadata_filter

        try:
            # Execute search
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=search_query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(file_search=file_search_config)]
                )
            )

            # Extract grounding metadata
            chunks = []
            if hasattr(response.candidates[0], 'grounding_metadata'):
                grounding_metadata = response.candidates[0].grounding_metadata
                chunks = self._extract_chunks(grounding_metadata)

            query_context.retrieved_chunks = chunks
            print(f"[RetrievalAgent] Retrieved {len(chunks)} relevant chunks")

            return chunks

        except Exception as e:
            print(f"[RetrievalAgent] Error during retrieval: {e}")
            return []

    def _build_search_query(self, query_context: QueryContext) -> str:
        """Build an optimized search query based on context"""
        base_query = query_context.query

        # Enhance query based on type
        if query_context.query_type == QueryType.SUMMARIZATION:
            return f"Provide a comprehensive summary addressing: {base_query}"
        elif query_context.query_type == QueryType.COMPARISON:
            return f"Compare and contrast the following: {base_query}"
        elif query_context.query_type == QueryType.ANALYTICAL:
            return f"Provide detailed analysis of: {base_query}"

        return base_query

    def _extract_chunks(self, grounding_metadata) -> List[Dict]:
        """Extract relevant chunks from grounding metadata"""
        chunks = []

        if hasattr(grounding_metadata, 'grounding_chunks'):
            for chunk in grounding_metadata.grounding_chunks:
                chunks.append({
                    'text': getattr(chunk, 'text', ''),
                    'source': getattr(chunk, 'source', ''),
                })

        return chunks


class ResponseAgent:
    """
    Generates contextual responses using retrieved information.
    Synthesizes information from multiple sources into coherent answers.
    """

    def __init__(self, client: Any):
        self.client = client

    def generate_response(
        self,
        query_context: QueryContext,
        store_names: List[str],
        metadata_filter: Optional[str] = None,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response using File Search and context.

        Args:
            query_context: Query context with retrieved chunks
            store_names: Store names to search
            metadata_filter: Optional metadata filter
            include_citations: Whether to include citation information

        Returns:
            Dictionary with response text and metadata
        """
        print(f"[ResponseAgent] Generating response...")

        # Build enhanced prompt
        prompt = self._build_response_prompt(query_context)

        # Configure file search
        file_search_config = types.FileSearch(
            file_search_store_names=store_names
        )

        if metadata_filter:
            file_search_config.metadata_filter = metadata_filter

        try:
            # Generate response with file search
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(file_search=file_search_config)]
                )
            )

            result = {
                'text': response.text,
                'query_type': query_context.query_type.name,
                'citations': []
            }

            # Extract citations if requested
            if include_citations and hasattr(response.candidates[0], 'grounding_metadata'):
                result['citations'] = self._extract_citations(
                    response.candidates[0].grounding_metadata
                )

            print(f"[ResponseAgent] Response generated successfully")
            return result

        except Exception as e:
            print(f"[ResponseAgent] Error generating response: {e}")
            return {
                'text': f"I encountered an error processing your request: {str(e)}",
                'error': str(e)
            }

    def _build_response_prompt(self, query_context: QueryContext) -> str:
        """Build an enhanced prompt for response generation"""
        prompt_parts = []

        # Add conversation history if available
        if query_context.conversation_history:
            prompt_parts.append("Previous conversation context:")
            for msg in query_context.conversation_history[-3:]:  # Last 3 messages
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
            prompt_parts.append("")

        # Add type-specific instructions
        type_instructions = {
            QueryType.FACTUAL: "Provide accurate, factual information based on the documents.",
            QueryType.ANALYTICAL: "Provide a detailed analysis with insights and reasoning.",
            QueryType.COMPARISON: "Compare and contrast clearly, highlighting key differences and similarities.",
            QueryType.SUMMARIZATION: "Provide a comprehensive yet concise summary.",
            QueryType.CREATIVE: "Provide a thoughtful, creative response while staying grounded in the documents."
        }

        if query_context.query_type in type_instructions:
            prompt_parts.append(type_instructions[query_context.query_type])
            prompt_parts.append("")

        # Add the actual query
        prompt_parts.append(f"Question: {query_context.query}")

        return "\n".join(prompt_parts)

    def _extract_citations(self, grounding_metadata) -> List[Dict]:
        """Extract citation information"""
        citations = []

        if hasattr(grounding_metadata, 'grounding_chunks'):
            for i, chunk in enumerate(grounding_metadata.grounding_chunks):
                citations.append({
                    'index': i + 1,
                    'source': getattr(chunk, 'source', 'Unknown'),
                    'snippet': getattr(chunk, 'text', '')[:200] + '...'
                })

        return citations


class AgentOrchestrator:
    """
    Orchestrates all agents to process user queries end-to-end.
    Manages workflow, agent coordination, and conversation state.
    """

    def __init__(self, api_key: str):
        """
        Initialize the orchestrator with Gemini API key.

        Args:
            api_key: Gemini API key
        """
        if genai is None:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")

        self.client = genai.Client(api_key=api_key)

        # Initialize all agents
        self.file_manager = FileSearchManager(self.client)
        self.query_agent = QueryAgent(self.client)
        self.retrieval_agent = RetrievalAgent(self.client)
        self.response_agent = ResponseAgent(self.client)

        self.conversation_history: List[Dict[str, str]] = []
        self.current_store: Optional[str] = None

        print("[AgentOrchestrator] Initialized with all agents")

    def create_knowledge_base(
        self,
        store_name: str,
        file_paths: List[str],
        chunking_config: Optional[Dict] = None
    ) -> str:
        """
        Create a knowledge base by uploading and indexing files.

        Args:
            store_name: Name for the knowledge base
            file_paths: List of file paths to index
            chunking_config: Optional chunking configuration

        Returns:
            Store name/ID
        """
        print(f"\n[AgentOrchestrator] Creating knowledge base: {store_name}")

        # Create store
        store_id = self.file_manager.create_store(store_name)
        self.current_store = store_id

        # Default chunking config if not provided
        if chunking_config is None:
            chunking_config = {
                'white_space_config': {
                    'max_tokens_per_chunk': 500,
                    'max_overlap_tokens': 50
                }
            }

        # Upload and index each file
        for file_path in file_paths:
            if os.path.exists(file_path):
                self.file_manager.upload_and_index(
                    file_path=file_path,
                    store_name=store_id,
                    chunking_config=chunking_config
                )
            else:
                print(f"[AgentOrchestrator] Warning: File not found: {file_path}")

        print(f"[AgentOrchestrator] Knowledge base created with {len(file_paths)} files")
        return store_id

    def query(
        self,
        question: str,
        store_name: Optional[str] = None,
        metadata_filter: Optional[str] = None,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query through the agentic RAG pipeline.

        Args:
            question: User's question
            store_name: Store to search (uses current_store if None)
            metadata_filter: Optional metadata filter
            include_citations: Whether to include citations

        Returns:
            Response dictionary with answer and metadata
        """
        print(f"\n[AgentOrchestrator] Processing query: {question[:100]}...")

        # Use current store if none specified
        if store_name is None:
            store_name = self.current_store

        if store_name is None:
            return {
                'text': "No knowledge base is currently active. Please create one first.",
                'error': 'No active store'
            }

        # Step 1: Analyze query
        query_context = self.query_agent.analyze_query(
            question,
            self.conversation_history
        )

        # Step 2: Generate response (retrieval happens inside)
        result = self.response_agent.generate_response(
            query_context=query_context,
            store_names=[store_name],
            metadata_filter=metadata_filter,
            include_citations=include_citations
        )

        # Update conversation history
        self.conversation_history.append({'role': 'user', 'content': question})
        self.conversation_history.append({'role': 'assistant', 'content': result['text']})

        print(f"[AgentOrchestrator] Query processed successfully")
        return result

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("[AgentOrchestrator] Conversation history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_stores': len(self.file_manager.stores),
            'total_documents': len(self.file_manager.documents),
            'conversation_length': len(self.conversation_history),
            'current_store': self.current_store
        }


# Convenience function for quick setup
def create_agentic_rag(api_key: Optional[str] = None) -> AgentOrchestrator:
    """
    Create an agentic RAG system.

    Args:
        api_key: Gemini API key (uses GEMINI_API_KEY env var if None)

    Returns:
        AgentOrchestrator instance
    """
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        raise ValueError("API key must be provided or set in GEMINI_API_KEY environment variable")

    return AgentOrchestrator(api_key)


if __name__ == "__main__":
    # Example usage
    print("Agentic RAG System - Example Usage\n")
    print("This module provides an agentic RAG architecture using Gemini File Search.")
    print("\nBasic usage:")
    print("  from agentic_rag import create_agentic_rag")
    print("  rag = create_agentic_rag(api_key='your-api-key')")
    print("  rag.create_knowledge_base('my-docs', ['file1.pdf', 'file2.txt'])")
    print("  result = rag.query('What is...?')")
    print("  print(result['text'])")
