"""
Example Usage of the Agentic RAG System
========================================

This script demonstrates how to use the agentic RAG system programmatically
without the web interface.
"""

import os
from agentic_rag import create_agentic_rag

def main():
    # Set your API key (or use environment variable)
    api_key = os.environ.get('GEMINI_API_KEY', 'your-api-key-here')

    print("="*60)
    print("Agentic RAG System - Example Usage")
    print("="*60 + "\n")

    # Step 1: Create the RAG system
    print("Step 1: Initializing Agentic RAG system...")
    try:
        rag = create_agentic_rag(api_key=api_key)
        print("✓ System initialized successfully\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nPlease set GEMINI_API_KEY environment variable:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        return

    # Step 2: Create sample documents
    print("Step 2: Creating sample documents...")
    create_sample_documents()
    print("✓ Sample documents created\n")

    # Step 3: Create knowledge base
    print("Step 3: Creating knowledge base from documents...")
    store_name = rag.create_knowledge_base(
        store_name='example_knowledge_base',
        file_paths=[
            'sample_docs/ai_overview.txt',
            'sample_docs/machine_learning.txt',
            'sample_docs/rag_systems.txt'
        ],
        chunking_config={
            'white_space_config': {
                'max_tokens_per_chunk': 500,
                'max_overlap_tokens': 50
            }
        }
    )
    print(f"✓ Knowledge base created: {store_name}\n")

    # Step 4: Query the system
    print("Step 4: Asking questions...\n")

    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the benefits of RAG systems?",
        "Compare supervised and unsupervised learning"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)

        result = rag.query(
            question=question,
            include_citations=True
        )

        print(f"Query Type: {result['query_type']}")
        print(f"\nAnswer:\n{result['text']}")

        if result.get('citations'):
            print(f"\nCitations: {len(result['citations'])} sources")
            for citation in result['citations'][:2]:  # Show first 2
                print(f"  - {citation['source']}")

        print()

    # Step 5: Show statistics
    print("\n" + "="*60)
    print("System Statistics")
    print("="*60)
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Step 6: Clear conversation
    print("\nClearing conversation history...")
    rag.clear_conversation()
    print("✓ Conversation cleared")

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


def create_sample_documents():
    """Create sample documents for demonstration"""
    import os

    os.makedirs('sample_docs', exist_ok=True)

    # AI Overview document
    with open('sample_docs/ai_overview.txt', 'w') as f:
        f.write("""
Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence processes by
machines, especially computer systems. These processes include learning (the acquisition
of information and rules for using the information), reasoning (using rules to reach
approximate or definite conclusions) and self-correction.

AI applications include expert systems, natural language processing, speech recognition
and machine vision. AI can be categorized as either weak or strong. Weak AI, also known
as narrow AI, is an AI system that is designed and trained for a particular task. Virtual
personal assistants, such as Apple's Siri, are a form of weak AI.

Strong AI, also known as artificial general intelligence, is an AI system with generalized
human cognitive abilities. When presented with an unfamiliar task, a strong AI system is
able to find a solution without human intervention.

The field of AI research was founded at a workshop held at Dartmouth College in 1956.
AI winter refers to periods when funding for AI research decreased, which occurred in the
1970s and again in the late 1980s.
""")

    # Machine Learning document
    with open('sample_docs/machine_learning.txt', 'w') as f:
        f.write("""
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that provides systems the ability
to automatically learn and improve from experience without being explicitly programmed.
Machine learning focuses on the development of computer programs that can access data and
use it to learn for themselves.

Types of Machine Learning:

1. Supervised Learning: The algorithm learns from labeled training data, helping to predict
outcomes for unforeseen data. Examples include classification and regression problems.

2. Unsupervised Learning: The algorithm learns from unlabeled data, finding hidden patterns
or intrinsic structures in the input data. Examples include clustering and association.

3. Reinforcement Learning: The algorithm learns through trial and error, using feedback from
its own actions and experiences. It's commonly used in robotics, gaming, and navigation.

4. Semi-supervised Learning: This approach uses both labeled and unlabeled data for training,
typically a small amount of labeled data with a large amount of unlabeled data.

Deep learning is a subset of machine learning based on artificial neural networks. The depth
refers to the number of layers in the network. Deep learning has driven recent advances in
computer vision, natural language processing, and speech recognition.
""")

    # RAG Systems document
    with open('sample_docs/rag_systems.txt', 'w') as f:
        f.write("""
Retrieval Augmented Generation (RAG) Systems

Retrieval Augmented Generation (RAG) is an AI framework that combines the strengths of
retrieval-based and generation-based approaches to create more accurate and contextual
responses. RAG systems enhance large language models by grounding their responses in
external knowledge sources.

How RAG Works:

1. Retrieval Phase: When a query is received, the system searches through a knowledge base
or document collection to find relevant information. This is typically done using semantic
search with embeddings.

2. Augmentation Phase: The retrieved documents or passages are combined with the original
query to create an enhanced context.

3. Generation Phase: A large language model generates a response based on both the query
and the retrieved context, producing more accurate and factual answers.

Benefits of RAG Systems:

- Improved Accuracy: Responses are grounded in actual documents, reducing hallucinations.
- Up-to-date Information: Knowledge base can be updated without retraining the model.
- Transparency: Sources can be cited, making the system more trustworthy.
- Domain Specificity: Can be tailored to specific domains by using specialized documents.
- Cost Effective: More efficient than fine-tuning large models for specific tasks.

Popular RAG implementations use vector databases like Pinecone, Weaviate, or Chroma for
efficient semantic search. Google's Gemini API now offers built-in File Search capabilities,
making RAG implementation even more accessible.
""")


if __name__ == "__main__":
    main()
