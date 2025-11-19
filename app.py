"""
Flask Web Interface for Agentic RAG System
==========================================

Web application that provides a user-friendly interface to the agentic RAG system.
Supports file uploads, knowledge base creation, and interactive querying.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path

from agentic_rag import create_agentic_rag, AgentOrchestrator

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'doc', 'docx', 'xlsx', 'pptx', 'csv', 'json', 'md',
    'py', 'js', 'java', 'cpp', 'html', 'xml', 'ipynb'
}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Global RAG orchestrator instance
rag_orchestrator: AgentOrchestrator = None


def init_rag_system():
    """Initialize the RAG system with API key from environment"""
    global rag_orchestrator

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. RAG system will not be initialized.")
        return False

    try:
        rag_orchestrator = create_agentic_rag(api_key)
        print("RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    if rag_orchestrator is None:
        return jsonify({'error': 'RAG system not initialized. Check GEMINI_API_KEY.'}), 500

    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')
    store_name = request.form.get('store_name', f'store_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    uploaded_files = []
    file_paths = []

    # Save uploaded files
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(file_path)
            uploaded_files.append(filename)
            file_paths.append(file_path)
        else:
            return jsonify({'error': f'Invalid file: {file.filename}'}), 400

    # Create knowledge base
    try:
        # Chunking configuration
        chunking_config = {
            'white_space_config': {
                'max_tokens_per_chunk': int(request.form.get('max_tokens', 500)),
                'max_overlap_tokens': int(request.form.get('overlap_tokens', 50))
            }
        }

        store_id = rag_orchestrator.create_knowledge_base(
            store_name=store_name,
            file_paths=file_paths,
            chunking_config=chunking_config
        )

        # Store in session
        session['current_store'] = store_id
        session['store_name'] = store_name

        return jsonify({
            'success': True,
            'message': f'Successfully created knowledge base: {store_name}',
            'store_id': store_id,
            'files': uploaded_files,
            'file_count': len(uploaded_files)
        })

    except Exception as e:
        return jsonify({'error': f'Error creating knowledge base: {str(e)}'}), 500


@app.route('/query', methods=['POST'])
def query():
    """Handle user queries"""
    if rag_orchestrator is None:
        return jsonify({'error': 'RAG system not initialized. Check GEMINI_API_KEY.'}), 500

    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    # Get query parameters
    store_name = data.get('store_name') or session.get('current_store')
    metadata_filter = data.get('metadata_filter')
    include_citations = data.get('include_citations', True)

    if not store_name:
        return jsonify({'error': 'No knowledge base is active. Please upload files first.'}), 400

    try:
        result = rag_orchestrator.query(
            question=question,
            store_name=store_name,
            metadata_filter=metadata_filter,
            include_citations=include_citations
        )

        return jsonify({
            'success': True,
            'answer': result.get('text', ''),
            'query_type': result.get('query_type', 'GENERAL'),
            'citations': result.get('citations', []),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500


@app.route('/stats')
def stats():
    """Get system statistics"""
    if rag_orchestrator is None:
        return jsonify({'error': 'RAG system not initialized'}), 500

    try:
        stats = rag_orchestrator.get_stats()
        stats['session_store'] = session.get('store_name', 'None')
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    if rag_orchestrator is None:
        return jsonify({'error': 'RAG system not initialized'}), 500

    try:
        rag_orchestrator.clear_conversation()
        return jsonify({'success': True, 'message': 'Conversation history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stores')
def list_stores():
    """List all available stores"""
    if rag_orchestrator is None:
        return jsonify({'error': 'RAG system not initialized'}), 500

    try:
        stores = rag_orchestrator.file_manager.list_stores()
        store_list = [{'name': store.name, 'display_name': store.display_name} for store in stores]
        return jsonify({'stores': store_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_orchestrator is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413


if __name__ == '__main__':
    # Initialize RAG system
    if init_rag_system():
        print("Starting Flask application with Agentic RAG system...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "="*60)
        print("ERROR: Could not initialize RAG system")
        print("Please set the GEMINI_API_KEY environment variable:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("="*60 + "\n")
        print("Starting Flask application anyway (limited functionality)...")
        app.run(debug=True, host='0.0.0.0', port=5000)
