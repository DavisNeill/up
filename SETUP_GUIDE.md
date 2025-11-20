# Setup Guide - Persistent Document Storage

This guide will help you set up the persistent document storage feature for your Agentic RAG system.

## Overview

The system now supports **persistent document storage**, meaning:
- ✅ Upload documents once and they're saved forever
- ✅ Documents persist across sessions
- ✅ Query your knowledge base anytime without re-uploading
- ✅ View, manage, and delete documents
- ✅ Each user has their own isolated document collection

## Prerequisites

1. **Supabase Account** - Sign up at https://supabase.com (free tier available)
2. **Google Gemini API Key** - Get from https://ai.google.dev/
3. **Python 3.8+** installed

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Supabase Database

### 2.1 Create a Supabase Project

1. Go to https://supabase.com
2. Click "New Project"
3. Choose your organization
4. Set project name (e.g., "agentic-rag")
5. Set a strong database password (save it securely!)
6. Choose your region
7. Click "Create new project" (takes 2-3 minutes)

### 2.2 Run the Database Schema

1. Once your project is created, go to the **SQL Editor** (left sidebar)
2. Click "New query"
3. Copy the entire content from `database_schema.sql`
4. Paste it into the SQL editor
5. Click "Run" (bottom right)

You should see: **"Success. No rows returned"**

This creates two tables:
- `user_vector_stores` - Stores your Gemini vector store IDs
- `user_documents` - Tracks all your uploaded documents

### 2.3 Get Your Supabase Credentials

1. Go to **Project Settings** (gear icon in sidebar)
2. Click **API** tab
3. Copy these values:
   - **Project URL** (looks like: `https://xxxxx.supabase.co`)
   - **anon public key** (starts with `eyJ...`)

## Step 3: Configure Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and fill in your credentials:

```bash
# Google Gemini API Key (get from https://ai.google.dev/)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Supabase Configuration
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGc...your_actual_anon_key_here

# Flask Secret (generate with: python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=your_random_secret_key_here

# Flask Environment
FLASK_ENV=development
```

**Important:** To generate a secure SECRET_KEY, run:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Step 4: Run the Application

```bash
python app.py
```

You should see:
```
Supabase initialized successfully
RAG system initialized successfully with Google Gemini embeddings for memory
 * Running on http://127.0.0.1:5000
```

## Step 5: Test the System

1. Open your browser to `http://localhost:5000`
2. **Sign up** for a new account
3. **Upload documents:**
   - Go to "Upload Documents" tab
   - Select files to upload
   - Click "Upload to Knowledge Base"
   - Your documents are now saved persistently!
4. **Query your knowledge:**
   - Go to "Ask Questions" tab
   - Ask questions about your documents
   - The system automatically uses your persistent knowledge base
5. **Manage documents:**
   - View all your uploaded documents in the "Upload Documents" tab
   - Delete individual documents if needed
   - Your knowledge base persists across sessions

## How It Works

### Document Persistence Flow

```
Upload → Gemini Vector Store → Database Record → Query Anytime
```

1. **Upload**: You upload documents once
2. **Gemini Creates Store**: Documents are indexed in Gemini's vector database
3. **Database Tracks**: Store ID and document metadata saved to Supabase
4. **Query**: System automatically loads your store for all queries
5. **Persist**: Documents remain available until you delete them

### Multi-User Isolation

- Each user has their own isolated document collection
- User A cannot see User B's documents
- Enforced by Row Level Security (RLS) in Supabase
- Your data is private and secure

## Database Schema Details

### user_vector_stores
Stores the Gemini vector store IDs for each user:
- `id` - Unique identifier
- `user_id` - User who owns this store
- `store_id` - Gemini vector store ID
- `store_name` - Name (default: "default")
- `description` - Description of the store
- `created_at` - When created
- `updated_at` - Last modified

### user_documents
Tracks individual documents:
- `id` - Unique identifier
- `user_id` - Document owner
- `store_id` - Which vector store it belongs to
- `file_name` - Original filename
- `file_path` - Server storage path
- `file_size` - Size in bytes
- `mime_type` - File type
- `uploaded_at` - Upload timestamp

## API Endpoints

### Document Management

**List Documents**
```bash
GET /api/documents/list
Response: { documents: [...], count: 5 }
```

**Delete Document**
```bash
DELETE /api/documents/delete/{document_id}
Response: { success: true, message: "Document deleted" }
```

**Get Stats**
```bash
GET /api/documents/stats
Response: {
  total_documents: 10,
  total_size: 5242880,
  total_size_mb: 5.0,
  store_exists: true
}
```

## Troubleshooting

### "Authentication required" error
- Make sure you're logged in
- Check if your session expired (7-day lifetime)
- Try logging out and back in

### "No knowledge base found" error
- Upload at least one document first
- Check if database tables were created (Step 2.2)
- Verify Supabase credentials in .env

### Documents not showing up
- Check browser console for errors (F12)
- Verify database schema is installed
- Try refreshing with the 🔄 button
- Check Supabase dashboard → Table Editor → user_documents

### "Error uploading files"
- Check GEMINI_API_KEY is valid
- Verify file types are allowed (see ALLOWED_EXTENSIONS in app.py)
- Check file size < 100MB
- Look at Flask console logs for detailed error

## Security Notes

1. **RLS Enabled**: Row Level Security ensures users can only access their own data
2. **Session Security**: 7-day sessions with secure cookies
3. **API Key Safety**: Never commit your .env file to git (.gitignore protects it)
4. **HTTPS**: Use HTTPS in production (Supabase requires it)

## Production Deployment

For production deployment:

1. Change `FLASK_ENV=production` in .env
2. Use a strong SECRET_KEY (32+ characters random)
3. Enable HTTPS
4. Set proper CORS origins in app.py
5. Use a production WSGI server (gunicorn, waitress)
6. Consider using a managed Flask hosting (Heroku, Render, Railway)

Example with gunicorn:
```bash
pip install gunicorn
gunicorn app:app -w 4 -b 0.0.0.0:5000
```

## Support

If you encounter issues:
1. Check the Flask console for error messages
2. Check browser console (F12) for frontend errors
3. Verify all environment variables are set correctly
4. Check Supabase logs in the dashboard
5. Review database_schema.sql was run successfully

## Next Steps

Now that persistent storage is set up:
- Upload your knowledge base once
- Query anytime without re-uploading
- Share the system with team members (each gets their own isolated storage)
- Build up your knowledge base over time
- Use the memory features to personalize responses

Enjoy your persistent, intelligent knowledge system! 🚀
