-- Supabase Database Schema for Persistent Document Storage
-- Run this SQL in your Supabase SQL Editor to create the necessary tables

-- Table to store user's vector stores
CREATE TABLE IF NOT EXISTS user_vector_stores (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    store_id TEXT NOT NULL,
    store_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, store_name)
);

-- Table to track individual documents in vector stores
CREATE TABLE IF NOT EXISTS user_documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    store_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT,
    file_size BIGINT,
    mime_type TEXT,
    gemini_file_id TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_user_vector_stores_user_id ON user_vector_stores(user_id);
CREATE INDEX IF NOT EXISTS idx_user_documents_user_id ON user_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_user_documents_store_id ON user_documents(store_id);

-- Enable Row Level Security (RLS)
ALTER TABLE user_vector_stores ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_documents ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Users can only access their own data
CREATE POLICY "Users can view their own vector stores"
    ON user_vector_stores FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own vector stores"
    ON user_vector_stores FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own vector stores"
    ON user_vector_stores FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own vector stores"
    ON user_vector_stores FOR DELETE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can view their own documents"
    ON user_documents FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own documents"
    ON user_documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own documents"
    ON user_documents FOR DELETE
    USING (auth.uid() = user_id);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_user_vector_stores_updated_at
    BEFORE UPDATE ON user_vector_stores
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your setup)
-- Note: Supabase typically handles this automatically with RLS
