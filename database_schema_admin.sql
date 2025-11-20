-- Supabase Database Schema for Admin-Managed Document Storage
-- Run this SQL in your Supabase SQL Editor to upgrade to admin system

-- ============================================================================
-- STEP 1: Add user roles system
-- ============================================================================

-- User profiles table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT,
    role TEXT DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on user_profiles
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Users can view their own profile
CREATE POLICY "Users can view own profile"
    ON user_profiles FOR SELECT
    USING (auth.uid() = id);

-- Only admins can update roles (or users can update their own non-role fields)
CREATE POLICY "Users can update own profile"
    ON user_profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (
        auth.uid() = id AND
        (role = (SELECT role FROM user_profiles WHERE id = auth.uid()) OR
         (SELECT role FROM user_profiles WHERE id = auth.uid()) = 'admin')
    );

-- Auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email, role)
    VALUES (NEW.id, NEW.email, 'user');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger on auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();

-- Helper function to check if user is admin
CREATE OR REPLACE FUNCTION is_admin(user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM user_profiles
        WHERE id = user_id AND role = 'admin'
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- STEP 2: Update document tables for admin-managed system
-- ============================================================================

-- Drop old RLS policies (we'll create new ones)
DROP POLICY IF EXISTS "Users can view their own vector stores" ON user_vector_stores;
DROP POLICY IF EXISTS "Users can insert their own vector stores" ON user_vector_stores;
DROP POLICY IF EXISTS "Users can update their own vector stores" ON user_vector_stores;
DROP POLICY IF EXISTS "Users can delete their own vector stores" ON user_vector_stores;
DROP POLICY IF EXISTS "Users can view their own documents" ON user_documents;
DROP POLICY IF EXISTS "Users can insert their own documents" ON user_documents;
DROP POLICY IF EXISTS "Users can delete their own documents" ON user_documents;

-- Add visibility column to documents
ALTER TABLE user_documents
ADD COLUMN IF NOT EXISTS is_visible BOOLEAN DEFAULT true,
ADD COLUMN IF NOT EXISTS hidden_by UUID REFERENCES auth.users(id),
ADD COLUMN IF NOT EXISTS hidden_at TIMESTAMP WITH TIME ZONE;

-- Create index on visibility
CREATE INDEX IF NOT EXISTS idx_user_documents_visible ON user_documents(is_visible);

-- Make stores system-wide (remove user_id uniqueness constraint)
ALTER TABLE user_vector_stores DROP CONSTRAINT IF EXISTS user_vector_stores_user_id_store_name_key;

-- ============================================================================
-- STEP 3: New RLS Policies for Admin System
-- ============================================================================

-- Vector Stores: Only admins can manage
CREATE POLICY "Admins can view all vector stores"
    ON user_vector_stores FOR SELECT
    USING (is_admin(auth.uid()));

CREATE POLICY "Admins can insert vector stores"
    ON user_vector_stores FOR INSERT
    WITH CHECK (is_admin(auth.uid()));

CREATE POLICY "Admins can update vector stores"
    ON user_vector_stores FOR UPDATE
    USING (is_admin(auth.uid()));

CREATE POLICY "Admins can delete vector stores"
    ON user_vector_stores FOR DELETE
    USING (is_admin(auth.uid()));

-- Documents: Admins manage, all users can view visible ones
CREATE POLICY "Users can view visible documents"
    ON user_documents FOR SELECT
    USING (is_visible = true);

CREATE POLICY "Admins can view all documents"
    ON user_documents FOR SELECT
    USING (is_admin(auth.uid()));

CREATE POLICY "Admins can insert documents"
    ON user_documents FOR INSERT
    WITH CHECK (is_admin(auth.uid()));

CREATE POLICY "Admins can update documents"
    ON user_documents FOR UPDATE
    USING (is_admin(auth.uid()));

CREATE POLICY "Admins can delete documents"
    ON user_documents FOR DELETE
    USING (is_admin(auth.uid()));

-- ============================================================================
-- STEP 4: Helper functions for admin operations
-- ============================================================================

-- Function to get user role
CREATE OR REPLACE FUNCTION get_user_role(user_id UUID)
RETURNS TEXT AS $$
DECLARE
    user_role TEXT;
BEGIN
    SELECT role INTO user_role FROM user_profiles WHERE id = user_id;
    RETURN COALESCE(user_role, 'user');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to hide document
CREATE OR REPLACE FUNCTION hide_document(doc_id UUID, admin_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    IF is_admin(admin_id) THEN
        UPDATE user_documents
        SET is_visible = false, hidden_by = admin_id, hidden_at = NOW()
        WHERE id = doc_id;
        RETURN true;
    END IF;
    RETURN false;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to show document
CREATE OR REPLACE FUNCTION show_document(doc_id UUID, admin_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    IF is_admin(admin_id) THEN
        UPDATE user_documents
        SET is_visible = true, hidden_by = NULL, hidden_at = NULL
        WHERE id = doc_id;
        RETURN true;
    END IF;
    RETURN false;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to bulk delete documents
CREATE OR REPLACE FUNCTION bulk_delete_documents(doc_ids UUID[], admin_id UUID)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    IF is_admin(admin_id) THEN
        DELETE FROM user_documents WHERE id = ANY(doc_ids);
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        RETURN deleted_count;
    END IF;
    RETURN 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to bulk hide documents
CREATE OR REPLACE FUNCTION bulk_hide_documents(doc_ids UUID[], admin_id UUID)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    IF is_admin(admin_id) THEN
        UPDATE user_documents
        SET is_visible = false, hidden_by = admin_id, hidden_at = NOW()
        WHERE id = ANY(doc_ids);
        GET DIAGNOSTICS updated_count = ROW_COUNT;
        RETURN updated_count;
    END IF;
    RETURN 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to bulk show documents
CREATE OR REPLACE FUNCTION bulk_show_documents(doc_ids UUID[], admin_id UUID)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    IF is_admin(admin_id) THEN
        UPDATE user_documents
        SET is_visible = true, hidden_by = NULL, hidden_at = NULL
        WHERE id = ANY(doc_ids);
        GET DIAGNOSTICS updated_count = ROW_COUNT;
        RETURN updated_count;
    END IF;
    RETURN 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- STEP 5: Create first admin user (IMPORTANT!)
-- ============================================================================

-- After running this schema, you MUST manually set your first admin:
-- 1. Sign up through the app
-- 2. Go to Supabase Dashboard → Table Editor → user_profiles
-- 3. Find your user and change role from 'user' to 'admin'
-- OR run this SQL (replace with your email):
-- UPDATE user_profiles SET role = 'admin' WHERE email = 'your-email@example.com';

-- ============================================================================
-- SUMMARY
-- ============================================================================

-- This schema creates an admin-managed document system where:
-- ✅ Only admins can upload/delete/hide/show documents
-- ✅ All users can view visible documents
-- ✅ All users can query the shared knowledge base
-- ✅ Admins can perform bulk operations
-- ✅ Document visibility is tracked (who hid it and when)
-- ✅ First user needs to be manually promoted to admin

-- After running this schema:
-- 1. Make your first user an admin (see STEP 5 above)
-- 2. Admins can upload documents for all users
-- 3. Admins can hide/show documents
-- 4. Admins can bulk delete documents
-- 5. Regular users can only query, not manage documents
