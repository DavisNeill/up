# Admin System Setup Guide

This guide explains how to set up and use the admin-managed document system.

## 🔑 Overview

The system now has **role-based access control**:

- **👤 Regular Users**: Can only query documents
- **👑 Admins**: Can upload, delete, hide/show documents and perform bulk operations

### Key Features

✅ **Admin-Only Document Management** - Only admins can upload/modify documents
✅ **Bulk Operations** - Select and delete/hide/show multiple documents at once
✅ **Document Visibility** - Hide documents from regular users without deleting
✅ **Shared Knowledge Base** - All users query the same document collection
✅ **Role-Based UI** - Different interfaces for admins vs users

## 📋 Setup Instructions

### Step 1: Run the Admin Database Schema

1. Go to your **Supabase Dashboard** → **SQL Editor**
2. Open a new query
3. Copy the entire content from `database_schema_admin.sql`
4. Click **Run**

This will:
- Create the `user_profiles` table for roles
- Add visibility columns to documents (`is_visible`, `hidden_by`, `hidden_at`)
- Create admin-specific database functions
- Set up Row Level Security policies
- Create triggers for auto-profile creation

### Step 2: Create Your First Admin

After running the schema, you need to manually promote your first user to admin:

**Option A: Through Supabase Dashboard**
1. Sign up for an account through your app
2. Go to Supabase Dashboard → **Table Editor** → `user_profiles`
3. Find your user (by email)
4. Change `role` from `'user'` to `'admin'`
5. Click Save

**Option B: Through SQL**
```sql
-- Replace with your actual email
UPDATE user_profiles
SET role = 'admin'
WHERE email = 'your-email@example.com';
```

**Option C: For new signups**
```sql
-- Make the next signup an admin by changing the default
-- WARNING: Change this back after creating your admin!
ALTER TABLE user_profiles
ALTER COLUMN role SET DEFAULT 'admin';

-- After your admin signs up, change it back:
ALTER TABLE user_profiles
ALTER COLUMN role SET DEFAULT 'user';
```

### Step 3: Test the Admin System

1. **Log out** and **log back in** (to refresh your session with new role)
2. You should now see:
   - 🟢 **"ADMIN"** badge next to your email
   - 📁 **Upload Documents** tab (hidden for regular users)
   - ✅ Checkboxes on each document
   - 🔧 Hide/Show buttons
   - 🗑️ Bulk action buttons

## 🎯 Admin Features

### Document Upload (Admin Only)

1. Go to **📁 Upload Documents** tab
2. Select files
3. Configure chunking (optional)
4. Click "Upload to Knowledge Base"
5. Documents are now available to ALL users

### Individual Document Actions

For each document, admins can:

- **👁️‍🗨️ Hide** - Make document invisible to regular users
- **👁️ Show** - Make hidden document visible again
- **🗑️ Delete** - Permanently remove document
- **☑️ Checkbox** - Select for bulk operations

### Bulk Operations

Select multiple documents and perform actions on all at once:

1. Check individual documents OR click **"Select All"**
2. Click one of the bulk action buttons:
   - **🗑️ Delete Selected** - Permanently delete all selected
   - **👁️‍🗨️ Hide Selected** - Hide all selected documents
   - **👁️ Show Selected** - Show all selected documents
3. Confirm the action
4. All selected documents are updated instantly

### Document Visibility

**Visible Documents** (default):
- ✅ All users can query them
- ✅ No badge shown
- Admins can hide them

**Hidden Documents**:
- 🚫 Regular users cannot see or query them
- 👑 Admins can still see them (with "HIDDEN" badge)
- 🔍 Useful for temporarily removing documents
- 📊 Tracks who hid it and when

## 👤 Regular User Experience

Regular users (non-admins) have a simplified interface:

- ✅ Can ask questions
- ✅ Can view their memories
- ✅ Can see analytics
- ❌ Cannot see Upload tab
- ❌ Cannot upload documents
- ❌ Cannot delete documents
- ❌ Cannot see document list
- ❌ Cannot see hidden documents

They query the **shared knowledge base** managed by admins.

## 🔐 Security & Permissions

### Database-Level Security

- **Row Level Security (RLS)** enforces all permissions
- Users cannot bypass restrictions via API
- All checks happen in PostgreSQL functions
- Admins are verified for every operation

### Permission Matrix

| Action | Regular User | Admin |
|--------|--------------|-------|
| Query documents | ✅ (visible only) | ✅ (all) |
| Upload documents | ❌ | ✅ |
| Delete documents | ❌ | ✅ |
| Hide/Show documents | ❌ | ✅ |
| Bulk operations | ❌ | ✅ |
| View hidden docs | ❌ | ✅ |
| Manage memories | ✅ (own) | ✅ (own) |

## 📊 Database Schema Changes

### New Table: `user_profiles`
```sql
- id: UUID (references auth.users)
- email: TEXT
- role: TEXT ('admin' or 'user')
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
```

### Updated Table: `user_documents`
```sql
Added columns:
- is_visible: BOOLEAN (default true)
- hidden_by: UUID (admin who hid it)
- hidden_at: TIMESTAMP (when it was hidden)
```

### New Functions:
- `is_admin(user_id)` - Check if user is admin
- `get_user_role(user_id)` - Get user's role
- `hide_document(doc_id, admin_id)` - Hide a document
- `show_document(doc_id, admin_id)` - Show a document
- `bulk_delete_documents(doc_ids[], admin_id)` - Delete multiple
- `bulk_hide_documents(doc_ids[], admin_id)` - Hide multiple
- `bulk_show_documents(doc_ids[], admin_id)` - Show multiple

## 🔧 API Endpoints (Admin Only)

All these endpoints require admin role:

```bash
# Upload documents
POST /upload
Body: FormData with files

# Delete single document
DELETE /api/documents/delete/{document_id}

# Hide/Show single document
POST /api/documents/hide/{document_id}
POST /api/documents/show/{document_id}

# Bulk operations
POST /api/documents/bulk-delete
Body: { "document_ids": ["id1", "id2", ...] }

POST /api/documents/bulk-hide
Body: { "document_ids": ["id1", "id2", ...] }

POST /api/documents/bulk-show
Body: { "document_ids": ["id1", "id2", ...] }
```

## 🎨 UI Elements for Admins

### Admin Badge
- Displays next to user email in header
- Green badge with "ADMIN" text
- Only visible to admins

### Upload Tab
- Hidden for regular users
- Shows file upload interface
- Shows document list with management controls

### Document List (Admin View)
- ☑️ Checkbox for each document
- **Select All** checkbox
- Bulk action buttons at top
- Hide/Show buttons per document
- Delete buttons per document
- "HIDDEN" badge for invisible documents

### Document List (User View)
- Not visible to regular users
- Regular users only see query interface

## 🚨 Troubleshooting

### "Admin access required" error
- Make sure you promoted your user to admin in database
- **Log out and log back in** to refresh session
- Check `user_profiles` table to verify role is 'admin'

### Upload tab not showing
- Verify you're logged in as admin
- Hard refresh page (Ctrl+Shift+R or Cmd+Shift+R)
- Check browser console for errors

### Cannot hide/show documents
- Ensure `database_schema_admin.sql` was run successfully
- Check that functions exist: `SELECT * FROM pg_proc WHERE proname LIKE '%document%';`
- Verify RLS policies are in place

### Bulk operations not working
- Check that document IDs are valid UUIDs
- Verify selections are being tracked (check selectedCount)
- Look at Network tab in browser DevTools for error responses

## 🔄 Migration from Old System

If you already have documents in the old per-user system:

1. **Backup your data** first!
2. Run `database_schema_admin.sql`
3. Existing documents will have `is_visible = true` by default
4. Create your first admin (see Step 2 above)
5. Test with a non-admin account to verify permissions

## 💡 Use Cases

### Corporate Knowledge Base
- Admins curate company documents
- Employees query the knowledge base
- Hide outdated documents temporarily
- Delete obsolete information

### Educational Platform
- Teachers/instructors are admins
- Students are regular users
- Control what materials are accessible
- Update content without re-uploading everything

### Customer Support
- Support leads manage documentation
- Support agents query knowledge base
- Quickly hide incorrect information
- Bulk update seasonal content

## 🎯 Best Practices

1. **Have Multiple Admins** - Create 2-3 admin accounts for redundancy
2. **Use Hide Instead of Delete** - Hide documents first, delete only when certain
3. **Regular Audits** - Review hidden documents periodically
4. **Clear Naming** - Use descriptive filenames for easy management
5. **Bulk Operations** - Use bulk delete/hide for efficiency
6. **Document Versioning** - Include version/date in filenames
7. **Test Changes** - Use a non-admin account to verify user experience

## 🆘 Support

If you encounter issues:

1. Check browser console (F12) for JavaScript errors
2. Check Flask console for backend errors
3. Verify database schema was applied correctly
4. Test with a fresh user account
5. Review Supabase logs in dashboard

## 🔗 Related Files

- `database_schema_admin.sql` - Admin database schema
- `app.py` - Backend with admin middleware
- `templates/index.html` - Frontend with admin UI
- `SETUP_GUIDE.md` - General setup guide

---

**Ready to manage your knowledge base! 🚀**
