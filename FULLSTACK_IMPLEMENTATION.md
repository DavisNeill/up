# Full-Stack Agentic RAG with Authentication & Memory Management

## 🎯 Overview

Complete full-stack implementation with:
- ✅ **User Authentication** (Supabase)
- ✅ **Session Management** (Flask sessions)
- ✅ **Memory Management** (Mem0)
- ✅ **Memory Analytics** (Query patterns, usage stats)
- ✅ **Memory Export** (CSV/JSON download)
- ✅ **Multi-user Support** (User-isolated memories)

---

## 📁 Architecture

```
Frontend (Templates)           Backend (Flask)              Services
┌─────────────────┐           ┌──────────────┐            ┌──────────┐
│ Login/Signup UI │──────────>│ Auth Routes  │───────────>│ Supabase │
│ Main Dashboard  │           │   /api/auth  │            └──────────┘
│ Memory Panel    │           ├──────────────┤
│ Analytics View  │──────────>│ Memory API   │            ┌──────────┐
│ Export Button   │           │   /api/memory│───────────>│   Mem0   │
└─────────────────┘           ├──────────────┤            └──────────┘
                              │ RAG Queries  │            ┌──────────┐
                              │   /query     │───────────>│  Gemini  │
                              └──────────────┘            └──────────┘
```

---

## 🔧 Backend Implementation (app.py - ✅ COMPLETE)

### Authentication Endpoints

#### POST `/api/auth/signup`
```python
# Register new user
{
  "email": "user@example.com",
  "password": "secure_password",
  "name": "User Name"
}

Response:
{
  "success": true,
  "message": "Account created. Check email for verification.",
  "user": {"id": "uuid", "email": "user@example.com"}
}
```

#### POST `/api/auth/login`
```python
# Login user
{
  "email": "user@example.com",
  "password": "password"
}

Response:
{
  "success": true,
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "name": "User Name"
  }
}

# Sets session cookie with user_id, user_email, access_token
```

#### POST `/api/auth/logout`
```python
# Logout (requires authentication)
Response:
{
  "success": true,
  "message": "Logged out successfully"
}
```

#### GET `/api/auth/user`
```python
# Get current user info (requires authentication)
Response:
{
  "user": {
    "id": "uuid",
    "email": "user@example.com"
  }
}
```

---

### Memory Management Endpoints

#### GET `/api/memory/all?limit=50`
```python
# Get all memories for current user
Response:
{
  "success": true,
  "memories": [
    {
      "memory": "User prefers technical explanations",
      "metadata": {
        "query_type": "ANALYTICAL",
        "timestamp": "2025-01-15 10:30:00",
        "has_citations": true
      }
    },
    ...
  ],
  "count": 15
}
```

#### POST `/api/memory/add`
```python
# Manually add a preference
{
  "preference": "I prefer detailed technical explanations"
}

Response:
{
  "success": true,
  "message": "Preference added successfully"
}
```

#### POST `/api/memory/clear`
```python
# Clear all user memories
Response:
{
  "success": true,
  "message": "All memories cleared successfully"
}
```

---

### Memory Analytics Endpoint

#### GET `/api/memory/analytics`
```python
# Get memory analytics for current user
Response:
{
  "success": true,
  "analytics": {
    "total_memories": 45,
    "query_type_distribution": {
      "FACTUAL": 20,
      "ANALYTICAL": 15,
      "COMPARISON": 10
    },
    "activity_by_day": {
      "2025-01-15": 12,
      "2025-01-14": 18,
      "2025-01-13": 15
    },
    "recent_activity": 45
  }
}
```

---

### Memory Export Endpoint

#### GET `/api/memory/export?format=csv`
```python
# Export memories as CSV or JSON

# CSV Format:
Memory,Type,Timestamp,Has Citations
"User prefers technical...",ANALYTICAL,"2025-01-15 10:30:00",true
"User asked about ML...",FACTUAL,"2025-01-15 09:15:00",false

# JSON Format (format=json):
Downloads memories as JSON file
```

---

### Enhanced Query Endpoint

#### POST `/query`
```python
# Query with automatic memory integration
{
  "question": "Explain neural networks",
  "include_citations": true
}

Response:
{
  "success": true,
  "answer": "Neural networks are...",
  "query_type": "ANALYTICAL",
  "citations": [...],
  "memories_used": 3,  # NEW!
  "memory_enabled": true,  # NEW!
  "timestamp": "2025-01-15T10:30:00"
}

# Automatically:
# 1. Retrieves 5 relevant memories
# 2. Injects memory context into prompt
# 3. Generates personalized response
# 4. Stores new memory from interaction
```

---

## 🎨 Frontend Implementation (TO BUILD)

### Required Frontend Features

#### 1. Authentication UI
```html
<!-- Login/Signup Modal -->
- Email/password fields
- Toggle between login/signup
- Error messages
- Remember me checkbox
- Logout button in header
```

#### 2. Memory Dashboard Tab
```html
<!-- Memory Visualization -->
- List of all user memories
- Search/filter memories
- Delete individual memories
- Clear all button
- Add manual preference
- Real-time memory count
```

#### 3. Analytics Dashboard Tab
```html
<!-- Memory Analytics -->
- Total memories count
- Query type pie chart
- Activity timeline (line/bar chart)
- Most active days
- Memory growth trend
```

#### 4. Export Functionality
```html
<!-- Export Buttons -->
- Export as CSV button
- Export as JSON button
- Download triggers file download
```

#### 5. Enhanced Query Interface
```html
<!-- Query with Memory Info -->
- Show memories_used count
- Memory enabled indicator
- "Using X memories" badge
- Personalization indicator
```

---

## 🗄️ Database Setup (Supabase)

### Supabase Configuration

1. **Create Supabase Project**
   - Go to https://supabase.com
   - Create new project
   - Copy project URL and anon key

2. **Enable Email Auth**
   ```sql
   -- Authentication is enabled by default
   -- Configure email templates in Dashboard > Authentication > Templates
   ```

3. **Optional: Create User Profiles Table**
   ```sql
   CREATE TABLE profiles (
     id UUID REFERENCES auth.users PRIMARY KEY,
     email TEXT,
     name TEXT,
     created_at TIMESTAMPTZ DEFAULT NOW(),
     updated_at TIMESTAMPTZ DEFAULT NOW()
   );

   -- Enable RLS
   ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

   -- Policy: Users can read own profile
   CREATE POLICY "Users can view own profile"
     ON profiles FOR SELECT
     USING (auth.uid() = id);

   -- Policy: Users can update own profile
   CREATE POLICY "Users can update own profile"
     ON profiles FOR UPDATE
     USING (auth.uid() = id);
   ```

4. **Environment Variables**
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key-here
   ```

---

## 🔐 Security Implementation

### 1. Session Management
```python
# Flask sessions with 7-day expiry
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
session.permanent = True

# Stores:
# - user_id (UUID from Supabase)
# - user_email
# - access_token (Supabase JWT)
```

### 2. Authentication Decorator
```python
@login_required
def protected_route():
    # Checks session['user_id']
    # Returns 401 if not authenticated
    pass
```

### 3. User Isolation
```python
# All memory operations use session['user_id']
user_id = session.get('user_id')
rag_orchestrator.query(..., user_id=user_id)

# Memories are isolated per user
# User A cannot access User B's memories
```

### 4. CORS Configuration
```python
# Enable CORS for API requests
from flask_cors import CORS
CORS(app)
```

---

## 📊 Memory Analytics Visualization

### Frontend Chart Implementation (Example with Chart.js)

```javascript
// Query Type Distribution - Pie Chart
async function loadAnalytics() {
  const response = await fetch('/api/memory/analytics');
  const data = await response.json();

  // Pie chart for query types
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: Object.keys(data.analytics.query_type_distribution),
      datasets: [{
        data: Object.values(data.analytics.query_type_distribution),
        backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#4facfe']
      }]
    }
  });

  // Line chart for activity over time
  new Chart(ctx2, {
    type: 'line',
    data: {
      labels: Object.keys(data.analytics.activity_by_day),
      datasets: [{
        label: 'Queries per Day',
        data: Object.values(data.analytics.activity_by_day),
        borderColor: '#667eea',
        fill: false
      }]
    }
  });
}
```

---

## 🚀 Deployment Checklist

### Environment Variables
```bash
# Required
✅ GEMINI_API_KEY           # Google Gemini API
✅ SECRET_KEY               # Flask session encryption
✅ SUPABASE_URL             # Supabase project URL
✅ SUPABASE_KEY             # Supabase anon key

# Optional
⚪ OPENAI_API_KEY           # For Mem0 (if using OpenAI backend)
⚪ FLASK_ENV=production     # Production mode
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your keys

# Run application
python app.py
```

### Production Considerations
```bash
# Use a production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or use Docker (create Dockerfile)
docker build -t agentic-rag .
docker run -p 5000:5000 --env-file .env agentic-rag
```

---

## 🧪 Testing the System

### Test Authentication
```bash
# Signup
curl -X POST http://localhost:5000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123","name":"Test User"}'

# Login
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}' \
  -c cookies.txt

# Get current user (with cookies)
curl http://localhost:5000/api/auth/user -b cookies.txt
```

### Test Memory Management
```bash
# Add preference
curl -X POST http://localhost:5000/api/memory/add \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"preference":"I prefer technical explanations"}'

# Get all memories
curl http://localhost:5000/api/memory/all -b cookies.txt

# Get analytics
curl http://localhost:5000/api/memory/analytics -b cookies.txt

# Export memories
curl http://localhost:5000/api/memory/export?format=csv -b cookies.txt > memories.csv
```

### Test RAG Query with Memory
```bash
# Query (automatically uses memory)
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"question":"Explain machine learning"}'
```

---

## 📈 Features Implemented

### ✅ Backend (Complete)
- [x] Supabase authentication integration
- [x] User signup/login/logout
- [x] Session management with cookies
- [x] Login required decorator
- [x] Memory CRUD endpoints
- [x] Memory analytics endpoint
- [x] Memory export (CSV/JSON)
- [x] User-isolated memory storage
- [x] Automatic memory integration in queries
- [x] CORS enabled for API requests

### ⏸️ Frontend (To Be Built)
- [ ] Authentication UI (login/signup modal)
- [ ] Protected routes (redirect if not logged in)
- [ ] Memory dashboard tab
- [ ] Memory list with search/filter
- [ ] Analytics dashboard with charts
- [ ] Export buttons (CSV/JSON)
- [ ] User profile display
- [ ] Logout button
- [ ] Memory usage indicators
- [ ] Real-time stats updates

---

## 🎓 Usage Examples

### Complete User Flow

1. **User Signs Up**
   - Frontend: User fills signup form
   - Backend: Creates Supabase user
   - Backend: Sends verification email
   - Frontend: Shows success message

2. **User Logs In**
   - Frontend: User enters credentials
   - Backend: Authenticates with Supabase
   - Backend: Creates Flask session
   - Backend: Sets user_id in RAG system
   - Frontend: Redirects to dashboard

3. **User Uploads Documents**
   - Frontend: User selects PDF files
   - Backend: Saves files to uploads/
   - Backend: Creates Gemini File Search store
   - Backend: Indexes documents
   - Frontend: Shows success + file count

4. **User Asks Question**
   - Frontend: User types question
   - Backend: Retrieves 5 relevant memories
   - Backend: Adds memory context to prompt
   - Backend: Queries Gemini with context
   - Backend: Generates personalized response
   - Backend: Stores new memory
   - Frontend: Shows answer + "Used 3 memories"

5. **User Views Memory Dashboard**
   - Frontend: User clicks "Memories" tab
   - Backend: Returns all user memories
   - Frontend: Displays memory list
   - Frontend: Shows memory count

6. **User Views Analytics**
   - Frontend: User clicks "Analytics" tab
   - Backend: Analyzes memory patterns
   - Frontend: Displays charts:
     - Query type pie chart
     - Activity timeline
     - Total memories count

7. **User Exports Memories**
   - Frontend: User clicks "Export CSV"
   - Backend: Generates CSV file
   - Frontend: Downloads memories.csv

8. **User Logs Out**
   - Frontend: User clicks logout
   - Backend: Clears session
   - Backend: Signs out from Supabase
   - Frontend: Redirects to login

---

## 🔑 Key Implementation Details

### User ID Propagation
```python
# Login sets user_id in session
session['user_id'] = response.user.id

# RAG system stores current user
rag_orchestrator.set_user_id(response.user.id)

# Every query uses session user_id
user_id = session.get('user_id')
result = rag_orchestrator.query(..., user_id=user_id)
```

### Memory Workflow
```
1. User Query → Extract user_id from session
2. Memory Search → Find 5 relevant memories for this user
3. Context Building → Format memories as prompt context
4. RAG Query → Gemini File Search with memory context
5. Response Generation → Personalized based on memories
6. Memory Storage → Store new memory for this user
7. Return Response → With memories_used count
```

### Analytics Calculation
```python
# Count by query type
query_types = {}
for memory in memories:
    qtype = memory.get('metadata', {}).get('query_type', 'UNKNOWN')
    query_types[qtype] = query_types.get(qtype, 0) + 1

# Group by day
activity_by_day = {}
for timestamp in timestamps:
    date = timestamp.split(' ')[0]
    activity_by_day[date] = activity_by_day.get(date, 0) + 1
```

---

## 🎯 Next Steps for Complete Implementation

1. **Build Frontend UI** (index.html)
   - Authentication modal
   - Memory dashboard
   - Analytics charts
   - Export buttons

2. **Add Chart.js** for analytics visualization

3. **Add Session Persistence** Check on page load

4. **Add Loading States** for async operations

5. **Add Error Handling** UI for failed requests

6. **Add Toast Notifications** for success/error messages

7. **Deploy to Production**
   - Set up Gunicorn
   - Configure NGINX
   - Set environment variables
   - Enable HTTPS

---

## 📚 Dependencies

```
google-genai>=0.2.0      # Gemini API
mem0ai>=0.1.0            # Memory layer
Flask>=3.0.0             # Web framework
Flask-CORS>=4.0.0        # CORS support
supabase>=2.0.0          # Authentication
PyJWT>=2.8.0             # JWT tokens
pandas>=2.0.0            # Data handling
python-dotenv>=1.0.0     # Environment vars
```

---

## ✅ Status Summary

**Backend:** ✅ 100% Complete
**Authentication:** ✅ Fully Implemented
**Memory Management:** ✅ All CRUD Operations
**Analytics:** ✅ Complete with Export
**Session Management:** ✅ Secure & Isolated
**Frontend:** ⏸️ Requires Implementation

**The backend is production-ready! Frontend UI needs to be built to connect all the pieces.**

