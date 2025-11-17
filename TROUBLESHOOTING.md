# 🔧 AutoATC Troubleshooting Guide

## Common Image Upload Errors & Solutions

### 1. **Connection Error: Cannot connect to backend**

**Symptoms:**
- Frontend shows "Connection Error: Cannot connect to backend at http://localhost:8000"
- Test button fails

**Solutions:**
```bash
# Check if backend is running
netstat -ano | findstr :8000

# If not running, start backend:
cd AutoATC/backend
python simple_main.py

# Or use the batch file:
start_backend.bat
```

**Verify backend is working:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","service":"AutoATC Backend","version":"1.0.0"}
```

---

### 2. **Analysis Failed: 500 Internal Server Error**

**Symptoms:**
- Backend is running but analysis fails
- Error 500 in response

**Common Causes:**
- Missing dependencies
- Image processing error
- Database error

**Solutions:**
```bash
# Install all required dependencies
cd AutoATC/backend
pip install -r requirements.txt

# Check backend logs for specific error
# Look at the terminal where simple_main.py is running
```

---

### 3. **Invalid Image Data Error**

**Symptoms:**
- Error: "Invalid image data"
- Status 400

**Solutions:**
- Ensure image is in supported format: JPG, JPEG, PNG, BMP
- Check image file is not corrupted
- Try a different image
- Image size should be < 10MB

---

### 4. **Timeout Error**

**Symptoms:**
- Request takes longer than 60 seconds
- Timeout error in frontend

**Solutions:**
- Use smaller images (resize to max 1920x1080)
- Check if AI models are loaded properly
- Increase timeout in frontend (app.py line 404)

---

### 5. **CORS Error**

**Symptoms:**
- Browser console shows CORS policy error
- Request blocked by browser

**Solutions:**
- Backend already has CORS enabled for localhost:8501
- Clear browser cache
- Try different browser
- Check backend CORS settings in simple_main.py

---

## 🧪 Diagnostic Steps

### Step 1: Run the diagnostic script
```bash
cd AutoATC
python test_upload.py

# Or test with a specific image:
python test_upload.py path/to/your/image.jpg
```

### Step 2: Check Backend Logs
Look at the terminal where `simple_main.py` is running for error messages.

### Step 3: Test Backend Directly
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test status endpoint
curl http://localhost:8000/api/v1/status
```

### Step 4: Use Frontend Connection Test
1. Open Streamlit app (http://localhost:8501)
2. Go to "Analysis" page
3. Expand "Backend Connection Test"
4. Click "Test Backend Connection"

---

## 🚀 Quick Start (If Nothing Works)

### Complete Reset:

1. **Stop all processes:**
   - Close all terminals running backend/frontend
   - Kill processes on ports 8000 and 8501

2. **Start Backend:**
```bash
cd AutoATC/backend
python simple_main.py
```
Wait for: "Uvicorn running on http://0.0.0.0:8000"

3. **Start Frontend (in new terminal):**
```bash
cd AutoATC/frontend/streamlit_app
streamlit run app.py
```
Wait for: "You can now view your Streamlit app in your browser"

4. **Test:**
   - Open http://localhost:8501
   - Click "Backend Connection Test"
   - Upload an image

---

## 📊 Expected Response Format

The backend should return:
```json
{
  "status": "success",
  "data": {
    "analysis_id": "...",
    "animal_type": "cattle",
    "confidence": 0.85,
    "atc_score": {
      "score": 78.5,
      "grade": "A",
      "factors": {...}
    },
    "breed_classification": {
      "breed": "Holstein Friesian",
      "confidence": 0.88
    },
    "measurements": {...},
    "processing_time": 2.5
  }
}
```

---

## 🐛 Still Having Issues?

1. **Check Python version:** Python 3.8+ required
2. **Check dependencies:** Run `pip list` to verify installations
3. **Check ports:** Ensure 8000 and 8501 are not used by other apps
4. **Check firewall:** Allow Python through Windows Firewall
5. **Check antivirus:** May block local server connections

---

## 📝 Logging & Debugging

### Enable Debug Mode in Frontend:
The updated frontend now shows:
- Request URL
- Image size
- Response status
- Detailed error messages
- Raw response data (in expander)

### Check Backend Logs:
Look for these in the backend terminal:
- `INFO: Started server process`
- `INFO: Uvicorn running on http://0.0.0.0:8000`
- Any error stack traces

---

## 💡 Tips

1. **Always start backend before frontend**
2. **Use the diagnostic script first** (`test_upload.py`)
3. **Check browser console** (F12) for JavaScript errors
4. **Try incognito mode** to rule out cache issues
5. **Use small test images first** (< 1MB)

