"""
Test script to diagnose image upload issues
"""
import requests
import base64
import json
import sys
from pathlib import Path

API_BASE_URL = "http://localhost:8000/api/v1"

def test_backend_health():
    """Test if backend is running"""
    print("=" * 60)
    print("Testing Backend Health...")
    print("=" * 60)
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to backend!")
        print("   Make sure backend is running: cd AutoATC/backend && python simple_main.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_api_status():
    """Test API status endpoint"""
    print("\n" + "=" * 60)
    print("Testing API Status...")
    print("=" * 60)
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_image_upload(image_path: str = None):
    """Test image upload and analysis"""
    print("\n" + "=" * 60)
    print("Testing Image Upload & Analysis...")
    print("=" * 60)
    
    # Create a simple test image if none provided
    if not image_path:
        print("⚠️  No image provided, creating a test payload...")
        # Use a minimal base64 encoded 1x1 pixel image for testing
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    else:
        print(f"📁 Reading image from: {image_path}")
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                test_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                print(f"✅ Image loaded: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"❌ ERROR reading image: {str(e)}")
            return False
    
    # Prepare request
    data = {
        "image_data": test_image_b64,
        "filename": "test_image.jpg",
        "animal_id": "TEST_001",
        "include_breed_classification": True,
        "include_disease_detection": True,
        "include_measurements": True
    }
    
    print(f"📦 Request payload size: {len(json.dumps(data))} bytes")
    print(f"📡 Sending POST request to: {API_BASE_URL}/analyze")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=data,
            timeout=60
        )
        
        print(f"\n📨 Response Status: {response.status_code}")
        print(f"📨 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS! Analysis completed")
            print(f"\n📊 Response Structure:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"\n❌ FAILED with status {response.status_code}")
            print(f"Error Response: {response.text}")
            try:
                error_json = response.json()
                print(f"Error JSON: {json.dumps(error_json, indent=2)}")
            except:
                pass
            return False
            
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timeout (>60 seconds)")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Connection failed")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "🔬" * 30)
    print("AutoATC Image Upload Diagnostic Tool")
    print("🔬" * 30 + "\n")
    
    # Test 1: Backend health
    if not test_backend_health():
        print("\n⚠️  Backend is not running. Please start it first.")
        return
    
    # Test 2: API status
    test_api_status()
    
    # Test 3: Image upload
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    test_image_upload(image_path)
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

