import requests
import unittest

class TestRishiAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8001"

    def test_root_endpoint(self):
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Rishi API is running"})

    def test_process_input_endpoint(self):
        # Test basic input
        data = {"input_text": "Hello, world!"}
        response = requests.post(f"{self.BASE_URL}/api/process_input", json=data)
        print(f"Response: {response.text}")  # Add debug print
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["input_length"], len(data["input_text"]))
        self.assertFalse(result["has_background"])
        self.assertFalse(result["has_goals"])

        # Test input with keywords
        data = {"input_text": "My background includes Python and my goals are to learn FastAPI"}
        response = requests.post(f"{self.BASE_URL}/api/process_input", json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["has_background"])
        self.assertTrue(result["has_goals"])

    def test_invalid_input(self):
        # Test missing required field
        data = {}
        response = requests.post(f"{self.BASE_URL}/api/process_input", json=data)
        self.assertEqual(response.status_code, 422)  # Validation error

if __name__ == '__main__':
    unittest.main()
