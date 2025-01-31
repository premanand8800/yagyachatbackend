import requests
import json

def test_api(input_text):
    # API endpoint URL
    url = "http://127.0.0.1:8000/api/process_input"

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make the POST request
        response = requests.post(
            url,
            params={"input_text": input_text},
            headers=headers
        )
        
        # Print the response
        print(f"Status Code: {response.status_code}")
        print("Response Headers:", response.headers)
        print("Raw Response:", response.text)
        
        if response.status_code == 200:
            try:
                print("\nParsed Response:")
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError as e:
                print(f"\nError parsing JSON response: {e}")
        else:
            print(f"\nError Response: {response.text}")
        
        return response
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the FastAPI server is running.")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Use a test input
    test_input = "Hello, I need help with my project. My background is in software development and my goals are to improve code quality."
    
    # Test the API
    test_api(test_input)
