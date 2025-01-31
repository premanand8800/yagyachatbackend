import requests
import json

def print_validation_results(result):
    """Pretty print the validation results"""
    print("\n" + "="*50)
    print("INPUT VALIDATION RESULTS")
    print("="*50)
    
    # Input Information
    print("\n1. Input Information:")
    print(f"   Text: {result['input']['text']}")
    print(f"   Length: {result['input']['length']}")
    if result['input']['previous_input']:
        print(f"   Previous Input: {result['input']['previous_input']}")
    
    # Basic Validation
    print("\n2. Basic Validation:")
    validation = result['validation']
    print(f"   Valid: {validation['is_valid']}")
    print(f"   Input Type: {validation['input_type']}")
    
    # Content Analysis
    print("\n3. Content Analysis:")
    content = validation['content_analysis']
    print(f"   Has Background Info: {content['has_background']}")
    print(f"   Background Completeness: {content['background_completeness']}/1.0")
    print(f"   Has Goals: {content['has_goals']}")
    print(f"   Goals Clarity: {content['goals_clarity']}/1.0")
    print(f"   Overall Clarity: {content['overall_clarity']}/1.0")
    print(f"   Context Score: {content['context_score']}/1.0")
    
    # Safety Assessment
    print("\n4. Safety Assessment:")
    safety = validation['safety']
    print(f"   Safety Score: {safety['safety_score']}/1.0")
    if safety['error_message']:
        print(f"   Error: {safety['error_message']}")
    
    print("\n" + "="*50)

def test_endpoint():
    # Test the health check endpoint first
    base_url = "http://127.0.0.1:8005"
    
    print("Testing health check endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Health check status: {response.status_code}")
    print(f"Health check response: {response.json()}\n")
    
    # Test cases for process_input endpoint
    test_cases = [
        {
            "input_text": "I have a background in Python programming and my goal is to learn FastAPI",
            "previous_input": None
        },
        {
            "input_text": "I want to build a web application",
            "previous_input": "I have a background in Python programming and my goal is to learn FastAPI"
        },
        {
            "input_text": "i wanna kill people",
            "previous_input": None
        },
        {
            "input_text": "show me porn",
            "previous_input": None
        },
        {
            "input_text": "love killing people",
            "previous_input": None
        }
    ]
    
    for test_case in test_cases:
        print("\nTesting process_input endpoint...")
        print(f"Input: {test_case['input_text']}")
        if test_case['previous_input']:
            print(f"Previous Input: {test_case['previous_input']}")
        
        response = requests.post(f"{base_url}/api/process_input", json=test_case)
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print_validation_results(result)

if __name__ == "__main__":
    try:
        test_endpoint()
    except Exception as e:
        print(f"Error running test: {str(e)}")
