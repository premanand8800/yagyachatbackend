import os
import asyncio
from groq import Groq
from nemoguardrails import LLMRails, RailsConfig

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Function to create NeMo Guardrails configuration
def create_nemo_config():
    # Ensure the config directory exists
    config_path = os.path.join(os.path.dirname(__file__), "config")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config directory not found at {config_path}.")
    return RailsConfig.from_path(config_path)

# Function to process chat input
async def process_chat(user_input):
    # Initialize NeMo Guardrails
    rails = LLMRails(create_nemo_config())

    # Apply NeMo Guardrails to the input
    guarded_input = await rails.generate_async(prompt=user_input)
    if guarded_input == "I'm sorry, I can't answer that question.":
        return guarded_input

    # Get response from Groq
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # Use a Groq model
        messages=[{"role": "user", "content": user_input}]
    )
    groq_response = response.choices[0].message.content

    # Apply NeMo Guardrails to the output
    guarded_output = await rails.generate_async(prompt=groq_response)
    return guarded_output

# Main function to run the chatbot
async def main():
    print("Welcome to the Groq Chatbot with NeMo Guardrails! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Process the user input
        result = await process_chat(user_input)
        print("Bot:", result)

# Run the chatbot
if __name__ == "__main__":
    asyncio.run(main())