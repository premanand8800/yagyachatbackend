# app/tests/interactive_test.py
import asyncio
import json
from app.nodes.validation import (
    create_validation_workflow,
    GraphState,
    UserInput
)

async def main():
    """Main function for interactive testing."""
    # Create the validation workflow
    workflow = create_validation_workflow()
    
    while True:
        # Get user input from the terminal
        user_input = input("Enter your input (or 'exit' to quit): ")
        
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        # Initialize state
        initial_state = GraphState(
            user_input=user_input,
            messages=[],
            next_step=""
        )
        
        # Process input through the validation node
        final_state = await workflow.ainvoke(initial_state)
        
        # Print the output
        print(json.dumps(final_state, indent=2))
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())