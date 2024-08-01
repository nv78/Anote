import pandas as pd
from  QAFineTune import QuestionAnsweringJSON  # replace your_script_name with the actual name of your Python file without the .py extension
import json
import os

def test_QuestionAnsweringJSON():
    # Create a sample dataframe
    data = pd.DataFrame({
        'question': ['What is the capital of France?'],
        'evidence_text': ['The capital of France is Paris.'],
        'answer': ['Paris']
    })

    # Define a temporary filename for the test
    filename = 'temp_test_file.jsonl'

    # Run the function with the sample data
    QuestionAnsweringJSON(data, filename)

    # Read the generated file and check its contents
    with open(filename, 'r') as file:
        lines = file.readlines()
        assert len(lines) == 1  # Ensure there is one line per entry in the dataframe
        conversation = json.loads(lines[0])
        assert 'messages' in conversation  # Check if 'messages' key exists
        assert conversation['messages'][1]['content'] == "What is the capital of France? based on The capital of France is Paris.  "
        assert conversation['messages'][2]['content'] == "Paris"

    # Clean up: remove the temporary file after the test
    os.remove(filename)

