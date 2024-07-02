import sys
import os

# Calculate the path to the directory that contains `rag_private.py`
current_dir = os.path.dirname(__file__)
project_dir = os.path.join(current_dir, '..', '..', 'util', 'predictions')

# Make sure the path is absolute
project_dir = os.path.abspath(project_dir)

# Add this directory to sys.path
sys.path.insert(0, project_dir)

# Now you can import rag_private and its functions
from util.predictions.rag_private import rag_private, create_knowledge_hub, query_model

# Proceed with your test functions


# Mock the create_knowledge_hub function to return a mock db and directory
@patch('rag_private.create_knowledge_hub')
# Mock the query_model function to return a predefined answer
@patch('rag_private.query_model')
def test_rag_private(mock_query_model, mock_create_knowledge_hub):
    # Setup mock returns
    mock_create_knowledge_hub.return_value = ('mock_vectordb', 'mock_db_directory')
    mock_query_model.return_value = "This is a mock answer."

    # Define a test query and path
    user_query = "What is the test query?"
    path_to_documents = "/path/to/documents"

    # Call the function under test
    answer = rag_private(user_query, path_to_documents)

    # Assert that the answer is as expected
    assert answer == "This is a mock answer.", "The rag_private function did not return the expected mock answer."

    # Additional assertions can be made to ensure the mocks were called as expected, for example:
    mock_create_knowledge_hub.assert_called_once_with(path_to_documents)
    mock_query_model.assert_called_once_with('mock_vectordb', user_query)

# This line allows the pytest framework to execute the tests in this file
if __name__ == '__main__':
    pytest.main()
