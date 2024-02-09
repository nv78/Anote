import pytest

def run_all_tests():
    # Run your pytest test cases here
    pytest.main(['test_upload.py'])  # Replace 'test_evaluate.py' with your test file names
    pytest.main(['test_evaluate.py'])  # Add more test files as needed
    pytest.main(['test_sentiment_analysis.py'])  # Add more test files as needed

# if __name__ == "__main__":
#     run_all_tests()