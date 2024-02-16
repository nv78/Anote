import pytest
import os 
import sys

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(grandparent_dir)
print(grandparent_dir)



def run_all_tests():
    # Run your pytest test cases here
    #pytest.main(['test_upload.py'])  # Replace 'test_evaluate.py' with your test file names
    #pytest.main(['test_evaluate.py'])  # Add more test files as needed
    #pytest.main(['test_sentiment_analysis.py'])  # Add more test files as needed
    pytest.main(['test_answer.py'])



if __name__ == "__main__":
    run_all_tests()