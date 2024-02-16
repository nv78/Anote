import pytest
import os
import sys

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(grandparent_dir)
print(grandparent_dir)



def run_all_tests():
    # Run your pytest test cases here
    pytest.main(['./test_functions/test_upload.py'])
    pytest.main(['./test_functions/test_evaluate.py'])
    pytest.main(['./test_functions/test_ner.py'])
    pytest.main(['./test_functions/test_classify.py'])
    pytest.main(['./test_functions/test_rag.py'])
    pytest.main(['./test_functions/test_finetuning.py'])
    pytest.main(['./test_functions/test_answer.py'])

if __name__ == "__main__":
    run_all_tests()