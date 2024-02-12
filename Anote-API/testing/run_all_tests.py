import pytest

def run_all_tests():
    # Run your pytest test cases here
    pytest.main(['./test_functions/test_upload.py'])
    pytest.main(['./test_functions/test_evaluate.py'])
    pytest.main(['./test_functions/test_classify.py'])
    pytest.main(['./test_functions/test_ner.py'])
    pytest.main(['./test_functions/test_answer.py'])
    pytest.main(['./test_functions/test_rag.py'])
    pytest.main(['./test_functions/test_finetuning.py'])

if __name__ == "__main__":
    run_all_tests()