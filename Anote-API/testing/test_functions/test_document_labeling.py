import pytest
from unittest.mock import patch
from anote import upload, predict

# Mock the upload function to avoid actual file uploads
@pytest.fixture
def mock_upload():
    with patch("anote.upload") as mock:
        yield mock

# Test Zero-Shot Document Labeling
def test_document_labeling(mock_upload):
    API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"
    directory_path = '/data/doclabeling/'

    # Mock the return values for upload
    mock_upload.side_effect = [
        "Sample document 1",
        "Sample document 2",
        "Sample document 3",
    ]

    CATEGORIES = [
        "Contracts",
        "Regulatory",
        "Litigation",
        "Legal Opinions"
    ]

    DOCUMENTS = ["file1.txt", "file2.txt", "file3.txt"]  # Replace with actual file names

    PREDICTIONS = []
    for document in DOCUMENTS:
        text = upload(file=document, decomposition="PER_DOCUMENT")
        PREDICTIONS.append(predict(categories=CATEGORIES, text=text))

    # Assert the predictions match the expected output
    expected_predictions = ["Litigation", "Regulatory", "Legal Opinions"]
    assert PREDICTIONS == expected_predictions
