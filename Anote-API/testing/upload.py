from tika import parser as p
import requests
import pandas as pd
from io import StringIO


def upload(
    file,
    local: bool = True,
    document: bool = True,
    alreadyStructured: bool = False,
    hasHeader: bool = False,
    inputColIndex: int = 0,
    isHTML: bool = False
):
    """
    Decomposes docx, pdfs, images, hmtls using apache tika

    Args:
        local (bool):
            if local is true, gets filepath from local directory
            if local is false, gets filepath from online URL
        filepath (str):
            if local is false, can get from a web url
            if local is true, can get local path to file
        document (bool):
            if document is true, extract all text from the document
            if document is false, break down on a paragraph specific basis
    """
    if document == True:
        if isHTML == True:
            return _get_text_from_url(file)
        # Document based decomposition
        if local == True:
            # LOCAL DIRECTORY
            result = p.from_buffer(file)
            text = result["content"].strip()
            return text
        if local == False:
            # WEB URL
            response = requests.get(file)
            result = p.from_buffer(response.content)
            text = result["content"].strip()
            return text
    if document == False:
        # Paragraph based decomposition
        if local == True:
            if alreadyStructured == True:
                s=str(file,'utf-8')
                data = StringIO(s)
                df = pd.read_csv(data, header=None)
                # Drop header if exists.
                if hasHeader == True:
                    df = df.iloc[1: , :]
                df = df.iloc[: , inputColIndex]
                return df.values.tolist()
            else:
                # LOCAL DIRECTORY
                result = p.from_buffer(file)
                text_list = _tokenize(result["content"])
                while("" in text_list):
                    text_list.remove("")
                clean_text_list = []
                index = []
                for i, row in enumerate(text_list):
                    index.append(i)
                    # Get preprocessor to work.
                    # row = preprocessor(row)
                    clean_text_list.append(row)
                return clean_text_list
        if local == False:
            # WEB URL
            response = requests.get(file)
            result = p.from_buffer(response.content)
            text_list = _tokenize(result["content"])
            while("" in text_list):
                text_list.remove("")
            return text_list
        
def _tokenize(
    content: str,
):
    return content.split('\n')

def _parse_actual_labels_from_csv(
    fileBytes,
    hasHeader,
    labelColIndex,
):
    s=str(fileBytes,'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data, header=None, keep_default_na=False)
    # Drop header if exists.
    if hasHeader:
        df = df.iloc[1: , :]
    df = df.iloc[: , labelColIndex]
    return df.values.tolist()

def _get_text_from_url(
    web_url
):
    response = requests.get(web_url)
    result = p.from_buffer(response.content)
    text = result["content"].strip()
    text = text.replace("\n", "").replace("\t", "")
    #text = "".join(text)
    return text