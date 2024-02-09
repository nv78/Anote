from tika import parser as p

def upload(file, decomposition):
    if decomposition is "PER_DOCUMENT":
        result = p.from_buffer(file)
        text = result["content"].strip()
        return text
    else:
        #change this code to be per line
        result = p.from_buffer(file)
        text = result["content"].strip()
        return text


