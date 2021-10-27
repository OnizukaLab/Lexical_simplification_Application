import zipfile
with zipfile.ZipFile("./crawl-300d-2M-subword.zip") as existing_zip:
    existing_zip.extractall("./crawl-300d-2M-subword/")