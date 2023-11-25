import os

assert 0 == os.system(f"coverage run -m unittest discover -v")
