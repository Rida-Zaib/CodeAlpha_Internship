"""
Downloads the NLTK data this project needs into ./nltk_data (next to this file).

    python setup_nltk.py

Use it as part of the BUILD command on a hosting platform, so the data ends up
inside the project folder and is there when the server starts:

    pip install -r requirements.txt && python setup_nltk.py

On your own computer you can still use nltk.download() as before; the app finds
the data in either place. This script never fails the build: if a download does
not work the app still starts, using a simple fallback, and says so in its log.
"""

import os

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
# punkt_tab is what newer NLTK versions use, punkt is for older ones.
PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def main():
    try:
        import nltk
    except ImportError:
        print("NLTK is not installed. Run: pip install -r requirements.txt")
        return

    os.makedirs(TARGET, exist_ok=True)
    for package in PACKAGES:
        ok = nltk.download(package, download_dir=TARGET, quiet=True)
        print(f"{'ok' if ok else 'not available':14} {package}")
    print(f"NLTK data folder: {TARGET}")


if __name__ == "__main__":
    main()