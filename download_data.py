import urllib.request
import os

# Define the books and their Project Gutenberg IDs
books = {
    "adventures.txt": "1661",
    "memoirs.txt": "834",
    "return.txt": "108",
    "hound.txt": "2852"
}

# The path where the data should live
target_dir = "data/raw/train/"

# Ensure the directory exists
os.makedirs(target_dir, exist_ok=True)

for name, book_id in books.items():
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, os.path.join(target_dir, name))
        print(f"Successfully saved to {target_dir}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

print("\nAll done! Your raw data is ready.")