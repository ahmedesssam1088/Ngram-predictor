import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Avoid repeated downloads if resources already exist
for package in ['punkt', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package)

class Normalizer:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def load(self, folder_path):
        combined_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                    clean_content = self.strip_gutenberg(raw_content)
                    combined_text += clean_content + "\n"
        return combined_text

    def strip_gutenberg(self, text):
        start_pattern = re.compile(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE)
        end_pattern = re.compile(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE)
    
        start_match = start_pattern.search(text)
        end_match = end_pattern.search(text)

        start_idx = start_match.end() if start_match else 0
        end_idx = end_match.start() if end_match else len(text)
        return text[start_idx:end_idx]

    def lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]|_', '', text)

    def remove_numbers(self,text):
        pattern = r'\b(\d+|[ivxlcm]+)(?=[\.\-\)])'
        cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d+', '', cleaned)
        return cleaned.strip()

    def remove_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def normalize(self, text):
        text = self.lowercase(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def word_tokenize(self, sentence):
        """
        Splits sentence into words and converts them to root form (e.g., walking -> walk).
        """
        # 1. Standard tokenization
        tokens = word_tokenize(sentence)
    
        # 2. Lemmatize each word to its 'verb' root (pos='v')
        # We also use .lower() to ensure 'Walking' and 'walking' are the same
        return [self.lemmatizer.lemmatize(token.lower(), pos='v') for token in tokens]


    def save(self, sentences, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence_words in sentences:
                line = " ".join(sentence_words)
                f.write(line + "\n")


def main():
    print("Testing the Normalizer on a 100-sentence sample...")    
    raw_dir = "data/raw/train/" 
    sample_out = "data/processed/train_tokens_sample.txt"

    norm = Normalizer()
    
    # 3. Load the raw text and cleaning it
    try:
        raw_text = norm.load(raw_dir)
    except FileNotFoundError:
        print(f"Error: Could not find '{raw_dir}'. Did you download the books and put them in the right folder?")
        return
        
    if not raw_text:
        print("Error: The text loaded is empty. Check your .txt files.")
        return

    
    # 4. Sentence Tokenize
    print("Splitting into sentences...")
    # This might take a few seconds on large books!
    sentences = norm.sentence_tokenize(raw_text)
    
    # 5. Slice the first 100 sentences
    sample_sentences = sentences[:100]
    print(f"Processing a sample of {len(sample_sentences)} sentences...")
    
    # 6. Normalize and word-tokenize each sentence
    processed_sentences = []
    for sent in sample_sentences:
        cleaned_sent = norm.normalize(sent)
        
        # Only tokenize and save if the sentence isn't blank after cleaning
        if cleaned_sent.strip(): 
            words = norm.word_tokenize(cleaned_sent)
            processed_sentences.append(words)
            
    # 7. Save the sample
    print(f"Saving tokenized sample to {sample_out}...")
    norm.save(processed_sentences, sample_out)
    print("Done! Open 'data/processed/train_tokens_sample.txt' to check your results.")

if __name__ == "__main__":     
    main()