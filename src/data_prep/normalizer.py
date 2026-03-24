import os
import re
import nltk


class Normalizer:

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
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def normalize(self, text):
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        print("AE sentence_tokenize function called");
        """
        Splits text into a list of sentences.
        
        Args:
            text (str): The normalized text.
            
        Returns:
            list: A list of sentence strings.
        """
        return nltk.sent_tokenize(text)

    def word_tokenize(self, sentence):
        print("AE word_tokenize function called");
        """
        Splits a single sentence into a list of tokens (words).
        
        Args:
            sentence (str): A single sentence string.
            
        Returns:
            list: A list of word tokens.
        """
        return nltk.word_tokenize(sentence)

    def save(self, sentences, filepath):
        print("AE save function called");
        """
        Writes tokenized sentences to an output file, one sentence per line.
        
        Args:
            sentences (list): A list of sentences (where each sentence is a list of words).
            filepath (str): The path to save the output file.
        """
        # Make sure the directory exists before saving
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence_words in sentences:
                # Join the words with a single space [cite: 81]
                line = " ".join(sentence_words)
                f.write(line + "\n")

# Every module file must have a runnable entry point for testing during development [cite: 24]
#if __name__ == "__main__":
#    print("Testing the Normalizer...")
    # We will test this shortly!

def main():
    print("Testing the Normalizer on a 100-sentence sample...")
    
    # 1. Define paths (assuming you run this from the project root folder)
    raw_dir = "data/raw/train/" 
    sample_out = "data/processed/train_tokens_sample.txt"
    
    # 2. Instantiate our class
    norm = Normalizer()
    
    # 3. Load the raw text
    try:
        print(f"Loading texts from {raw_dir}...")
        raw_text = norm.load(raw_dir)
        print(raw_text[:500])  # Print the first 500 characters to verify loading
    except FileNotFoundError:
        print(f"Error: Could not find '{raw_dir}'. Did you download the books and put them in the right folder?")
        return
        
    if not raw_text:
        print("Error: The text loaded is empty. Check your .txt files.")
        return

    # 4. Clean Gutenberg headers
    print("Stripping Gutenberg headers...")
    text_no_headers = norm.strip_gutenberg(raw_text)
    
    # 5. Sentence Tokenize
    print("Splitting into sentences...")
    # This might take a few seconds on large books!
    sentences = norm.sentence_tokenize(text_no_headers)
    
    # 6. Slice the first 100 sentences
    sample_sentences = sentences[:100]
    print(f"Processing a sample of {len(sample_sentences)} sentences...")
    
    # 7. Normalize and word-tokenize each sentence
    processed_sentences = []
    for sent in sample_sentences:
        cleaned_sent = norm.normalize(sent)
        
        # Only tokenize and save if the sentence isn't blank after cleaning
        if cleaned_sent.strip(): 
            words = norm.word_tokenize(cleaned_sent)
            processed_sentences.append(words)
            
    # 8. Save the sample
    print(f"Saving tokenized sample to {sample_out}...")
    norm.save(processed_sentences, sample_out)
    print("Done! Open 'data/processed/train_tokens_sample.txt' to check your results.")

if __name__ == "__main__":
    # Ensure nltk dependencies are downloaded for a first-time run
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    main()