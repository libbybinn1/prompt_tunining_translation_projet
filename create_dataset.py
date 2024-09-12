from transformers import MarianMTModel, MarianTokenizer
import string

# Load the Helsinki-NLP models for Hebrew to English and English to Hebrew
heb_to_eng_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
eng_to_heb_model_name = 'Helsinki-NLP/opus-mt-en-he'

heb_to_eng_tokenizer = MarianTokenizer.from_pretrained(heb_to_eng_model_name)
heb_to_eng_model = MarianMTModel.from_pretrained(heb_to_eng_model_name)

eng_to_heb_tokenizer = MarianTokenizer.from_pretrained(eng_to_heb_model_name)
eng_to_heb_model = MarianMTModel.from_pretrained(eng_to_heb_model_name)

def translate(text, tokenizer, model, num_return_sequences=1, num_beams=1):
    """Translate text using a specific tokenizer and model with an option to return multiple translations."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(**inputs, num_return_sequences=num_return_sequences, num_beams=num_beams)
        # Return all translations if multiple sequences are requested
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        return translated_texts
    except Exception as e:
        print(f"Translation error: {e}")
        return [""]

# def translate(text, tokenizer, model, num_return_sequences=1):
#     """Translate text using a specific tokenizer and model with an option to return multiple translations."""
#     try:
#         inputs = tokenizer(text, return_tensors="pt", padding=True)
#         translated_tokens = model.generate(**inputs, num_return_sequences=num_return_sequences, num_beams=5)
#         # Return all translations if multiple sequences are requested
#         translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
#         return translated_texts
#     except Exception as e:
#         print(f"Translation error: {e}")
#         return [""]


def preprocess_text(text):
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def chunk_text(text, n):
    """Splits the text into chunks of n words."""
    words = text.split()
    return [words[i:i + n] for i in range(0, len(words), n)]


def check_back_translation(english_translation, hebrew_last_word, eng_to_heb_tokenizer, eng_to_heb_model):
    """Check if the last word, two words, or three words in the English translation match the Hebrew last word."""

    english_words = english_translation.split()
    last_english_word = english_words[-1]

    # Translate the last English word back to Hebrew
    back_to_hebrew_last_word = translate(last_english_word, eng_to_heb_tokenizer, eng_to_heb_model)[0]

    # First check: Compare the last word
    if preprocess_text(back_to_hebrew_last_word) == preprocess_text(hebrew_last_word):
        return True, "Match found with the last word.", english_translation

    # Second check: Compare the last two words
    last_two_english_words = ' '.join(english_words[-2:])
    back_to_hebrew_last_two_words = translate(last_two_english_words, eng_to_heb_tokenizer, eng_to_heb_model)[0]
    if preprocess_text(back_to_hebrew_last_two_words) == preprocess_text(hebrew_last_word):
        return True, "Match found using 2 last English words.", english_translation

    # Third check: Compare the last three words
    last_three_english_words = ' '.join(english_words[-3:])
    back_to_hebrew_last_three_words = translate(last_three_english_words, eng_to_heb_tokenizer, eng_to_heb_model)[0]
    if preprocess_text(back_to_hebrew_last_three_words) == preprocess_text(hebrew_last_word):
        return True, "Match found using 3 last English words.", english_translation

    # If none of the checks pass, return the mismatch
    mismatch_message = (f"Mismatch found:\n"
                        f"the English sentence was: {english_translation}\n"
                        f"Original Hebrew word: {hebrew_last_word}\n"
                        f"Back to Hebrew using the last English word: {back_to_hebrew_last_word}\n"
                        f"Back to Hebrew using 2 last English words: {back_to_hebrew_last_two_words}\n"
                        f"Back to Hebrew using 3 last English words: {back_to_hebrew_last_three_words}")

    # Append mismatch details
    english_translation += f" (regional word: {hebrew_last_word}, got: {back_to_hebrew_last_word}, {back_to_hebrew_last_two_words}, {back_to_hebrew_last_three_words})"
    return False, mismatch_message, english_translation

def process_translation(input_file="dataset.txt", output_file="translated_output.txt"):
    # Read the input text from the dataset.txt file
    with open(input_file, "r", encoding="utf-8") as infile:
        text = infile.read()

    # Preprocess the Hebrew text (removes punctuation)
    chunks = chunk_text(preprocess_text(text), 5)

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            hebrew_chunk = ' '.join(chunk)
            print(f"Processing chunk {idx + 1}/{len(chunks)}: {hebrew_chunk}")

            # Translate the whole chunk to English (initial translation with beam search)
            english_translations = translate(hebrew_chunk, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1, num_beams=5)
            english_translation = english_translations[0]

            if not english_translation:
                print("Skipping due to translation failure.")
                continue

            # Get the last word in the Hebrew chunk
            hebrew_last_word = chunk[-1]

            # Check back-translation for matches or mismatches
            is_match, message, final_translation = check_back_translation(english_translation, hebrew_last_word, eng_to_heb_tokenizer, eng_to_heb_model)

            # If a mismatch is found, try higher beam search
            if not is_match:
                print(message)  # Log the first mismatch
                print("Retrying with higher beams...")

                # Retry with higher beam search for better translation
                alternative_translations = translate(hebrew_chunk, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1, num_beams=10)
                alternative_translation = alternative_translations[0]

                is_match, alt_message, alt_final_translation = check_back_translation(alternative_translation, hebrew_last_word, eng_to_heb_tokenizer, eng_to_heb_model)
                if is_match:
                    print("Alternative translation matched!")
                    final_translation = alt_final_translation
                    message = alt_message
                else:
                    print("No match found even after retrying.")

            # Log and write the result based on match or mismatch
            print(message)
            f.write(f"{final_translation}\n")



# def process_translation(input_file="dataset.txt", output_file="translated_output.txt"):
#     # Read the input text from the dataset.txt file
#     with open(input_file, "r", encoding="utf-8") as infile:
#         text = infile.read()
#
#     # Preprocess the Hebrew text (removes punctuation)
#     chunks = chunk_text(preprocess_text(text), 5)
#
#     with open(output_file, "w", encoding="utf-8") as f:
#         for idx, chunk in enumerate(chunks):
#             hebrew_chunk = ' '.join(chunk)
#             print(f"Processing chunk {idx + 1}/{len(chunks)}: {hebrew_chunk}")
#
#             # Translate the whole chunk to English (initial translation)
#             english_translations = translate(hebrew_chunk, heb_to_eng_tokenizer, heb_to_eng_model,
#                                              num_return_sequences=1)
#             english_translation = english_translations[0]
#
#             if not english_translation:
#                 print("Skipping due to translation failure.")
#                 continue
#
#             # Get the last word in the Hebrew chunk
#             hebrew_last_word = chunk[-1]
#
#             # Check back-translation for matches or mismatches
#             is_match, message, final_translation = check_back_translation(english_translation, hebrew_last_word,
#                                                                           eng_to_heb_tokenizer, eng_to_heb_model)
#
#             # If a mismatch is found, try alternative translations (num_return_sequences > 1)
#             if not is_match:
#                 print(message)  # Log the first mismatch
#                 print("Trying alternative translations...")
#
#                 # Generate alternative translations
#                 alternative_translations = translate(hebrew_chunk, heb_to_eng_tokenizer, heb_to_eng_model,
#                                                      num_return_sequences=3)
#
#                 for alt_translation in alternative_translations[1:]:  # Skip the first, since we already tried it
#                     is_match, alt_message, alt_final_translation = check_back_translation(alt_translation,
#                                                                                           hebrew_last_word,
#                                                                                           eng_to_heb_tokenizer,
#                                                                                           eng_to_heb_model)
#                     if is_match:
#                         print("Alternative translation matched!")
#                         final_translation = alt_final_translation
#                         message = alt_message
#                         break
#
#             # Log and write the result based on match or mismatch
#             print(message)
#             f.write(f"{final_translation}\n")


# Example usage: Process text from "dataset.txt" and save results to "translated_output.txt"
process_translation(input_file="dataset.txt", output_file="translated_output__using_beam.txt")
