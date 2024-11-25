# create output dir if not exists
import os
import re
import sys
# generate six-character uuid as file basename
import uuid

import nltk
import pandas as pd

from collections import Counter
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

def transform_to_google_sheets_format(text):
    lines = text.split('\n')
    responses = {}
    current_response = None

    for line in lines:
        print(line)
        if line.startswith("Words") or line.startswith("Response"):
            continue

        match = re.match(r'^Response (\d+)$', line)
        if match:
            current_response = int(match.group(1))
            responses[current_response] = ""  # Initialize with an empty string
        elif current_response is not None:
            # Always add a space before appending, handling empty lines correctly
            responses[current_response] += (" " if responses[current_response] else "") + line.strip()
            print(responses[current_response])
    google_sheet_format = "Response #\tText\n"
    for i in range(1, 17):
        google_sheet_format += f"{i}\t{responses.get(i, '')}\n"  # Use .get() to handle missing responses

    return google_sheet_format


def transform_response_text_into_list(text):
    # break string into list of strings broken at each "\nResponse"
    responses = re.split(r'\nResponse \d+', text)
    # remove empty strings from list
    responses = [response.strip() for response in responses if response.strip()]
    return responses


def most_frequent_words(text, n=1):
    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Remove punctuation and convert to lower case
    text = re.sub(r'[^\w\s]', '', text).lower()

    # Split text into words
    words = text.split()

    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Count words
    word_counts = Counter(filtered_words)

    # Return the n most common words
    most_common = word_counts.most_common(n)
    return [word for word, count in most_common]

if __name__ == "__main__":
    input_dir = "input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    import_file = sys.argv[1]
    import_file_path = os.path.join(input_dir, import_file)
    with open(import_file_path, "r") as f:
        text = f.read()
    output_dir = "output"


    list_of_responses = transform_response_text_into_list(text)
    # remove lines beginning with "Response X" and "Words" from each item in the list
    cleaned_responses = []
    for response in list_of_responses:
        lines = response.splitlines()
        cleaned_lines = [line for line in lines if
                         not line.startswith("Words") and not line.startswith("Characters") and not line.startswith(
                             "Paragraphs")]
        cleaned_responses.append("\n".join(cleaned_lines))

    response_df = pd.DataFrame(cleaned_responses)
    print(response_df)

    most_frequent = response_df.applymap(lambda x: most_frequent_words(x, n=3) if pd.notna(x) else [])
    print(most_frequent)

    # save to file
    file_basename = str(uuid.uuid4())[:8] + ".csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, file_basename)
    response_df.to_csv(file_basename)




