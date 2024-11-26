import os
import re
import sys
import uuid
import argparse
import streamlit as st
from openai import OpenAI
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

xai_api_key = os.getenv('xAI_API_KEY')
base_url = "https://api.x.ai/v1"

todays_edition = None

if xai_api_key is None:
    st.error("You must provide an API key from xAI.")
if not xai_api_key.startswith("xai"):
    st.error("That doesn't look like an API key from xAI.")


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
    responses = re.split(r'\nResponse \d+', text)
    responses = [response.strip() for response in responses if response.strip()]
    return responses


def most_frequent_words(text, n=1):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    most_common = word_counts.most_common(n)
    return [word for word, count in most_common]


def call_xai_api(prompt, model="grok-beta"):
    client = OpenAI(
        api_key=xai_api_key,
        base_url=base_url,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content


def main(args):
    input_dir = "input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    import_file = args.input
    import_file_path = os.path.join(input_dir, import_file)
    with open(import_file_path, "r") as f:
        text = f.read()
    output_dir = "output"

    list_of_responses = transform_response_text_into_list(text)
    cleaned_responses = []
    for i, response in enumerate(list_of_responses):
        lines = response.splitlines()
        cleaned_lines = [line for line in lines if
                         not line.startswith("Words") and not line.startswith("Characters") and not line.startswith(
                             "Paragraphs") and not line.startswith("Response ")]
        cleaned_responses.append(
            {"Response": f"Response {i + 1}", "Text": "\n".join(cleaned_lines)})

    response_df = pd.DataFrame(cleaned_responses)
    print(response_df)

    if args.use_xai:
        api_responses = []
        for index, row in response_df.iterrows():
            prompt = f"In fifteen words or less, identify the key weaknesses in this text: {row['Text']}"
            api_response = call_xai_api(prompt)
            print(api_response)
            api_responses.append(api_response)
        response_df['API Response'] = api_responses

    most_frequent = response_df.apply(
        lambda row: most_frequent_words(row['Text'], n=3) if pd.notna(row['Text']) else [], axis=1)
    response_df['Most Frequent Words'] = most_frequent
    print(most_frequent)

    file_basename = str(uuid.uuid4())[:8] + ".csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, file_basename)
    response_df.to_csv(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text responses and optionally use xAI API.')
    parser.add_argument('--input', required=True, help='Path to the input file')
    parser.add_argument('--use_xai', action='store_true', help='Flag to enable xAI API for text analysis')
    args = parser.parse_args()

    main(args)