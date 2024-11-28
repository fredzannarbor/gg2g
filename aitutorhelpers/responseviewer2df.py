import io
import os
import re
import sys
import uuid
import argparse
import streamlit as st
from openai import OpenAI
import nltk
import pandas as pd
from collections import Counter, OrderedDict
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

xai_api_key = os.getenv('xAI_API_KEY')
base_url = "https://api.x.ai/v1"

todays_edition = None

if xai_api_key is None:
    st.error("You must provide an API key from xAI.")
if not xai_api_key.startswith("xai"):
    st.error("That doesn't look like an API key from xAI.")

st.set_page_config("Responses Viewer", "⑯", layout="wide")
st.title("Responses Viewer ⑯")
st.markdown("_Deterministic Text Analytics for Response Evaluation_")
if 'notes' not in st.session_state:
    st.session_state.notes = {}
def transform_to_google_sheets_format(text):
    lines = text.split('\n')
    responses = {}
    current_response = None

    for line in lines:
        print(line)
        if line.startswith("Words") or line.startswith("Response") or line.startswith("All Responses"):
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
    responses = [response.strip() for response in responses if response.strip() and not response.strip().startswith("All Responses")]
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


def create_concordance(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    concordance = {}
    for word in filtered_tokens:
        if word not in concordance:
            concordance[word] = []
        # Here, we're just adding the position; in a real concordance, you'd want context
        concordance[word].append(filtered_tokens.index(word))

    return concordance


def searchable_index(index):
    word_to_search = st.text_input("Search for a word in the index:")
    if word_to_search:
        search_term = word_to_search.lower()
        if any(search_term == key.lower() for key in index):
            matching_key = next(key for key in index if key.lower() == search_term)
            # Consolidate identical response numbers
            locations = sorted(set(index[matching_key]))  # Remove duplicates and sort
            locations_str = ', '.join(map(str, locations))
            st.write(f"response **{word_to_search}** found in responses {locations_str}")
        else:
            st.write(f"**{word_to_search}** not found in the index.")
    else:
        st.write("Enter a word to search for in the index.")

def show_dataframe_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


def display_4x4_grid(responses=None):
    if responses is None:
        # Initialize responses if not provided
        responses_with_notes = {f'response_{i}': '' for i in range(1, 17)}
    responses_with_notes = responses
    # Create 4 columns
    cols = st.columns(4)
    with st.form("16-responses"):
        # Display responses in a 4x4 grid
        for i, col in enumerate(cols):
            for j in range(4):
                index = i * 4 + j
                index_show = index + 1
                key = f'response_{index}'
                responses_with_notes[index] = col.text_area(
                    f"Response {index_show}",
                    value="",
                    height=150,
                    key=key  # Use a unique key for each text_area
                )
        if st.form_submit_button('Save All My Notes'):
            # if responses_with_notes:
            #     st.write("All My Notes:")
            #     st.json(responses_with_notes)
            st.session_state["responses_with_notes"] = responses_with_notes
    return


def extract_key_phrases(text):
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag

    # Tokenize sentences
    sentences = sent_tokenize(text)
    phrases = []

    for sentence in sentences:
        # Tokenize words and tag parts of speech
        words = word_tokenize(sentence)
        tagged = pos_tag(words)

        # Look for phrases with specific patterns, e.g., noun phrases
        phrase = []
        for word, pos in tagged:
            if pos.startswith('N'):  # Nouns
                phrase.append(word)
            else:
                if phrase:
                    phrases.append(' '.join(phrase))
                    phrase = []
        if phrase:
            phrases.append(' '.join(phrase))

    # Filter for phrases with more than one word to get key phrases rather than single words
    # This removes duplicates
    unique_phrase_candidates = [phrase for phrase in phrases if len(phrase.split()) > 1]
    unique_phrases = list(set(unique_phrase_candidates))
    # sorted alphabetically, indifferent to case
    unique_phrases.sort(key=lambda x: x.lower())
    return unique_phrases

def create_index_by_row_number(df):
    stop_words = set(stopwords.words('english'))
    index_dict_by_page = OrderedDict()
    for index, row in df.iterrows():
        if pd.notna(row['Text']):
            words = row['Text'].lower().split()
            for word in words:
                if word not in stop_words:  # Skip stopwords
                    if word not in index_dict_by_page:
                        index_dict_by_page[word] = []
                    index_dict_by_page[word].append(index + 1)  # Adding 1 to match response numbers

    # Sort the OrderedDict by keys (words) alphabetically
    sorted_index_dict = OrderedDict(sorted(index_dict_by_page.items()))

    return sorted_index_dict


def count_placeholders(text):

    # The pattern to match is \[ {string} \] where string can be any characters except newlines,
    # with 0 or 1 space before and after the brackets
    pattern =  r'\[ ?[^\]]* ?\]'

    # Use re.findall to find all non-overlapping matches of pattern in text
    matches = re.findall(pattern, text, re.DOTALL)

    return len(matches)

def index_to_dataframe(index):
    data = []
    for word, locations in index.items():
        data.append({"Word": word, "Locations": ", ".join(map(str, locations))})
    index_df = pd.DataFrame(data)
    #st.dataframe(df)
    return index_df

def display_index_as_tree(index):
    for word, locations in index.items():
        with st.expander(f"{word} ({len(locations)} locations)"):
            for location in locations:
                st.write(f"- Row {location}")

def analyze_prompt(prompt):
    prompt_keyphrases = extract_key_phrases(prompt)
    return prompt_keyphrases

def main(args):

    output_dir, text = initialize(args)
    #st.write(st.session_state)

    list_of_responses = transform_response_text_into_list(text)
    cleaned_responses, response_df = create_cleaned_df(list_of_responses)

    # optionally use xai api to create generative metadata for responses
    call_xai_api_on_responses(args, response_df)

    #create index of responses + api enhancements

    concordance = create_concordance(text)

    # begin adding metadata to response_df
    # inspect responses for gaffes
    response_df['placeholders_found'] = response_df['Text'].apply(count_placeholders)

    most_frequent = response_df.apply(
        lambda row: most_frequent_words(row['Text'], n=3) if pd.notna(row['Text']) else [], axis=1)
    response_df['Most Frequent Words'] = most_frequent

    # get key phrases from each response
    response_df['Key Phrases'] = response_df['Text'].apply(extract_key_phrases)

    # process context_prompt and compare its key phrases to responses
    with st.expander("Latest User Prompt"):
        with open(args.context_prompt, "r") as f:
            context_prompt = f.read()
        st.write(f"**Latest user prompt to model:** {context_prompt}")
        prompt_keyphrases = extract_key_phrases(context_prompt)
        st.markdown(f"**Prompt Key Phrases:** {prompt_keyphrases}")
        response_df['prompt_phrases_included'] = response_df['Text'].apply(lambda text: any(phrase in text for phrase in prompt_keyphrases))

    with st.expander("Dataframe of Responses"):
        st.write(response_df)


    with st.expander("Search Index of Responses"):
        index = create_index_by_row_number(response_df)
        searchable_index(index)


    with st.expander("16-up notepad"):
       #   st.write(st.session_state["responses_with_notes"])
        updated_responses = display_4x4_grid(st.session_state["responses_with_notes"])
        #st.write(updated_responses)
        st.session_state.responses = updated_responses

    index_to_dataframe(index)


    file_basename = str(uuid.uuid4())[:8] + ".csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, file_basename)
    response_df.to_csv(filepath)


def initialize(args):
    input_dir = "input"
    import_file = args.input
    import_file_path = os.path.join(input_dir, import_file)
    with open(import_file_path, "r") as f:
        text = f.read()
    output_dir = "output"

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if 'responses' not in st.session_state:
        st.session_state.responses = {
            f'response_{i}': f"Sample content for Response {i}" for i in range(1, 17)
        }  # Example prepopulated content
    if 'responses_with_notes' not in st.session_state:
        st.session_state["responses_with_notes"] = {}
        # loop 1 through 17
        for i in range(1, 17):
            st.session_state["responses_with_notes"][i] = ''

    return output_dir, text


def call_xai_api_on_responses(args, response_df):
    if args.use_xai:
        api_responses = []
        for index, row in response_df.iterrows():
            prompt = f"In fifteen words or less, identify the key weaknesses in this text: {row['Text']}"
            api_response = call_xai_api(prompt)
            print(api_response)
            api_responses.append(api_response)
        response_df['API Response'] = api_responses


def create_cleaned_df(list_of_responses):
    cleaned_responses = []
    for i, response in enumerate(list_of_responses):
        lines = response.splitlines()
        cleaned_lines = [line for line in lines if
                         not line.startswith("Words") and not line.startswith("Characters") and not line.startswith(
                             "Paragraphs")]
        cleaned_responses.append({"Response": f"Response {i + 1}", "Text": "\n".join(cleaned_lines)})
    response_df = pd.DataFrame(cleaned_responses)
    return cleaned_responses, response_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text responses and optionally use xAI API.')
    parser.add_argument('--input', required=False, help='Path to the input file', default="response.txt")
    parser.add_argument("--context_prompt", required=False, help='Path to the file holding the context prompt', default="prompt.txt")
    parser.add_argument('--use_xai', action='store_true', help='Flag to enable xAI API for text analysis')
    args = parser.parse_args()

    main(args)