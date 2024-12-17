
import io
import os
import re
import sys
import traceback
import uuid
import argparse
import random

import textstat
import sumy
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
import nltk
import pandas as pd
from collections import Counter, OrderedDict
from nltk.corpus import stopwords


from utilities import Pandas_Utilities

from PIL import Image, ImageDraw, ImageFont
# Download stopwords if not already present
nltk.download('stopwords', quiet=True)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def get_xai_api_key():
    global xai_api_key, base_url
    xai_api_key = os.getenv('xAI_API_KEY')
    base_url = "https://api.x.ai/v1"

get_xai_api_key()

# Constants
NUMBER_OF_RESPONSES = 16

st.set_page_config("Responses Viewer", "⑯", layout="wide")
st.title("Responses Viewer ⑯")
st.markdown("_Deterministic Text Analytics for Response Evaluation_")

# Initialize session state
if 'notes' not in st.session_state:
    st.session_state.notes = {}
if 'response_names' not in st.session_state:
    st.session_state.response_names = []
if 'responses' not in st.session_state:
    st.session_state.responses = {f'response_{i}': f"Sample content for Response {i}" for i in
                                  range(1, NUMBER_OF_RESPONSES + 1)}
if 'responses_with_notes' not in st.session_state:
    st.session_state["responses_with_notes"] = {i: '' for i in range(1, NUMBER_OF_RESPONSES + 1)}


def transform_response_text_into_list(text):
    responses = re.split(r'\nResponse \d+', text)
    return [response.strip() for response in responses if
            response.strip() and not response.strip().startswith("All Responses")]


def most_frequent_words(text, n=1):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(n)]


def call_xai_api(prompt, model="grok-beta"):
    try:
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
    except Exception as e:
        st.error(f"An error occurred while calling xAI API: {e}")
        return None


def create_concordance(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    concordance = {}
    for word in filtered_tokens:
        if word not in concordance:
            concordance[word] = []
        concordance[word].append(filtered_tokens.index(word))

    return concordance


def searchable_index(index):
    word_to_search = st.text_input("Search for a word in the index:")
    if word_to_search:
        search_term = word_to_search.lower()
        if any(search_term == key.lower() for key in index):
            matching_key = next(key for key in index if key.lower() == search_term)
            locations = sorted(set(index[matching_key]))  # Remove duplicates and sort
            locations_str = ', '.join(map(str, locations))
            st.write(f"Word **{word_to_search}** found in responses {locations_str}")
        else:
            st.write(f"**{word_to_search}** not found in the index.")
    else:
        st.write("Enter a word to search for in the index.")


def show_dataframe_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())


def display_4x4_grid(responses=None):
    if responses is None:
        responses = {i: '' for i in range(1, NUMBER_OF_RESPONSES + 1)}

    cols = st.columns(4)
    with st.form("16-notes"):
        for i, col in enumerate(cols):
            for j in range(4):
                index = i * 4 + j + 1
                responses[index] = col.text_area(
                    f"Response {index}",
                    value=responses.get(index, ""),
                    height=150,
                    key=f'response_{index}'
                )
        if st.form_submit_button('Save All My Notes'):
            st.session_state["responses_with_notes"] = responses
    return responses


def display_4x4_text(responses=None, key="keyphrases"):
    if responses is None:
        responses = {i: '' for i in range(1, NUMBER_OF_RESPONSES + 1)}
    st.write(responses)
    cols = st.columns(4, vertical_alignment="top")
    for i, col in enumerate(cols):
        for j in range(4):
            index = i * 4 + j + 1
            display_text = responses[index - 1] if isinstance(responses, list) else [responses[index -1]]
            with col.container(border=True):
                col.markdown(f"- {'- '.join(display_text)}")
                col.write('---')


import streamlit as st

def display_4x4_baseball_cards(list_of_responses):
    # Create 4 columns
    cols = st.columns(4)

    for i in range(4):  # Outer loop for rows
        with st.container(height=600):
            for j, col in enumerate(cols):  # Inner loop for columns
                index = i * 4 + j  # Calculate the index in the list
                if index < len(list_of_responses):  # Check if we have data for this position
                    with col:
                        col.write(f"Response {index + 1}")
                        with st.container():
                            card = create_baseball_card_container(list_of_responses[index])


def create_word_cloud_from_response(text):
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate(text)
    wordcloud_image = wordcloud.to_image()
    img_byte_arr = io.BytesIO()
    wordcloud_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def create_and_display_4x4_images(responses=None, key="from-responses"):
    if responses is None:
        responses = {i: '' for i in range(1, NUMBER_OF_RESPONSES + 1)}

    cols = st.columns(4)
    with st.form(f"16-{key}"):
        for i, col in enumerate(cols):
            for j in range(4):
                index = i * 4 + j + 1
                col.info(f"Generating image {index}")
                text = responses[index - 1]
                if isinstance(text, list):  # Check if it is a list
                    text = ' '.join(text)  # and make it a str if so.
                wordcloud_image = create_word_cloud_from_response(text)  # Then generate the image
                col.image(wordcloud_image, caption=f"Word Cloud of Response {index}", use_container_width=False)


def create_card_name(list_of_responses):
    name_components = list_of_responses[0].split()[:3]
    name = "-".join(name_components)
    if name in st.session_state.response_names:
        name += "-" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=3))
    st.session_state.response_names.append(name)
    return name


def create_baseball_card_container(text):
    with st.container():
        # Use custom CSS to set the width
        st.markdown(
            """
            <style>
            .stTabs [data-baseweb="tab-list"] {
                width: 400px;
            }
            </style>
            """,
            unsafe_allow_html=True)

    tab1, tab2 =  st.tabs(["Front", "Back"])
    with tab1:
        key_phrases = extract_key_phrases_by_relevance(text)
        card_name = create_card_name(key_phrases)
        wordcloud_image = create_word_cloud_from_response(text)
        st.markdown(f"""### {card_name}""")
        reading_level = textstat.textstat.flesch_kincaid_grade(text)
        st.markdown(f"""**Reading Level (grade):** {reading_level}""")
        top_3 = extract_top_three_sentences_7_words(text)
        st.json(top_3)
        all_key_phrases = extract_key_phrases_by_relevance(text, n=50)
        # number of phrases that are in prompt that are found in response text
        prompt_phrases_in_responses = [phrase for phrase in all_key_phrases if phrase in text]
        #st.markdown(f"""**Prompt overlaps:** {len(prompt_phrases_in_responses)}""")

        st.image(wordcloud_image)

    with tab2:
        # key phrases as markdown bullets
        st.json(key_phrases, expanded=True)
        st.write("**Markdown Structure**    ")
        toc = get_markdown_toc(text)
        st.write(toc)

def clean_lines_beginning_with_words(lines):
    ignore_prefixes = {"Words", "Characters", "Paragraphs", "User Avatar", "Load", "Save", "TaskId"}
    return [line for line in lines if not any(line.startswith(prefix) for prefix in ignore_prefixes)]

def extract_top_three_sentences_7_words(text):
    # split text into lines
    raw_lines = text.splitlines()
    # clean up lines
    clean_lines = clean_lines_beginning_with_words(raw_lines)
    # turn clean_lines back into text
    text = " ".join(clean_lines)

    top_3_sentences = summarize_text_using_statistical_methods(text)
    # convert list of sentence objects to list of strings with sentence truncated after 7 words
    top_3_sentences_strings = [str(sentence) for sentence in top_3_sentences]
    top_3_sentences_strings = [sentence.split()[:7] for sentence in top_3_sentences_strings]
    top_3_sentences_strings = [" ".join(sentence) for sentence in top_3_sentences_strings]
    # add "..." to each item in list
    top_3_sentences_strings = [sentence + " ..." for sentence in top_3_sentences_strings]

    return top_3_sentences_strings


def extract_key_phrases_by_relevance(text, n=5):
    if isinstance(text, list):
        text = " ".join(text)
    elif not isinstance(text, str):
        text = str(text)

    cleaned_text = " ".join(clean_lines_beginning_with_words(text.splitlines()))
    sentences = nltk.sent_tokenize(cleaned_text)
    phrases = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        phrase = []
        for word, pos in tagged:
            if pos.startswith('N'):
                phrase.append(word)
            else:
                if phrase:
                    phrases.append(' '.join(phrase))
                    phrase = []
        if phrase:
            phrases.append(' '.join(phrase))

    unique_phrases = list(set([phrase for phrase in phrases if len(phrase.split()) > 1]))
    return unique_phrases[:n]


def clean_lines_beginning_with_words(lines):
    ignore_prefixes = {"Words", "Characters", "Paragraphs", "User Avatar", "Load", "Save", "TaskId"}
    return [line for line in lines if not any(line.startswith(prefix) for prefix in ignore_prefixes)]


def create_index_by_row_number(df):
    stop_words = set(stopwords.words('english'))
    index_dict = OrderedDict()
    for index, row in df.iterrows():
        if pd.notna(row['Text']):
            words = row['Text'].lower().split()
            for word in words:
                if word not in stop_words:
                    if word not in index_dict:
                        index_dict[word] = []
                    index_dict[word].append(index + 1)

    return OrderedDict(sorted(index_dict.items()))


def count_placeholders(text):
    return len(re.findall(r'\[ ?[^\]]* ?\]', text, re.DOTALL))


def index_to_dataframe(index):
    data = [{"Word": word, "Locations": ", ".join(map(str, locations))} for word, locations in index.items()]
    return pd.DataFrame(data)


def display_index_as_tree(index):
    for word, locations in index.items():
        with st.expander(f"{word} ({len(locations)} locations)"):
            for location in locations:
                st.write(f"- Row {location}")


def analyze_prompt(prompt):
    return extract_key_phrases_by_relevance(prompt)


def initialize(args):
    input_dir = "aitutorhelpers/input"
    import_file = args.input
    import_file_path = os.path.join(input_dir, import_file)

    try:
        with open(import_file_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        st.error(f"File {import_file_path} not found.")
        return None, None

    output_dir = "output"

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir, text

def get_markdown_toc(text, bolds=True):
    # if text contains markdown headers beginning with #, extract them (only) and display
# the toc
    lines = text.splitlines()
    lines = clean_lines_beginning_with_words(lines)
    candidates = [line for line in lines if not line.startswith('#') and len(line.strip()) > 0 and '.' not in line and '\n\n' in text]
    toc = "\n\n".join(candidates)
    return toc


def call_xai_api_on_responses(args, response_df):
    if args.use_xai_api:
        api_responses = []
        for index, row in response_df.iterrows():
            prompt = f"In fifteen words or less, identify the key weaknesses in this text: {row['Text']}"
            api_response = call_xai_api(prompt)
            if api_response is not None:
                api_responses.append(api_response)
            else:
                api_responses.append("API error")
        response_df['API Response'] = api_responses


def create_response_df(list_of_responses):
    cleaned_responses = []
    for i, response in enumerate(list_of_responses):
        cleaned_lines = clean_lines_beginning_with_words(response.splitlines())
        cleaned_responses.append({"Response ": f"{i + 1}", "Text": "\n".join(cleaned_lines)})


    return pd.DataFrame(cleaned_responses)

def summarize_text_using_statistical_methods(text):
    # use sumi
    try:
        summarizer = LsaSummarizer()
        parser = sumy.parsers.plaintext.PlaintextParser.from_string(text, Tokenizer("english"))
        summary_list = summarizer(parser.document, sentences_count=3)
    except Exception as e:
        print(traceback.format_exc())
      #  st.write(traceback.format_exc(()))
    #summary_text = " ".join([str(sentence) for sentence in summary])
    return summary_list


def extract_criteria_from_hard_coded_criteria_text(text):
    lines = text.splitlines()
    output_rows = []
    c_index = 1
    d_text = None
    for i, line in enumerate(lines):
        if re.match(r'^C\d+', line):
            if d_text:
                output_rows.append(f"C{c_index}: {d_text}")
                #st.write(f"C{c_index}: {d_text}")
                c_index += 1
            d_text = None
        elif line.startswith('D'):
            if i + 1 < len(lines):
                d_text = lines[i + 1]
            else:
                d_text = ""
    if d_text:
        output_rows.append(f"C{c_index}: {d_text}")

    return output_rows

def get_Flesch_Kinkaid_grade_level(text):
    return textstat.textstat.flesch_kincaid_grade(text)

def main(args):
    if args.use_xai_api:
        get_xai_api_key() # sets up globals
    if xai_api_key is None:
        st.error("You must provide an API key from xAI.")
        return
    if not xai_api_key.startswith("xai"):
        st.error("That doesn't look like an API key from xAI.")
        return

    output_dir, text = initialize(args)
    if text is None:
        return

    list_of_responses = transform_response_text_into_list(text)
    response_df = create_response_df(list_of_responses)

    # Use xAI API if enabled
    if args.use_xai_api:
        call_xai_api_on_responses(args, response_df)
    st.toast("No large language models were harmed in the creation of this script.")

    concordance = create_concordance(text)
    context_location = st.radio("Context Location", ["Default File System", "Upload"])
    try:
        with open(args.context_prompt, "r") as f:
            context = f.read()

    except FileNotFoundError:
        st.error(f"Context prompt file not found: {args.context_prompt}")
    # Process context prompt
    with st.expander("Full Context To Date", expanded=False):
        st.write(f"**Latest user context dialog:** {context}")
    enhanced_response_df = enhance_response_df(response_df, context)
    #st.write(enhanced_response_df)

    with st.expander("Top 10 Key Phrases from Prompt By Relevance", expanded=True):
        prompt_keyphrases = extract_key_phrases_by_relevance(context, n=10)
        st.markdown(f"**Key Phrases:** {prompt_keyphrases}")
        task_name = create_card_name(prompt_keyphrases[0])


    with st.expander("User's Ask", expanded=True):
        try:
            with open(args.ask_file_path, "r") as f:
                ask = f.read()
                st.write(ask)

        except Exception as e:
            st.error(f"Ask file not read: {args.ask_file_path}")


    with st.expander("Crafted Criteria", expanded=True):
        try:
            with open(args.raw_criteria, "r") as f:
                criteria_text = f.read()
                st.write(extract_criteria_from_hard_coded_criteria_text(criteria_text))
        except FileNotFoundError:
            st.error(f"Criteria file not found: {args.raw_criteria}")

    # with st.expander("Dataframe of Responses"):

    st.session_state['edited_df'] = enhanced_response_df

    with st.expander("Load Previous Work"):
        # load saved dataframe to continue previous work
        utils = Pandas_Utilities()
        retrieved_df = utils.streamlit_file_picker()
        if retrieved_df is not None:
            st.session_state['edited_df'] = retrieved_df


    with st.expander("Transposed for Easy Editing"):
        cols = []
        for r in range(1, 17):
            cols.append(f"Response {r}")
        transposed_df = enhanced_response_df.T
        # column names for tra
        transposed_df.columns = cols

        # Convert object columns to strings
        for col in transposed_df.columns:
            if transposed_df[col].dtype == object:  #Could also use transposed_df[col].apply(type).eq(list).any():
                transposed_df[col] = transposed_df[col].astype(str)  # Or handle lists specifically

        edited_df = st.data_editor(transposed_df, hide_index=True, num_rows="dynamic")
        save_df = st.button("Save Dataframe With Tutor Notes")
        if save_df:
            saved_df = edited_df
            saved_df.to_csv(f"aitutorhelpers/output/{task_name}.csv")
            st.session_state["edited_df"] = saved_df


    with st.expander("Transposed Back for Easy Sorting (read-only)"):
        retransposed_df = st.session_state['edited_df'].T
        st.dataframe(retransposed_df, hide_index=True)



    with st.expander("Search Index of Responses"):
        index = create_index_by_row_number(response_df)
        searchable_index(index)

    # with st.expander("League Leaders"):
    #     col1, col2, col3, col4 = st.columns(4)
    #     easiest_reads = response_df.nsmallest(2, 'Grade Level')
    #     hardest_reads = response_df.nlargest(2, 'Grade Level')
    #     # drop indexes
    #     easiest_reads = easiest_reads.reset_index(drop=True)
    #     hardest_reads = hardest_reads.reset_index(drop=True)
    #     # show me just the response number and the grade level
    #     col1.write("Easiest Reads")
    #     col1.dataframe(easiest_reads[['R#', 'Grade Level']], hide_index=True)
    #     col2.write("Hardest Reads")
    #     col2.dataframe(hardest_reads[['R#', 'Grade Level']], hide_index=True)


    with st.expander("16-up: Baseball Cards"):
        if list_of_responses:  # only display if there are responses
            display_4x4_baseball_cards(list_of_responses)  # Pass the list, not the dictionary.

    with st.expander("16-up: Five Most Relevant Key Phrases"):
        display_4x4_text(response_df['Key Phrases'].to_list(), "key phrases by relevance")

    with st.expander("16-up: Display Word Cloud of Responses"):
        create_and_display_4x4_images(list_of_responses)

    with st.expander("16-up: Display Word Clouds of Key Phrases"):
        create_and_display_4x4_images(response_df['Key Phrases'].to_list(), key="key_phrases")


    with st.expander("16-up notepad"):
        updated_responses = display_4x4_grid(st.session_state["responses_with_notes"])
        st.session_state["responses_with_notes"] = updated_responses

    index_df = index_to_dataframe(index)
    with st.expander("Index as DataFrame"):
        st.dataframe(index_df, hide_index=True)

    file_basename = str(uuid.uuid4())[:8] + ".csv"
    filepath = os.path.join(output_dir, file_basename)
    response_df.to_csv(filepath)


def enhance_response_df(response_df, context_prompt, ask=None):

    response_df['Grade Level'] = response_df['Text'].apply(get_Flesch_Kinkaid_grade_level)
    response_df['Words'] = response_df['Text'].apply(lambda text: len(text.split()))
    response_df['Top 3 Sents'] = response_df['Text'].apply(extract_top_three_sentences_7_words)
    response_df['Key Phrases'] = response_df['Text'].apply(extract_key_phrases_by_relevance)
    response_df['Most Frequent Words'] = response_df.apply(
        lambda row: most_frequent_words(row['Text'], n=7) if pd.notna(row['Text']) else [], axis=1)
    prompt_keyphrases = extract_key_phrases_by_relevance(context_prompt, n=10)
    task_name = create_card_name(prompt_keyphrases[0])
    response_df['prompt_phrases_included'] = response_df['Text'].apply(
        lambda text: any(phrase in text for phrase in prompt_keyphrases))

    response_df.insert(0, 'Tutor Notes', '')
    response_df["Complies with Ask?"] = pd.Series([None] * len(response_df))
    response_df["R#"] = response_df.index + 1
    # move to spot 1
    response_df = response_df.reindex(columns=['R#', 'Tutor Notes', 'Grade Level', 'Words','Complies with Ask?', 'Text',  'Top 3 Sents', 'Key Phrases', 'Most Frequent Words', 'prompt_phrases_included', ])


    return response_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text responses and optionally use xAI API.')
    parser.add_argument('--input', required=False, help='Path to the input file', default="response.txt")
    parser.add_argument("--context_prompt", required=False, help='Path to the file holding the context prompt',
                        default="aitutorhelpers/input/prompt.txt")
    parser.add_argument('--use_xai_api', action='store_true', help='Flag to enable xAI API for text analysis')
    parser.add_argument("--l0_input", required=False, help="L0 input for review by L1",
                        default="aitutorhelpers/input/l0_input.txt")
    parser.add_argument("--raw_criteria", required=False, help="L0 stated criteria",
                        default="aitutorhelpers/input/criteria_raw.txt")
    parser.add_argument("--ask_file_path", required=False,help="User stated intent", default="aitutorhelpers/input/ask.txt")
    args = parser.parse_args()
    use_xai = args.use_xai_api
    main(args)
