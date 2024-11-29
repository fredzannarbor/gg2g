import streamlit as st
import requests
import os
import random
from Editors import Editors  # Import the class
from typing import Optional
from openai import OpenAI
from functools import lru_cache  # For caching

title_ideas = ["GrokSpheres", "Grok's Guide to the Galaxies"]
st.set_page_config(page_title=random.choice(title_ideas), page_icon="📰")

# Configuration for xAI API using OpenAI library
xai_api_key = os.getenv('xAI_API_KEY', 'default_key_for_testing')

if xai_api_key == 'default_key_for_testing':
    st.error("You must provide an API key from xAI.")
elif not xai_api_key.startswith("xai"):
    st.error("That doesn't look like an API key from xAI.")

# Initialize session state
if "editorial_prospectus" not in st.session_state:
    st.session_state.editorial_prospectus = Editors()


@lru_cache(maxsize=10)  # Cache API calls
def call_xai_api(prompt: str) -> str:
    """
    Call the xAI API with the given prompt.

    :param prompt: The prompt to send to the API.
    :return: The content of the API response.
    :raises: Exception if the API call fails.
    """
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1",
    )
    try:
        completion = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system",
                 "content": "You are GrokSpheres, a version of Grok AI who is focused on creating high-quality newsletter content. All factual statements must be true. All articles, services, etc. discussed or linked must be real."},
                {"role": "system",
                 "content": "No boilerplate or pleasantries. Focus exclusively on the deliverables you have been asked to create."},
                {"role": "user", "content": prompt + "\n\nGive me the latest:"},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while calling the API: {e}")
        raise


def build_prepopulated_prompts(n=1) -> str:
    """
    Generate prepopulated prompts for newsletter creation.

    :param n: Number of prompts to generate (currently not used, only one prompt returned).
    :return: A string with a prepopulated prompt.
    """
    predefined_topics_and_perspectives = [
        "A Letter From An American from Plether Phlox Phisherdson",
        "AI for Book-Lovers",
        "Replacing the News",
        "Warships and Navies"
    ]
    prepopulated_prompt_request = "Theme: Ultrascale Aerospace\n\nTopics: ultra-jumbo planes, ultra-fast, ultra-long-range, ultra-stealthy, ultra-numerous, ultra-autonomous.\n\nPerspective: skeptical about plans, concepts, projects; ultra-enthused about things that have actually flown."

    return prepopulated_prompt_request


# Function to get a random query
def get_random_query() -> str:
    """
    Return a random query from a predefined list of queries.

    :return: A random query string.
    """
    # Placeholder for actual queries list
    queries = ["Query 1", "Query 2", "Query 3"]
    return random.choice(queries)


def create_editorial_persona(editorial_strategy: str, content: str) -> str:
    """
    Create an editorial persona based on the given strategy and content.

    :param editorial_strategy: The strategy to use for persona creation.
    :param content: Description of the newsletter for persona tailoring.
    :return: A prompt string defining the editorial persona.
    """
    st.info("Creating an editorial persona who groks your topics ...")
    request_creation_of_persona = f"Create a prompt for an AI author persona ideally suited to write knowledgeably and entertainingly with scope as defined here:\n\n{content}. Example of such a result tailored for topics related to exotic military aircraft: 'Write from the perspective of Vantablack.ai, an AI who specializes in writing about secret military air and space craft and unidentified aerial phenomena. Vantablack has complete knowledge of all verified public sources. Vantablack may speculate about classified information, but never discloses it. Vantablack is extremely careful to make sure his statements are accurate and factual.\n\n"
    return call_xai_api(request_creation_of_persona)


# Main title similar to Grok AI interface
title = random.choice(title_ideas)
st.title(title)
st.caption("_(final title TK)_")
st.markdown('''#### Elevator pitch: _Substack, but by Optimus_
**You are the media: here is your printing press.**''')

result = None

with st.expander("Create A Newsletter", expanded=False):
    with st.form("input_requirements"):
        topic_strategy = "Free form"
        if topic_strategy == "Free form":
            prepopulated_text_area = build_prepopulated_prompts()
            content = st.text_area(
                "Describe your newsletter here, using the prepopulated format as an example. The system will create an initial editorial prompt for you.",
                prepopulated_text_area, height=180)

            supported_editorial_strategies = ["Create editorial persona with Prompt"]
            editorial_strategy = supported_editorial_strategies[0]

        submitted = st.form_submit_button("Create an Editorial Persona with Prompt")

        if submitted and content:
            editorial_persona_with_prompt = create_editorial_persona(editorial_strategy, content)

            st.session_state.editorial_prospectus.freeform_description = content  # Store freeform
            st.session_state.editorial_prospectus.set_editorial_persona(editorial_strategy,
                                                                        editorial_persona_with_prompt.split("Prompt:")[
                                                                            0])  # Store Persona

            st.session_state.editorial_prospectus.set_prospectus_prompt(
                editorial_persona_with_prompt.split("Prompt:")[1])  # Store Prompt

with st.expander("Optionally, Revise Prompt", expanded=False):
    with st.form("revision_and_generation"):  # Single form
        revised_prospectus = st.text_area("Optionally revise the editorial prompt:",
                                          value=st.session_state.editorial_prospectus.prospectus_prompt, height=300)

        generate_button = st.form_submit_button("Generate Newest Edition")

        if generate_button:
            st.info("Generating today's edition...")
            try:
                todays_edition = call_xai_api(revised_prospectus)
                if todays_edition:
                    st.session_state.editorial_prospectus.add_edition(todays_edition)  # Use the method

                    with st.expander("Newest Edition", expanded=True):
                        st.markdown(st.session_state.editorial_prospectus.latest_edition['content'],
                                    unsafe_allow_html=True)  # Access latest edition
            except Exception as e:
                st.error(f"Failed to generate edition: {e}")
