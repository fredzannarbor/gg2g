"""
Class to define editorial objects for use by grokstack.py.
Objects:
- input freeform description (user dialog)
- prompt(s) to convert freeform to editorial prospectus
- editorial prospectus
    attributes:
        editor_name
        editor_persona
        prospectus_prompt
        prospectus_title
- editions
    most_recently_generated date
    content of previous editions
    list of content of previous editions
- generation schedules

CRUD, standard python methods, object is dictionary
"""

import datetime
from typing import Optional, Dict, List

class Editors:
    def __init__(self, freeform_description: Optional[str] = None):
        self.editions: List[Dict] = []
        self.editorial_persona: Dict = {}
        self.editorial_prospectus: Dict = {}
        self.freeform_description: Optional[str] = freeform_description
        self.generation_schedule: Optional[str] = None
        self.persona_template: str = f"Create a prompt for an AI author persona ideally suited to write knowledgeably and entertainingly with scope as defined here:\n\n{freeform_description}. Example of such a result tailored for topics related to exotic military aircraft: 'Write from the perspective of Vantablack.ai, an AI who specializes in writing about secret military air and space craft and unidentified aerial phenomena. Vantablack has complete knowledge of all verified public sources. Vantablack may speculate about classified information, but never discloses it. Vantablack is extremely careful to make sure his statements are accurate and factual.\n\n"
        self.prompts: List[str] = []
        self.prospectus_prompt: str = ""

    def add_prompt(self, prompt: str) -> None:
        """
        Add a prompt to the list of prompts.

        :param prompt: The prompt to add.
        """
        self.prompts.append(prompt)

    def set_editorial_persona(self, editorial_strategy: str, content: str, persona_template: Optional[str] = None) -> None:
        """
        Set the editorial persona based on strategy and content.

        :param editorial_strategy: The strategy for persona creation.
        :param content: The content for persona tailoring.
        :param persona_template: Optional template for persona creation.
        """
        if persona_template is None:
            persona_template = self.persona_template
        self.editorial_persona = {"strategy": editorial_strategy, "content": content, "template": persona_template}
        self.editorial_prospectus["editor_persona"] = self.editorial_persona

    def set_prospectus_prompt(self, prompt: str) -> None:
        """
        Set the prospectus prompt.

        :param prompt: The prompt to set for the prospectus.
        """
        self.prospectus_prompt = prompt

    @property
    def latest_edition(self) -> Optional[Dict]:
        """
        Get the latest edition.

        :return: The last edition in the list or None if no editions exist.
        """
        if self.editions:
            return self.editions[-1]
        return None

    def set_editorial_prospectus(self, prospectus_data: Dict) -> None:
        """
        Set the editorial prospectus data.

        :param prospectus_data: Dictionary containing prospectus information.
        """
        self.editorial_prospectus = prospectus_data

    def add_edition(self, content: str) -> None:
        """
        Add a new edition with the current date.

        :param content: The content of the edition.
        """
        edition_data = {"date": datetime.date.today(), "content": content}
        self.editions.append(edition_data)

    def set_generation_schedule(self, schedule: str) -> None:
        """
        Set the generation schedule for editions.

        :param schedule: String representation of the schedule.
        """
        self.generation_schedule = schedule

    def to_dict(self) -> Dict:
        """
        Convert the object to a dictionary for serialization.

        :return: Dictionary representation of the object.
        """
        return {
            "freeform_description": self.freeform_description,
            "prompts": self.prompts,
            "editorial_prospectus": self.editorial_prospectus,
            "editions": self.editions,
            "generation_schedule": self.generation_schedule,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Editors':
        """
        Create an Editors instance from a dictionary.

        :param data: Dictionary containing data to initialize the object.
        :return: New instance of Editors class.
        """
        prospectus = cls()
        prospectus.freeform_description = data.get("freeform_description")
        prospectus.prompts = data.get("prompts", [])
        prospectus.editorial_prospectus = data.get("editorial_prospectus", {})
        prospectus.editions = data.get("editions", [])
        prospectus.generation_schedule = data.get("generation_schedule")
        return prospectus