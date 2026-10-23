# Lab-Safety-Assistant

**Web-app**: https://lab-safety-assistant.streamlit.app/

**Project Github**: https://github.com/FooStoch/Lab-Safety-Assistant

**Paper**: https://docs.google.com/document/d/1O4taAI5eBaK6lfRY7YJvcu4eGYLYG5Guhzsq4llxWx0/edit?usp=sharing

**Presentation**: https://docs.google.com/presentation/d/1AjlPGo7cCoIXNXDwpwV8y9gLK6abdGDszWZSZgFHJy4/edit?usp=sharing

The Lab Safety Assistant is an API-only, safety-first conversational laboratory assistant that helps students, researchers, and hobbyists alike to identify hazards and control measures for everyday chemistry work. It accepts short text as input plus optional photos of the lab bench/setup, retrieves relevant SDS passages from the NIOSH Pocket Guide stored locally, and returns a structured JSON safety assessment along with a human-readable "official" safety summary. The assistant is explicitly conservative to put safety first always: it cites only the retrieved SDS passages for factual claims, avoids procedural instructions for hazardous operations, and reminds the reader to consult a trusted supervisor if uncertain.

## Repository layout

- `app.py` — powers Streamlit app by designing the UI and referencing `lab_safety.py`.
- `lab_safety.py` — designs and configures the LLM for lab safety help with reasoning, safety guardrails, etc.
- `convert_niosh_to_txt.py` — text-processing pipeline that extracts structured safety profiles for ~700 chemical hazards from the NIOSH Pocket Guide; individual text summaries are shown in the `output_txt` folder.
