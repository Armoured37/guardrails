import logging
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import AzureChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import spacy

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Azure LLM Setup ---
try:
    azure_llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_ID"),
        model="gpt-35-turbo",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("API_VERSION")
    )
    azure_llm.metadata = {"engine": "azure"}
    logger.debug("Azure LLM initialized successfully.")
except Exception as e:
    logger.error(f"Azure LLM initialization failed: {e}")
    raise

# --- Blocked Topics List (Dynamic) ---
blocked_topics = [
    "politics", "public figure", "legal advice",
    "hate speech", "financial advice"
]

# --- Load spaCy English Model ---
try:
    nlp_en = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    raise

# --- PII Masking (English only) ---
def mask_pii(text):
    logger.debug("Masking PII in text: %s", text)

    # Skip masking if non-English (e.g., contains Kanji/Hiragana/Katakana)
    if any('\u3000' <= c <= '\u9FFF' for c in text[:100]):
        logger.debug("Text detected as Japanese or non-English. Skipping PII masking.")
        return text

    try:
        doc = nlp_en(text)
        masked_text = text
        label_map = {
            'PERSON': 'name',
            'GPE': 'location',
            'DATE': 'date',
            'ORG': 'organization',
            'PHONE': 'phone number',
            'EMAIL': 'email'
        }

        for ent in reversed(sorted(doc.ents, key=lambda x: x.start_char)):
            if ent.label_ in label_map:
                replacement = f"{{{label_map[ent.label_]}}}"
                masked_text = masked_text[:ent.start_char] + replacement + masked_text[ent.end_char:]

        logger.debug("PII masked result: %s", masked_text)
        return masked_text
    except Exception as e:
        logger.error(f"Error during PII masking: {e}")
        return text

# --- Register Custom Action for Flow ---
def pass_to_next_module(query: str) -> str:
    return query

# --- Initialize NeMo Guardrails ---
try:
    base_dir = Path(__file__).resolve().parent.parent  # Points to new_approach/
    config = RailsConfig.from_path(str(base_dir))
    rails = LLMRails(config=config, llm=azure_llm)
    rails.register_action(pass_to_next_module, name="pass_to_next_module")
    logger.debug("NeMo Guardrails loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load NeMo Guardrails: {e}")
    raise

# --- Guardrail Execution ---
def run_guardrails_check(user_query):
    logger.debug("Received user query: %s", user_query)

    try:
        messages = [{"role": "user", "content": user_query}]
        response = rails.generate(messages=messages)

        content = response.get("content") if isinstance(response, dict) else str(response)
        logger.debug("Guardrails raw output: %s", content)

        # Log full response
        logger.debug("Full LLM response object: %s", response)

        # Check if guardrail blocked it
        is_blocked = "BLOCKED_TOPIC_DETECTED" in content

        if is_blocked:
            # Try to extract actual topic from LLM output
            detected_topic = None
            for topic in blocked_topics:
                if topic in content.lower():
                    detected_topic = topic
                    break
            if not detected_topic:
                detected_topic = "blocked_topic"

            logger.debug("Blocked topic detected via guardrail: %s", detected_topic)

            return (
                False,
                "blocked_topic",
                "Your input has been flagged for sensitive content by LLM Guardrails and has been blocked for security reasons.",
                [detected_topic]
            )

        # Passed guardrail, now apply PII masking
        masked = mask_pii(user_query)
        reason = "pii_masked" if masked != user_query else "no pii detected"

        return True, reason, masked, []

    except Exception as e:
        logger.exception(f"Guardrail check failed: {e}")
        raise
