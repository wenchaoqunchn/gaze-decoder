import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path configuration — defaults to locations relative to this script.
# Override via CLI arguments or GAZEDC_INPUT_JSON / GAZEDC_OUTPUT_JSON envvars.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT = os.environ.get(
    "GAZEDC_INPUT_JSON",
    os.path.join(_HERE, "context_features", "context_extraction_raw.json"),
)
_DEFAULT_OUTPUT = os.environ.get(
    "GAZEDC_OUTPUT_JSON",
    os.path.join(_HERE, "context_features", "complete_econtext.json"),
)

parser = argparse.ArgumentParser(
    description="Embed context descriptions and code snippets."
)
parser.add_argument(
    "--input", default=_DEFAULT_INPUT, help="Path to context_extraction_raw.json"
)
parser.add_argument(
    "--output", default=_DEFAULT_OUTPUT, help="Path for complete_econtext.json output"
)
args, _ = parser.parse_known_args()

INPUT_JSON = args.input
OUTPUT_JSON = args.output

# Load NLP Model (Global)
print("Loading Sentence Transformer Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Simulated LLM Knowledge Base from Software Docs
DOMAIN_KNOWLEDGE = {
    "HomePage": "Portal homepage containing global navigation, search bar, library photos carousel, and news list.",
    "Overview": "Library overview page providing basic information and side navigation.",
    "LibIntro": "Library introduction page showing history, area, collection size, and service models.",
    "LeaderSpeech": "Page displaying the library director's message.",
    "LibRule": "Page listing library rules and regulations.",
    "ServiceTime": "Table showing opening hours for different library areas (Reading Room, Stacks, Study Room).",
    "ServiceOverview": "Table listing services (e.g., Borrowing, Sci-tech Search) with locations and contact numbers.",
    "LibLayout": "Page showing floor plans of the library (F2, F3, F4) via carousel.",
    "Resources": "Navigation page for digital and physical library resources.",
    "CoreJournal": "Guide to core journal indexes like SCI, SSCI, CSSCI and usage methods.",
    "EBook": "List of electronic book databases like Springer, Ebsco, and SuperStar.",
    "LibThesis": "Information about university thesis collection, coverage, and query methods.",
    "CommonApp": "Download page for tools like NoteExpress, EndNote, Adobe Reader, CAJViewer.",
    "Copyright": "Copyright announcements and regulations for digital resources protection.",
    "ServicePage": "Overview of various library services.",
    "BookBorrow": "Information on borrowing rules, self-service machines, and reading areas.",
    "CardProcess": "Guide for campus card activation and permissions for borrowed items.",
    "AncientRead": "Rules and reservation process for accessing ancient books.",
    "DiscRequest": "Request methods for accompanying discs of books (Database, Management System).",
    "DocumentTransfer": "Inter-Library Loan (ILL) service for obtaining documents not held by the library.",
    "TechSearch": "Information on Sci-tech novelty search services, qualifications, and delegation process.",
    "InfoTeaching": "Information retrieval curriculum settings for graduate students.",
    "VolunteerTeam": "Library volunteer organization info and registration via QR code.",
    "SeatReserve": "Entrance to seat reservation system, includes 'Start Reserve' button.",
    "Reserve": "Main reservation container managing the 4-step wizard process (Floor -> Time -> Seat -> Confirm).",
    "FloorSelect": "Step 1 of reservation: Select library floor (F2, F3, F4) via cards.",
    "TimeSelect": "Step 2 of reservation: Select date (future week) and time slot (7:30-22:30).",
    "SeatSelect": "Step 3 of reservation: Interactive seat map. Contains simulated usability issues (30% failure rate, random selection logic).",
    "InfoConfirm": "Step 4 of reservation: Confirm booking details (Floor, Date, Seat) before submission.",
    "ContactUs": "Page listing contact information for various library departments and general inquiries.",
    "SessionReady": "Experiment configuration page before starting a session (Session Index, Calibration entry).",
    "SessionDone": "Completion page after finishing the reservation task, allows data analysis or return to home.",
    "Calibration": "Eye tracker calibration page usually containing target points.",
    "aoi nav": "Global Top Navigation Bar for module switching.",
    "aoi logo": "System Branding Logo, clicking usually returns to Home.",
    "filter-button": "Interactive filter control for data listings.",
    "key-aoi": "Critical interaction element essential for task completion.",
    "aoi breadcrumb": "Navigation aid indicating the current page location in the hierarchy.",
    "aoi footer": "Page footer containing links to related organizations and copyright info.",
}


def analyze_semantic_function(item):
    """
    Simulates LLM-based functional recognition based on Component Info and Code.
    """
    info = item["componentInfo"].lower()
    code = item.get("code_snippet", "").lower()
    path = item.get("src_rel_path", "")

    description_parts = []

    # 1. Component Identity
    if "nav" in info:
        description_parts.append("Navigation Component")
    elif "button" in info or "btn" in info:
        description_parts.append("Action Button")
    elif "input" in info:
        description_parts.append("Data Entry Field")
    elif "img" in info or "logo" in info:
        description_parts.append("Visual Element")
    else:
        description_parts.append("UI Container/Element")

    # 2. Contextual Role (Based on View/Path)
    found_context = False
    for key, desc in DOMAIN_KNOWLEDGE.items():
        if key.lower() in path.lower() or key.lower() in info:
            description_parts.append(f"Related to {key}: {desc}")
            found_context = True

    if not found_context:
        description_parts.append(f"Located in {os.path.basename(path)}")

    # 3. Code Analysis (Simulated)
    if "@click" in code or "onclick" in code:
        description_parts.append("Interactive (Clickable)")
    if "v-for" in code:
        description_parts.append("List/Collection Item")

    return ". ".join(description_parts)


def get_text_embedding(text):
    """
    Generate embedding using SentenceTransformer.
    """
    if not text.strip():
        # Return zero vector if empty
        return np.zeros(384).tolist()  # all-MiniLM-L6-v2 output dim is 384

    embedding = model.encode(text)
    return embedding.tolist()


def main():
    if not os.path.exists(INPUT_JSON):
        print("Raw context file not found.")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    enhanced_context = {}

    print("Generating Semantic Descriptions and Embeddings...")
    for item in raw_data:
        key = item["key"]

        # 1. Semantic Recognition (Text)
        func_desc = analyze_semantic_function(item)

        # 2. Embeddings (Vector)
        # Embed Text
        text_vec = get_text_embedding(func_desc)
        # Embed Code
        code_vec = get_text_embedding(item.get("code_snippet", ""))

        enhanced_context[key] = {
            "semantic_description": func_desc,
            "embed_text": text_vec,
            "embed_code": code_vec,
            "original_info": item["componentInfo"],
            # Keep raw data reference if needed
            "src_path": item["src_rel_path"],
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(enhanced_context, f, indent=2, ensure_ascii=False)

    print(f"Generated Global Complete Econtext for {len(enhanced_context)} items.")
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
