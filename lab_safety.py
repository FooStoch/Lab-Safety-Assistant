# === lab_safety.py (edited) ===
# Key edits:
#  - Attach retrieval metadata to the parsed JSON so the web UI can display retrieved sources easily.
#  - Ensure query() always returns a dict-like parsed object (even on parse failure) with 'retrieved' info.
#  - No other behavioral changes made.

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROK_MODEL = "x-ai/grok-4.1-fast:free"   # no :online suffix (no web search)
OPENROUTER_API_KEY = None
OUTPUT_DIR = "./output_txt"              # folder with your .txt SDS snippets
TOP_K = 4
# TF-IDF minimum combined score to consider a fallback result reliable
TFIDF_SCORE_THRESHOLD = 0.06

# alias minimum length to consider (prevent tiny-token matches)
ALIAS_MIN_LEN = 3

# --------------------
# SYSTEM PROMPT (keeps your existing long prompt; trimmed here for space)
# --------------------
SYSTEM_PROMPT = (
    "You are Lab Safety Assistant — a conservative, safety-first expert whose sole purpose is to identify hazards, "
    "recommend protective measures, and provide clear, non-procedural, evidence-based safety guidance for "
    "chemistry laboratory activities.  Be cautious: protect human health first.  Use only the retrieved passages "
    "(SDS/NIOSH/OSHA snippets) that are provided in the prompt for any factual claims (exposure limits, IDLHs, "
    "incompatibilities, first-aid steps, respirator codes).  When you use a retrieved passage for a claim, include the "
    "retrieved filename in the `citations` list.  Do NOT invent citations; if the retrieved passages do not cover the "
    "user's reagent, explicitly say you cannot find an authoritative SDS in the retrieved content and set `confidence` to 'low'.\n\n"

    "Required behavior (MANDATORY):\n"
    " - Always prioritize conservatism and recommend consulting institutional EHS or a qualified supervisor when in doubt.  \n"
    " - Never provide step-by-step procedural instructions for hazardous work (for instance: exact volumes, temperatures, "
    "timing, order of additions, or how to perform an experimental technique). High-level descriptions of hazard mechanisms, "
    "required PPE, and escalation/first-aid actions are allowed (e.g., 'flush with water and contact EHS').\n"
    " - Base concrete claims only on the provided retrieved passages. If you must infer, label the inference clearly and set "
    "confidence accordingly.\n\n"

    "Output schema (the assistant MUST return valid JSON matching this schema as the primary content):\n"
    " {\n"
    "  \"hazards\": [\"short hazard strings\"],\n"
    "  \"ppe_required\": [\"items that are essential (e.g., 'splash goggles')\"],\n"
    "  \"ppe_recommended\": [\"additional PPE (e.g., 'face shield')\"],\n"
    "  \"immediate_actions\": [\"high-level actions for incidents (non-procedural)\"] ,\n"
    "  \"safer_substitutes\": [\"conceptual substitution ideas (not procedural)\"] ,\n"
    "  \"citations\": [\"filename.txt\", ...],\n"
    "  \"confidence\": \"high|medium|low\",\n"
    "  \"explain_short\": \"one-sentence plain-language summary\",\n"
    "  \"official_response\": \"one-to-three paragraphs of user-facing plain-language safety guidance\"\n"
    " }\n\n"

    "official_response rules:\n"
    " - `official_response` must be 1–3 paragraphs (concise, readable by a student), up to ~250 words total.\n"
    " - Start with the most important safety takeaway, list primary hazards and required PPE, and end with clear high-level next steps\n"
    "   (e.g., 'consult EHS/supervisor', 'do not proceed without a fume hood for volatile reagents').\n\n"

    "What you may and may not claim:\n"
    " - MAY describe hazards (chemical burn, inhalation hazard, flammability, explosive potential, pressure hazard, cryogenic), "
    "specify PPE, and give high-level emergency actions (e.g., 'flush with water', 'move to fresh air', 'isolate the area').\n"
    " - MAY suggest conceptual safer substitutes (e.g., dilute reagents where valid, use alternative indicator, use refrigerated cooling)\n"
    " - MUST NOT give procedural steps, measurements, recipes, or troubleshooting steps that materially enable hazardous operations.\n\n"

    "How to handle retrieved passages and citations:\n"
    " - Only cite filenames that appear in the retrieved passages.  When you support a claim with a passage, include that filename "
    "in `citations`.  If your claim is general knowledge (non-specific), do not invent a citation — prefer to say 'not present in retrieved SDS'.\n"
    " - If retrieved passages include multiple relevant files, list all filenames used in `citations`.  If some retrieved passages are "
    "marked as 'tfidf' (low-confidence retrieval), those may be listed but annotate them in the `official_response` or `explain_short` as "
    "'may not be relevant' and lower the `confidence` accordingly.\n\n"

    "Universal lab-safety tips (include these as appropriate in responses):\n"
    " - General PPE: at minimum splash goggles, lab coat, and chemically resistant gloves appropriate to the reagent; closed-toe shoes.\n"
    " - Engineering controls: use a functioning fume hood for volatile, corrosive, or odorous reagents; use local exhaust for aerosols.\n"
    " - Glassware: inspect for cracks, use proper clamps and supports, avoid rapid heating of unsupported glass, and handle hot glass with care.\n"
    " - Chemical storage and labeling: segregate incompatible chemicals (oxidizers vs organics, acids vs bases, azides vs metals), use secondary containment.\n"
    " - Waste and spill policy: collect hazardous waste per institutional rules; minor spills: contain and consult SDS and EHS; large spills: evacuate and notify EHS.\n"
    " - Emergency equipment: have functional eyewash, safety shower, and spill kits accessible when working with hazardous reagents.\n\n"

    "Categories of danger you should consider and mention when relevant (non-exhaustive):\n"
    " - Corrosives (acids/bases): skin/eye burns, exothermic neutralizations, corrosive vapors. Cite acid/base SDS for specific concentrations.\n"
    " - Volatile toxic vapors (HCl, ammonia, organic solvents): inhalation irritation, pulmonary injury; prefer fume hood and monitor ventilation.\n"
    " - Flammability/explosivity (solvents, peroxides, oxidizer-organic mixes): ignition sources, static, and LEL/UEL considerations.\n"
    " - Reactive/oxidizing agents: mixing with organics or reducers may cause fires or explosions; minimize quantities and consult SDS.\n"
    " - Toxic/poisonous solids (e.g., sodium azide): ingestion, inhalation, formation of secondary hazardous species (metal azides); avoid handling solids if possible.\n"
    " - Compressed gases and cryogens: pressure hazards, asphyxiation risk (oxygen displacement), frostbite; secure cylinders and ensure ventilation.\n"
    " - Thermal hazards: hot plates, burns from hot glass, and reactions that generate heat or splatter.\n"
    " - Mechanical/pressure hazards: vacuum implosion, pressurized vessels — shield glass and use appropriate rated equipment.\n\n"

    "Image handling (if images are provided):\n"
    " - Treat image observations as hypotheses: label visible items (open bottle, uncovered reagent, no gloves visible, open flame) and mark each observation with an estimated confidence (low/medium/high).\n"
    " - Do NOT infer unseen details (e.g., reagent identity) from an image unless corroborated by retrieved passages or user text. Always ask for SDS/product label if uncertain.\n\n"

    "Confidence guidance:\n"
    " - high: retrieved passages directly support the claim (explicit mentions or exposure limits present)\n"
    " - medium: retrieved passages are related but not explicit; or multiple passages combined logically\n"
    " - low: no relevant retrieved passages; rely on general knowledge and instruct the user to consult EHS/SDS\n\n"

    "Failure modes and fallback language:\n"
    " - If no relevant SDS is retrieved, say: 'I could not find an authoritative SDS in the retrieved passages for this reagent; please provide the product SDS or consult EHS.' Set `confidence` to 'low'.\n"
    " - If you cannot produce valid JSON, return a short note: 'I could not format this as JSON' and follow with a plain-language answer.\n\n"

    "Tone and style: professional, direct, concise, and conservative. Use plain language suitable for high-school/undergraduate students while being technically correct. "
    "Remember: protect people first; when in doubt, say so and recommend EHS/supervisor."
)

# --------------------
# FEW_SHOT_EXAMPLES (short; keep the ones you already loaded in your file)
# --------------------
FEW_SHOT_EXAMPLES = [
    # (examples omitted here for brevity in this patch — keep your originals in real file)
]

# --------------------
# Document loader (same as before)
# --------------------
def load_documents(directory: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    p = Path(directory)
    files = sorted([f for f in p.glob("*.txt")])
    corpus = []
    meta = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        fname = f.stem
        pretty_name = re.sub(r'[_]+', ' ', fname).strip()
        first_line = text.splitlines()[0].strip() if text.splitlines() else pretty_name
        aliases = set()
        aliases.add(pretty_name.lower())
        header_short = re.split(r'[-–—:,(]', first_line)[0].strip()
        if header_short:
            aliases.add(header_short.lower())
        cas_match = re.search(r'CAS[:#]?\s*([0-9\-]{3,20})', text, re.I)
        cas = cas_match.group(1).strip() if cas_match else None
        if cas:
            aliases.add(cas)
        formula_match = re.search(r'Formula\s*[:\s]*([A-Za-z0-9\(\)\+\-]+)', text, re.I)
        formula = formula_match.group(1).strip() if formula_match else None
        if formula:
            aliases.add(formula.lower())
        syn_match = re.search(r'Synonyms/Trade Names[:\s]*(.+?)(?:\n|$)', text, re.I)
        synonyms = []
        if syn_match:
            syn_str = syn_match.group(1)
            raw_syns = re.split(r';|\(|\)|,', syn_str)
            for s in raw_syns:
                s2 = s.strip()
                if len(s2) > 2:
                    synonyms.append(s2)
                    aliases.add(s2.lower())
        corpus.append(text)
        meta.append({
            "filename": f.name,
            "pretty_name": pretty_name,
            "header": first_line,
            "aliases": list(aliases),
            "cas": cas,
            "formula": formula,
            "synonyms": synonyms,
            "text": text
        })
    return corpus, meta

# --------------------
# Vectorizers
# --------------------
def build_vectorizers(corpus: List[str]):
    word_v = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=20000)
    X_word = word_v.fit_transform(corpus)
    char_v = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6), max_features=20000)
    X_char = char_v.fit_transform(corpus)
    return (word_v, X_word), (char_v, X_char)


def cosine_sim(q_vec, X):
    sims = (X @ q_vec.T).toarray().ravel()
    return sims

# --------------------
# Helper: create a short conversation summary from chat_history
# --------------------
def summarize_history(chat_history: List[Dict], max_turns: int = 6) -> str:
    if not chat_history:
        return ""
    last = chat_history[-max_turns:]
    lines = []
    for m in last:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            texts = []
            for itm in content:
                if isinstance(itm, dict) and itm.get("type") == "text":
                    texts.append(itm.get("text", ""))
            content_text = " ".join(texts).strip() or "(image provided)"
        else:
            content_text = str(content)
        content_text = re.sub(r'\s+', ' ', content_text)
        if len(content_text) > 200:
            content_text = content_text[:197] + "..."
        lines.append(f"{role.capitalize()}: {content_text}")
    return "\n".join(lines)

# --------------------
# Search logic (CAS -> formula -> alias -> TFIDF) with fixes
# --------------------
def search_documents(query: str, corpus, meta, word_pair, char_pair, top_k=TOP_K) -> List[Dict]:
    q = query.strip()
    q_lower = q.lower()
    results = []

    # 1) CAS exact match
    cas_match = re.search(r'\b(\d{2,7}-\d{2}-\d)\b', q)
    if cas_match:
        cas_q = cas_match.group(1)
        for i, m in enumerate(meta):
            if m.get("cas") and cas_q in (m["cas"] or ""):
                results.append({"text": corpus[i], "source": m["filename"], "score": 1.0, "method": "cas"})
                return results

    # 2) formula exact match (tokens)
    tokens = re.findall(r'[A-Za-z0-9\(\)\+\-]+', q)
    token_set = set(t.lower() for t in tokens)
    for i, m in enumerate(meta):
        fm = m.get("formula")
        if fm and fm.lower() in token_set:
            results.append({"text": corpus[i], "source": m["filename"], "score": 0.98, "method": "formula"})
    if results:
        return results[:top_k]

    # 3) alias match but safer: require alias length >= ALIAS_MIN_LEN and match whole word when possible
    alias_hits = []
    for i, m in enumerate(meta):
        for a in m["aliases"]:
            if not a:
                continue
            a_clean = a.strip().lower()
            if len(a_clean) < ALIAS_MIN_LEN:
                continue
            if " " in a_clean:
                if a_clean in q_lower:
                    alias_hits.append((i, 0.95))
                    break
            else:
                if re.search(r'\b' + re.escape(a_clean) + r'\b', q_lower):
                    alias_hits.append((i, 0.95))
                    break
    if alias_hits:
        seen = set()
        out = []
        for idx, score in alias_hits:
            if idx not in seen:
                seen.add(idx)
                out.append({"text": corpus[idx], "source": meta[idx]["filename"], "score": score, "method": "alias"})
        return out[:top_k]

    # 4) TF-IDF fallback with threshold guard
    word_v, X_word = word_pair
    char_v, X_char = char_pair
    q_word = word_v.transform([q])
    q_char = char_v.transform([q])
    sim_word = cosine_sim(q_word, X_word)
    sim_char = cosine_sim(q_char, X_char)
    combined = 0.45 * sim_word + 0.55 * sim_char
    max_score = float(np.max(combined)) if combined.size else 0.0
    if max_score < TFIDF_SCORE_THRESHOLD:
        return []
    idxs = np.where(combined >= TFIDF_SCORE_THRESHOLD)[0]
    if idxs.size == 0:
        return []
    ranked = idxs[np.argsort(combined[idxs])[::-1]]
    results = []
    for idx in ranked[:top_k]:
        results.append({"text": corpus[idx], "source": meta[idx]["filename"], "score": float(combined[idx]), "method": "tfidf"})
    return results

# --------------------
# Compose messages (adds history summary into the system content)
# --------------------
def compose_messages(system_prompt: str, retrieved: List[Dict], few_shots: List[Dict],
                     user_content_items: List[Dict], chat_history: List[Dict]) -> List[Dict]:
    retrieved_block = "RETRIEVED PASSAGES:\n"
    for r in retrieved:
        snippet = r["text"][:600].replace("\n", " ")
        method = r.get("method", "tfidf")
        note = " (may not be relevant)"
        retrieved_block += f"- Source: {r['source']} (score={r['score']:.3f}; method={method}){note}\n  {snippet}\n\n"

    few_shot_block = "\nFEW_SHOT_EXAMPLES:\n"
    for ex in few_shots:
        few_shot_block += f"INPUT: {ex['user']}\nOUTPUT_JSON: {json.dumps(ex['assistant_json'])}\n\n"

    history_summary = summarize_history(chat_history, max_turns=6)
    history_block = f"\nCONVERSATION_SUMMARY:\n{history_summary}\n\n" if history_summary else ""

    system_content = system_prompt + "\n\n" + history_block + retrieved_block + "\n" + few_shot_block
    messages = [{"role": "system", "content": system_content}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_content_items})
    return messages

# --------------------
# Call OpenRouter
# --------------------
def call_openrouter(messages: List[Dict]) -> Dict:
    payload = {
        "model": GROK_MODEL,
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0.0
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# --------------------
# Synthesize paragraph (same)
# --------------------
def synthesize_paragraph_from_struct(parsed: Dict[str, Any]) -> str:
    hazards = ", ".join(parsed.get("hazards", [])) or "Potential hazards unknown"
    ppe_req = ", ".join(parsed.get("ppe_required", [])) or "standard lab PPE"
    ppe_rec = ", ".join(parsed.get("ppe_recommended", [])) or ""
    immediate = " ".join(parsed.get("immediate_actions", [])[:2]) if parsed.get("immediate_actions") else ""
    explain = parsed.get("explain_short") or ""
    paragraph = f"{explain} Primary hazards: {hazards}. Required PPE: {ppe_req}."
    if ppe_rec:
        paragraph += f" Recommended: {ppe_rec}."
    if immediate:
        paragraph += f" Immediate actions: {immediate}."
    return paragraph

# --------------------
# Interactive assistant class (uses improved search and summary)
# --------------------
class LabSafetyAssistantV3:
    def __init__(self, docs_dir=OUTPUT_DIR):
        print("Loading SDS documents from:", docs_dir)
        self.corpus, self.meta = load_documents(docs_dir)
        if not self.corpus:
            raise RuntimeError(f"No documents found in {docs_dir}.")
        print(f"Loaded {len(self.corpus)} documents.")
        print("Building TF-IDF vectorizers...")
        self.word_pair, self.char_pair = build_vectorizers(self.corpus)
        print("Vectorizers ready.")
        self.chat_history: List[Dict] = []
        intro_text = ("Hello — I'm Lab Safety Assistant. Tell me about your planned experiment or upload a photo (type 'image:<URL or dataURL>'). "
                      "I'll identify potential hazards, required PPE, and high-level safety advice. I will cite SDS passages when available.")
        print("\n" + intro_text + "\n")
        self.intro_text = intro_text

    def user_input_to_content(self, raw_input: str) -> Tuple[List[Dict], str]:
        raw_input = raw_input.strip()
        m = re.match(r'(?i)^\s*image\s*:\s*(.+)$', raw_input)
        if m:
            image_url = m.group(1).strip()
            text_part = "(image provided)"
            return [{"type": "text", "text": text_part}, {"type": "image_url", "image_url": {"url": image_url}}], text_part
        else:
            return [{"type": "text", "text": raw_input}], raw_input

    def query(self, raw_input: str) -> Dict:
        user_content_items, retrieval_query = self.user_input_to_content(raw_input)
        retrieved = search_documents(retrieval_query, self.corpus, self.meta, self.word_pair, self.char_pair, top_k=TOP_K)
        messages = compose_messages(SYSTEM_PROMPT, retrieved, FEW_SHOT_EXAMPLES, user_content_items, self.chat_history)
        resp = call_openrouter(messages)
        assistant_content = resp["choices"][0]["message"]["content"]
        self.chat_history.append({"role": "user", "content": user_content_items})
        self.chat_history.append({"role": "assistant", "content": assistant_content})
        try:
            parsed = json.loads(assistant_content)
        except Exception:
            parsed = {"raw_text": assistant_content}

        # Ensure an 'official_response' exists
        if isinstance(parsed, dict) and "official_response" not in parsed:
            if any(k in parsed for k in ["hazards", "ppe_required", "explain_short"]):
                parsed["official_response"] = synthesize_paragraph_from_struct(parsed)
            else:
                parsed["official_response"] = "I could not format a paragraph summary. Please consult SDS/EHS for authoritative guidance."

        # Attach retrieved metadata for UI convenience
        parsed["retrieved_sources"] = [r.get("source") for r in retrieved]
        parsed["retrieved_meta"] = retrieved

        return {"parsed": parsed, "raw_model_response": resp, "retrieved": retrieved}

# --------------------
# Run loop (CLI) omitted in module context; use LabSafetyAssistantV3 in app.py
# --------------------



