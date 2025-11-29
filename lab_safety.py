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
    {
        "user": "I will heat concentrated HCl on a hotplate and then neutralize with NaOH on the bench. Photo caption: beaker on hotplate, HCl bottle visible, no gloves.",
        "assistant_json": {
            "hazards": [
                "corrosive burns to eyes/skin from HCl and NaOH",
                "respiratory irritation from HCl vapors increased by heating",
                "exothermic neutralization splash risk",
                "corrosion of metal surfaces"
            ],
            "ppe_required": ["splash goggles", "lab coat", "nitrile gloves"],
            "ppe_recommended": ["face shield for pouring", "use fume hood for heating"],
            "immediate_actions": [
                "flush eyes/skin with water for 15 minutes if contact occurs",
                "move to fresh air if inhalation occurs and notify EHS"
            ],
            "safer_substitutes": [
                "use dilute solutions consistent with your experimental goals",
                "perform heating inside a fume hood or use a controlled water bath"
            ],
            "citations": ["Hydrogen_chloride.txt", "Sodium_hydroxide.txt"],
            "confidence": "high",
            "explain_short": "Heating HCl raises vapor release and inhalation risk; both HCl and NaOH can cause severe burns—use splash PPE and engineering controls.",
            "official_response": (
                "Do not heat concentrated hydrochloric acid on an open bench. Heating increases HCl vapor release, which can irritate or damage the respiratory tract, and neutralization with NaOH is exothermic and can cause splashing. "
                "At minimum wear splash goggles, a lab coat, and chemically resistant nitrile gloves, and perform heating inside a functioning fume hood. Have eyewash and safety shower access and consult your supervisor or EHS before proceeding."
            )
        }
    },
    {
        "user": "I'm going to distill acetone on a benchtop to purify it. Any major concerns?",
        "assistant_json": {
            "hazards": ["high flammability and fire risk from acetone vapors", "inhalation of vapors causing dizziness or irritation", "ignition from hotplate or static"],
            "ppe_required": ["splash goggles", "lab coat"],
            "ppe_recommended": ["nitrile gloves when handling, use of fume hood, fire extinguisher nearby"],
            "immediate_actions": ["stop and evacuate if fire occurs; notify EHS; ventilate area if heavy vapor release occurs"],
            "safer_substitutes": ["purchase pre-purified solvent or use contract purification", "use lower-volatility solvent if compatible"],
            "citations": ["Acetone.txt"],
            "confidence": "high",
            "explain_short": "Acetone is highly flammable—do not distill on an open benchtop; use proper ventilation and hot-work controls.",
            "official_response": (
                "Do not perform acetone distillation on an open benchtop because acetone vapors are highly flammable and can form explosive mixtures with air. Use an appropriate hood or ventilated, rated distillation setup and keep ignition sources away. "
                "Wear splash goggles and a lab coat, and ensure a suitable fire extinguisher is on hand. If you lack proper controls, consult EHS or your supervisor."
            )
        }
    },
    {
        "user": "I need to weigh powdered sodium azide to make a small solution. What should I know?",
        "assistant_json": {
            "hazards": [
                "systemic toxicity by ingestion or inhalation",
                "formation of explosive metal azides on contact with metals",
                "dust inhalation hazard"
            ],
            "ppe_required": ["nitrile gloves", "splash goggles", "lab coat"],
            "ppe_recommended": ["work in fume hood, use plastic spatulas and avoid metal contact", "respiratory protection if dust is unavoidable"],
            "immediate_actions": ["avoid creating dust, isolate area and contact EHS for spills/exposures", "seek immediate medical attention for suspected ingestion or significant exposure"],
            "safer_substitutes": ["use commercially prepared aqueous sodium azide solutions to avoid solid handling"],
            "citations": ["Sodium_azide.txt"],
            "confidence": "high",
            "explain_short": "Sodium azide is highly toxic and can form explosive metal azides—minimize solid handling and consult EHS for protocols.",
            "official_response": (
                "Handling solid sodium azide is high risk: it is toxic, and in contact with metals can form shock-sensitive metal azides. Avoid weighing the solid if possible—use pre-made solutions from a reputable supplier. "
                "If you must handle it, work in a fume hood, wear gloves and eye protection, avoid metal tools and containers, and consult EHS for spill and waste procedures."
            )
        }
    },
    {
        "user": "We'll use potassium permanganate to oxidize an organic substrate in a hood. Any safety notes?",
        "assistant_json": {
            "hazards": ["strong oxidizer—can cause fires with organic material", "eye/skin irritation", "dust inhalation from solids"],
            "ppe_required": ["splash goggles", "lab coat", "nitrile gloves"],
            "ppe_recommended": ["perform in fume hood", "minimize quantities and avoid mixing with organics outside of controlled procedure"],
            "immediate_actions": ["isolate area and extinguish small fires per institutional policy; contact EHS for larger incidents"],
            "safer_substitutes": ["use milder oxidants or catalytic methods if compatible and approved by supervisor"],
            "citations": ["Potassium_permanganate.txt"],
            "confidence": "high",
            "explain_short": "Potassium permanganate is a powerful oxidizer—avoid uncontrolled contact with organics and use engineering controls.",
            "official_response": (
                "Potassium permanganate is a strong oxidizing agent and can ignite organic materials on contact. Use minimal reagent amounts, perform work inside a functioning hood, and wear splash goggles, gloves, and a lab coat. "
                "If a fire or uncontrolled reaction occurs, evacuate per your lab's emergency plan and notify EHS."
            )
        }
    },
    {
        "user": "Working with concentrated nitric acid outside a hood — is that acceptable?",
        "assistant_json": {
            "hazards": ["chemical burns", "toxic corrosive fumes (inhalation)", "oxidizing agent hazards"],
            "ppe_required": ["acid-resistant gloves", "splash goggles", "lab coat"],
            "ppe_recommended": ["face shield for pouring", "perform inside fume hood"],
            "immediate_actions": ["stop the operation until in appropriate engineering control; flush exposures with water and contact EHS"],
            "safer_substitutes": ["use lower concentrations or alternative reagents if experimentally valid and permitted by EHS"],
            "citations": ["Nitric_acid.txt"],
            "confidence": "high",
            "explain_short": "Concentrated nitric acid produces corrosive fumes and must be handled in a fume hood with acid PPE.",
            "official_response": (
                "Do not handle concentrated nitric acid outside a fume hood. It emits corrosive and toxic fumes and can cause severe burns. Use acid-resistant gloves, splash goggles, and a face shield for high splash risk, and perform work inside a functioning hood. "
                "If exposure occurs, flush with water and seek medical attention; consult EHS for disposal and spill response."
            )
        }
    },
    {
        "user": "I will use liquid nitrogen to chill samples briefly. Any hazards and PPE?",
        "assistant_json": {
            "hazards": ["cryogenic cold burns/frostbite", "asphyxiation risk in confined spaces due to oxygen displacement", "pressure buildup if stored in sealed containers"],
            "ppe_required": ["cryogenic gloves/gauntlets", "splash goggles", "face shield", "lab coat and long pants"],
            "ppe_recommended": ["use in well-ventilated area", "closed-toe shoes"],
            "immediate_actions": ["if skin contact occurs, slowly warm affected area; move to fresh air if inhalation effects occur and contact EHS for large releases"],
            "safer_substitutes": ["use refrigerated chillers for small sample cooling if compatible"],
            "citations": ["Liquid_nitrogen.txt"],
            "confidence": "high",
            "explain_short": "Liquid nitrogen can cause severe cold burns and displace oxygen—use cryogenic PPE and ensure good ventilation.",
            "official_response": (
                "Liquid nitrogen can produce severe frostbite and can displace oxygen in confined spaces, causing asphyxiation. Use proper cryogenic gloves, eye/face protection, and conduct work in a well-ventilated area. Never store LN2 in sealed containers. "
                "If in doubt about ventilation or PPE, consult EHS before proceeding."
            )
        }
    },
    {
        "user": "I plan to use compressed oxygen cylinders to feed a reaction. What precautions?",
        "assistant_json": {
            "hazards": ["pressure hazards from cylinders", "oxygen-enriched atmospheres that increase fire risk", "cylinder as projectile if valve damaged"],
            "ppe_required": ["safety glasses", "lab coat"],
            "ppe_recommended": ["secure cylinder to fixed support, use appropriate regulators, avoid oil/grease on fittings"],
            "immediate_actions": ["if a leak occurs, move to fresh air and notify EHS; isolate area and follow cylinder emergency procedures"],
            "safer_substitutes": ["use smaller-scale oxygen sources or alternative oxidants if approved by EHS"],
            "citations": [],
            "confidence": "low",
            "explain_short": "Compressed oxygen cylinders present pressure and fire-enrichment hazards—secure cylinders and avoid contamination of fittings.",
            "official_response": (
                "Compressed oxygen cylinders are high-pressure and can create oxygen-enriched atmospheres that dramatically increase fire risk. Secure cylinders, use approved regulators, and never use oil or grease on fittings. Ensure good ventilation and keep ignition sources away. However, I could not reference an SDS; please consult your supervisor as well."
                "If you are unfamiliar with cylinder handling, consult EHS or your supervisor before using."
            )
        }
    },
    {
        "user": "Using a vacuum pump to remove solvent — any special concerns?",
        "assistant_json": {
            "hazards": ["exposure to solvent vapors (toxic or flammable)", "risk of drawing flammable vapors into pumps/exhaust", "glassware implosion under vacuum"],
            "ppe_required": ["splash goggles", "lab coat"],
            "ppe_recommended": ["use cold traps and proper ventilation, explosion-proof pumps for flammable solvents, shield glassware"],
            "immediate_actions": ["if glass implosion occurs, evacuate area and notify EHS; ventilate area if solvent vapors are present"],
            "safer_substitutes": ["use closed-loop solvent recovery or serviced rotary evaporators with safety features"],
            "citations": [],
            "confidence": "low",
            "explain_short": "Vacuum removal of solvents can expose you to toxic/flammable vapors and implosion risk—use traps, shields, and rated pumps.",
            "official_response": (
                "Using a vacuum pump with volatile solvents can expose personnel to toxic or flammable vapors and risks of glassware implosion. Use appropriate cold traps, explosion-rated pumps when needed, and shield glassware. Ensure proper ventilation and consult EHS if unsure about the configuration or solvent class. However, since no SDS document was provided, please also consult your supervisor before proceeding."
            )
        }
    },
    {
        "user": "Mixing an oxidizer with an organic substrate in a hood — what should I watch out for?",
        "assistant_json": {
            "hazards": ["ignition or explosion from oxidizer-organic contact", "toxic vapors and reactive intermediates"],
            "ppe_required": ["splash goggles", "lab coat", "chemical-resistant gloves"],
            "ppe_recommended": ["perform in fume hood with minimal quantities", "have fire suppression available"],
            "immediate_actions": ["if an uncontrolled reaction occurs, evacuate and notify EHS; follow institutional emergency response"],
            "safer_substitutes": ["use less reactive oxidants or catalytic methods if compatible and approved"],
            "citations": ["Potassium_permanganate.txt", "Perchloric_acid.txt"],
            "confidence": "medium",
            "explain_short": "Oxidizers plus organics can ignite or explode—only do such reactions with strict controls and EHS approval.",
            "official_response": (
                "Combining oxidizers and organic materials can produce fires or explosions. Use the smallest practical amounts, perform the work in a functioning hood, and ensure your supervisor and EHS have approved the protocol. Wear full splash PPE and keep extinguishing equipment nearby."
            )
        }
    },
    {
        "user": "Titration of ammonia (aq) with HCl — are there extra hazards compared to HCl/NaOH?",
        "assistant_json": {
            "hazards": ["pungent ammonia vapors causing respiratory irritation", "eye/skin irritation from splashes", "exothermic neutralization potential"],
            "ppe_required": ["splash goggles", "lab coat", "nitrile gloves"],
            "ppe_recommended": ["use fume hood if higher concentrations or strong odors are present"],
            "immediate_actions": ["flush exposed areas with water; move to fresh air for inhalation; consult EHS for large exposures"],
            "safer_substitutes": ["use dilute ammonia solutions or perform simulations when possible"],
            "citations": ["Ammonia.txt", "Hydrogen_chloride.txt"],
            "confidence": "high",
            "explain_short": "Ammonia adds vapor inhalation risk beyond standard acid-base hazards; use ventilation and splash PPE.",
            "official_response": (
                "Titrating ammonia introduces additional inhalation risk because ammonia vapors are pungent and can irritate the respiratory tract. Use splash goggles, gloves, and a lab coat; perform the procedure in a fume hood if the ammonia is volatile or concentrated. "
                "If you lack a hood or adequate ventilation, use dilute solutions or consider an alternative measurement method and consult EHS."
            )
        }
    },
    {
        "user": "Will phenolphthalein as an indicator cause a big safety problem?",
        "assistant_json": {
            "hazards": ["eye/skin irritation from indicator solutions", "historical carcinogenicity concerns (product-dependent)"],
            "ppe_required": ["splash goggles", "nitrile gloves", "lab coat"],
            "ppe_recommended": ["work in well-ventilated area; avoid aerosolization"],
            "immediate_actions": ["flush eyes/skin with water for 15 min and seek medical attention if needed", "consult SDS/EHS for the exact reagent"],
            "safer_substitutes": ["use alternative indicators or a pH meter when available"],
            "citations": [],   # no SDS retrieved in this example — be explicit about uncertainty
            "confidence": "low",
            "explain_short": "Phenolphthalein has historical safety concerns—check the product SDS and consult EHS before routine use.",
            "official_response": (
                "Phenolphthalein is commonly used as an acid–base indicator, but some historical reports raised safety concerns. If you cannot locate the reagent's SDS, treat it as a potential irritant and use splash goggles, gloves, and a lab coat. "
                "For routine teaching use, consider alternatives or a pH meter and consult EHS for product-specific guidance."
            )
        }
    }
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
    """
    Create a compact summary of the last few turns. This is high-signal and short,
    and will be inserted into system content so the model reliably sees the context.
    """
    if not chat_history:
        return ""
    # take last max_turns entries (mix of user/assistant)
    last = chat_history[-max_turns:]
    lines = []
    for m in last:
        role = m.get("role", "")
        content = m.get("content", "")
        # content may be list (user with image). extract text parts if so.
        if isinstance(content, list):
            # find first text item
            texts = []
            for itm in content:
                if isinstance(itm, dict) and itm.get("type") == "text":
                    texts.append(itm.get("text", ""))
            content_text = " ".join(texts).strip() or "(image provided)"
        else:
            # string content
            content_text = str(content)
        # truncate politely
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
            # if alias contains whitespace (multi-word), allow substring match
            if " " in a_clean:
                if a_clean in q_lower:
                    alias_hits.append((i, 0.95))
                    break
            else:
                # match whole word to avoid "an" matching everything
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
        # no reliable TF-IDF matches — return empty list rather than arbitrary docs
        return []
    # otherwise return only docs above threshold, sorted descending
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

    # summary of last conversation turns to make memory explicit
    history_summary = summarize_history(chat_history, max_turns=6)
    history_block = f"\nCONVERSATION_SUMMARY:\n{history_summary}\n\n" if history_summary else ""

    system_content = system_prompt + "\n\n" + history_block + retrieved_block + "\n" + few_shot_block
    messages = [{"role": "system", "content": system_content}]
    # append previous turns (already in API message shape)
    if chat_history:
        messages.extend(chat_history)
    # finally add current user message (user_content_items is a list per OpenRouter image spec)
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
        # store assistant intro in history as assistant content (string)
        self.chat_history.append({"role": "assistant", "content": intro_text})

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
        # push user message and assistant reply into history for next turns
        self.chat_history.append({"role": "user", "content": user_content_items})
        # store assistant reply as string content (keeps history readable)
        self.chat_history.append({"role": "assistant", "content": assistant_content})
        # parse JSON if possible
        try:
            parsed = json.loads(assistant_content)
        except Exception:
            parsed = {"raw_text": assistant_content}
        # ensure official_response
        if isinstance(parsed, dict) and "official_response" not in parsed:
            if any(k in parsed for k in ["hazards", "ppe_required", "explain_short"]):
                parsed["official_response"] = synthesize_paragraph_from_struct(parsed)
            else:
                parsed["official_response"] = "I could not format a paragraph summary. Please consult SDS/EHS for authoritative guidance."
        return {"parsed": parsed, "raw_model_response": resp, "retrieved": retrieved}

# --------------------
# Run loop
# --------------------
def main():
    assistant = LabSafetyAssistantV3(docs_dir=OUTPUT_DIR)
    print("Type a question or 'exit' to quit. To include an image, type: image:https://... or image:data:image/jpeg;base64,...\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        out = assistant.query(user_text)
        print("\nAssistant (parsed):")
        print(json.dumps(out["parsed"], indent=2))
        readable = []
        for r in out["retrieved"]:
            method = r.get("method", "tfidf")
            readable.append(f"{r['source']} [{method}]")
        print("\nRetrieved sources (some may not be relevant):", readable)
        print("\n---\n")

if __name__ == "__main__":

    main()
