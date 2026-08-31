""" framework is from json (for NCA only)
From NCA.json


PDF Document
 ↓
Gatekeeper (Relevance Check)
 ↓
Split → Sections
 ↓
FOR EACH Section:
    ↓
    Split → small chunks
    ↓
    For each chunk:
        BM25 + FAISS
    ↓
    Merge all candidates
    ↓
    Deduplicate
    ↓
    Cross-Encoder (Section vs Candidate)
    ↓
    Top-K results
    ↓
    Context Assembly (Parent chunks)
    ↓
    LLM (Micro-Evaluation for this section)
    ↓
    Store JSON result
END LOOP
 ↓
Collect all section results
 ↓
Deduplicate findings (across sections)
 ↓
Aggregate scores
 ↓
LLM (Final Evaluation / Executive Summary)
 ↓
Final JSON Output
"""

import os
import glob
import json
import math
import re
import random
import time
from typing import Optional, Dict, Any, List, NamedTuple

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document  
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi 

load_dotenv()

# --- TEMPORARY DIAGNOSTIC -------------------------------------------------
# Dumps each section's raw model response BEFORE parsing, next to what the
# parser extracted from it, so a detection failure (model says "compliant")
# can be told apart from a parsing failure (model reports violations that the
# parser drops). Set LYTREX_DEBUG_RAW=0 to silence, or delete this flag and
# the blocks that reference it once the question is settled.
DEBUG_RAW_RESPONSES = os.getenv("LYTREX_DEBUG_RAW", "1") == "1"


# --- LLM PROVIDER SELECTION ----------------------------------------------
# Each provider has its own free quota, so when one is exhausted or degraded
# the audit can be moved to another by changing PROVIDER in .env
# and restarting. Groq and Gemini use their native clients; Cerebras is
# reached through the OpenAI-compatible client pointed at its base URL.
PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "groq": {
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
        "base_url": None,  # native ChatGroq client, no base URL needed
        "console": "https://console.groq.com/keys",
        # Groq throttles on tokens-per-minute rather than request count, so
        # request pacing is left off here and the backoff handles TPM.
        "rpm": 0,
    },
    "cerebras": {
        "key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        # Cerebras publishes model ids without a vendor prefix, so this is
        # "gpt-oss-120b" rather than Groq's "openai/gpt-oss-120b". Override
        # with CEREBRAS_MODEL if the account exposes a different id.
        "default_model": "gpt-oss-120b",
        "base_url": "https://api.cerebras.ai/v1",
        "console": "https://cloud.cerebras.ai",
        "rpm": 0,
    },
    "gemini": {
        "key_env": "GOOGLE_API_KEY",
        "model_env": "GEMINI_MODEL",
        # gemini-3.6-flash was a preview model capped at 20 requests/day, which
        # is barely one audit. The whole 2.5 family (flash, flash-lite, pro)
        # returns 404 "no longer available to new users" on current keys, so
        # 3.5-flash is the stable choice that actually resolves. Its lite
        # sibling gemini-3.5-flash-lite trades some reasoning quality for a
        # higher request ceiling if throughput becomes the constraint.
        "default_model": "gemini-3.5-flash",
        # Gemini is not OpenAI-compatible; it gets its own client rather than
        # a base URL. Listed here so every provider reads the same way.
        "base_url": None,
        "console": "https://aistudio.google.com/apikey",
        # Gemini free tier caps requests per minute as well as per day.
        # 8/min sits under the usual 10-15 RPM free-tier ceiling with room
        # to spare. Raise it with LLM_RPM on a paid tier.
        "rpm": 8,
    },
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": "ANTHROPIC_MODEL",
        # Exact id, no date suffix -- "claude-haiku-4-5-20251001" style strings
        # are not the current form and will not resolve.
        "default_model": "claude-haiku-4-5",
        "base_url": None,
        "console": "https://console.anthropic.com/settings/keys",
        # Paid credit carries far higher limits than the free tiers, so this is
        # a safety net rather than a real constraint: 30/min is ~2s apart and
        # sits well under the entry-tier ceiling. Raise via LLM_RPM.
        "rpm": 30,
    },
}


def _resolve_provider() -> str:
    name = (os.getenv("PROVIDER") or "groq").strip().lower()
    if name not in PROVIDER_CONFIG:
        raise ValueError(
            f"PROVIDER='{name}' is not supported. "
            f"Use one of: {', '.join(sorted(PROVIDER_CONFIG))}."
        )
    return name


def _debug_dump(label: str, raw_text: str, parsed: Any, findings_key: str) -> None:
    """Prints one section's raw response and the parse result side by side."""
    if not DEBUG_RAW_RESPONSES:
        return
    print(f"\n=== RAW {label} ===")
    print(raw_text if raw_text.strip() else "<EMPTY RESPONSE>")
    print(f"=== END RAW {label} ===")
    if isinstance(parsed, dict):
        found = parsed.get(findings_key)
        print(f"--- PARSED {label} ---")
        print(f"    keys        : {sorted(parsed.keys())}")
        print(f"    '{findings_key}' present: {findings_key in parsed}")
        if isinstance(found, list):
            print(f"    '{findings_key}' count  : {len(found)}")
            for item in found[:3]:
                print(f"      * {str(item)[:160]}")
        else:
            print(f"    '{findings_key}' value  : {type(found).__name__} -> {str(found)[:160]}")
        print(f"--- END PARSED {label} ---\n")
    else:
        print(f"--- PARSED {label}: NOT A DICT ({type(parsed).__name__}) ---\n")


# --- FINDING CLASSIFICATION & SCORING -------------------------------------
# Two kinds of finding come back from the map phase and they are not
# equivalent evidence:
#
#   CONTRADICTION - the policy states something that conflicts with a control
#                   ("all employees receive global admin rights"). Real defect.
#   ABSENCE       - a control is not covered by this excerpt ("MFA is not
#                   documented"). Weaker evidence, and systematically
#                   over-reported: each section sees one fragment, so the same
#                   missing control is reported once per section that lacks it.
#
# Counting both at a flat rate let a single undocumented control cost as much
# as four separate contradictions, purely because the document was chunked.
_ABSENCE_RE = re.compile(
    r"\b("
    r"not\s+(?:explicitly\s+|clearly\s+|formally\s+)?(?:documented|mentioned|specified|"
    r"defined|addressed|covered|stated|described|included|present|provided|referenced|"
    r"established|detailed)"
    r"|no\s+(?:evidence|mention|reference|documentation|policy|procedure|statement|details?|"
    r"information|provision)"
    r"|does\s+not\s+(?:address|mention|specify|define|cover|document|describe|include|contain|"
    r"provide|establish)"
    r"|(?:is|are)\s+(?:absent|missing|undocumented)"
    r"|absent\s+from|missing\s+from"
    r"|lacks?\b|lacking\b"
    r"|silent\s+on"
    r"|fails?\s+to\s+(?:mention|document|specify|address|define|establish)"
    r"|omits?\b|omitted\b"
    r"|(?:this\s+)?(?:excerpt|section|chunk|fragment)\s+does\s+not"
    r"|outside\s+the\s+scope\s+of\s+this"
    # -- EXTENSION 1: "not <adverb>* <existence verb>" -----------------------
    # The shipped branch above covers "not documented/defined/established/...".
    # These four verbs complete the family: a control that was never identified,
    # implemented, approved, or put in place is absent, not contradicted.
    # The adverbs are optional and repeatable so "not yet implemented" and
    # "not consistently in place" are caught too.
    #
    # DELIBERATELY EXCLUDED: enforced, mandatory, tracked, conducted, encrypted,
    # monitored, configured. Those describe a control the policy HAS but
    # undermines ("MFA is documented but not enforced") -- a contradiction.
    r"|not\s+(?:yet\s+|currently\s+|explicitly\s+|clearly\s+|formally\s+|consistently\s+)*"
    r"(?:identified|implemented|approved|in\s+place)"
    # -- EXTENSION 2: "No <noun phrase> <existence verb>" --------------------
    # The single largest miss. "No HR cybersecurity requirements documented" is
    # the same claim as "HR cybersecurity requirements are not documented", but
    # only the latter matched, so the most clear-cut absences in the corpus were
    # scored as contradictions at 3x the weight.
    #
    # The gap is capped at six tokens, which is what the longest real finding
    # needs ("No cloud computing or hosting cybersecurity requirements
    # identified"). Tokens are word-ish only, so a comma, semicolon or full stop
    # ends the noun phrase and the match cannot run across a clause boundary.
    # Contrast conjunctions are excluded outright so a genuine breach phrased as
    # "no X, but Y is documented" is not swallowed by the trailing verb.
    r"|no(?:\s+(?!but\b|however\b|although\b|though\b|while\b|whereas\b|yet\b|"
    r"despite\b|except\b)[\w/&()-]+){0,6}"
    r"\s+(?:documented|identified|approved|implemented|defined|established|specified|"
    r"in\s+place)\b"
    # -- EXTENSION 3: explicit absence heads ---------------------------------
    # Already reachable through the "no (evidence|mention|reference)" branch
    # above; restated so the intent survives any future edit to that branch.
    r"|no\s+(?:mention|evidence|reference)\s+(?:of|to|that)"
    r")",
    re.IGNORECASE,
)

# Pulls the framework control out of the citation prefix, e.g.
# "[Company Page: 7 | Company Section: 2.1 | Framework Control: SEC_0007] ..."
_CONTROL_RE = re.compile(
    r"framework\s+(?:controls?|sections?)\s*:\s*([^|\]]+)", re.IGNORECASE
)

# A canonical control identifier inside that text: SEC_0007, AC-03, 2.1.1, ...
_CONTROL_ID_RE = re.compile(r"\b([A-Z]{2,}[_-]?\d{1,5}|\d+(?:\.\d+){1,3})\b")

# The severity the model assigned, read ONLY from the structured field the
# prompts now request inside the citation bracket:
#   [Company Page: 4 | Company Section: 3.2 | Framework section: 5-1 | Severity: CRITICAL]
# Synonyms are folded so a model that reaches for "HIGH" or "LOW" instead of
# the requested words is still understood rather than silently defaulted.
#
# DELIBERATELY NOT MATCHED: a bare "CRITICAL:" prefix on the finding text. The
# 2911fa67 run volunteered exactly that on 30 of 30 findings, so honouring it
# would double the failure mass of an entire report on a marker that carries
# zero discriminating information. Only the requested field counts.
_SEVERITY_RE = re.compile(
    r"severity\s*:\s*(critical|high|severe|major|moderate|medium|minor|low)\b",
    re.IGNORECASE,
)

# Maps what the model may write onto the three tiers that carry weights.
_SEVERITY_SYNONYMS: Dict[str, str] = {
    "critical": "critical", "high": "critical", "severe": "critical", "major": "critical",
    "moderate": "moderate", "medium": "moderate",
    "minor": "minor", "low": "minor",
}

# --- FALLBACK PENALTY MODEL ------------------------------------------------
# These drive the PENALTY ESTIMATE, which is now the fallback rather than the
# primary model: it is what the score falls back to when the coverage
# denominator (controls_assessed) is unavailable, which is the case for every
# report archived before this change. Exponents below 1 make each additional
# finding of a kind cost less than the last, so a thorough audit degrades
# smoothly instead of hitting the floor.
#
# Their values are deliberately UNCHANGED. An archived report recomputed today
# must still yield the number it was filed with, otherwise the archive stops
# being a record of what the pipeline said. Tune the coverage constants below
# instead; these exist to reproduce history.
CONTRADICTION_WEIGHT = 6.0
CONTRADICTION_EXPONENT = 0.9
ABSENCE_WEIGHT = 2.0
ABSENCE_EXPONENT = 0.8


def _env_float(name: str, default: float) -> float:
    """Reads a float override from the environment, in the LLM_RPM style."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[INIT] {name}={raw!r} is not a number; using default {default}.")
        return default


def _env_int(name: str, default: int) -> int:
    """Integer twin of _env_float, for sizes that must not become floats."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[INIT] {name}={raw!r} is not an integer; using default {default}.")
        return default


# --- MAP PHASE: SECTION SIZE ----------------------------------------------
# How the uploaded document is cut into the units the map phase audits, one
# LLM call each. These are the two knobs that set BOTH the cost of an audit
# (one map call per section) and the quality of its retrieval (the section
# text IS the retrieval query).
#
# MEASURED on the repo's 30-page NCA PDF: 500 chars yields 5.40 sections per
# page (162 sections); 1500 yields ~1.93 per page (~58), a 2.8x reduction in
# map calls and in reduce payload.
#
# UNVERIFIED: whether findings stay as good. At 500 chars the 12,000-char
# framework-context cap fired on 155 of 162 sections (96%), so the model was
# already reading far more framework text than company text. A 1500-char
# section is a 3x richer retrieval query and WILL retrieve differently. That
# can only be settled by a live run. Back off to 1000 with
# LYTREX_MAP_CHUNK_SIZE rather than editing this line.
MAP_CHUNK_SIZE = _env_int("LYTREX_MAP_CHUNK_SIZE", 1500)
MAP_CHUNK_OVERLAP = _env_int("LYTREX_MAP_CHUNK_OVERLAP", 300)


# --- COVERAGE MODEL: SEVERITY TIERS ---------------------------------------
# A finding's cost is TIER x KIND, not a flat per-finding constant, so that
# "shared administrator credentials" and "the backup retention period is not
# stated in this excerpt" stop costing the same thing.
#
# TIERS. Calibrated so moderate == 1.0 is the neutral unit and the other two
# are a stated multiple of it. critical is 5x minor, which is the spread a
# human auditor's critical/minor split actually implies -- wide enough that
# three critical breaches outrank eight minor gaps (3x2.0=6.0 > 8x0.4=3.2),
# which is the ordering the Acme calibration case exists to enforce.
SEVERITY_TIER_WEIGHTS: Dict[str, float] = {
    "critical": _env_float("LYTREX_SEVERITY_CRITICAL", 2.0),
    "moderate": _env_float("LYTREX_SEVERITY_MODERATE", 1.0),
    "minor": _env_float("LYTREX_SEVERITY_MINOR", 0.4),
}

# The tier a finding gets when the model did not label it. MEASURED reason for
# choosing the middle tier rather than the worst: across the five archived ECC
# reports, four carry no severity marker at all and the fifth marks 30 of 30
# findings CRITICAL. A model left to volunteer severity defaults to alarm, so
# an unlabelled finding is evidence of nothing about its severity and must not
# be scored as though it were the worst case. "moderate" also makes the
# default-tier coverage score depend only on the contradiction/absence split,
# which is the one classification this pipeline has actually validated.
DEFAULT_SEVERITY_TIER = (os.getenv("LYTREX_DEFAULT_SEVERITY") or "moderate").strip().lower()
if DEFAULT_SEVERITY_TIER not in SEVERITY_TIER_WEIGHTS:
    print(f"[INIT] LYTREX_DEFAULT_SEVERITY={DEFAULT_SEVERITY_TIER!r} is not one of "
          f"{sorted(SEVERITY_TIER_WEIGHTS)}; using 'moderate'.")
    DEFAULT_SEVERITY_TIER = "moderate"

# KIND. A stated breach is stronger evidence than an absence: "MFA is not
# required for remote access" is a property of the policy, while "MFA is not
# documented in this excerpt" is frequently a property of the CHUNKING -- the
# control was simply in a different section. Absences are therefore discounted
# rather than dropped, at every tier, so a critical absence still outranks a
# moderate one without ever reaching a contradiction of the same tier.
KIND_WEIGHT_CONTRADICTION = _env_float("LYTREX_KIND_CONTRADICTION", 1.0)
KIND_WEIGHT_ABSENCE = _env_float("LYTREX_KIND_ABSENCE", 0.45)

# --- COVERAGE MODEL: TRANSFER FUNCTION ------------------------------------
# strain = (critical_mass / assessed) ** COVERAGE_CRITICAL_EXPONENT
#          + (noncritical_mass / assessed)
# score  = 100 * exp(-COVERAGE_DECAY * strain)
#
# RECALIBRATED against live record ef15a3b0 (Acme InfoSec policy vs ECC):
# 31 map-phase findings, census critical 12 / moderate 10 / minor 9, failure
# mass 32.76 over 20 assessed controls -> ratio 1.638. The previous shape
# (strain = ratio, DECAY = 3.0) scored that 0.73. The reducer's own holistic
# read was 45 and the human estimate 30-45, so the transfer function, not the
# severity model, was wrong: severity DID discriminate on this run.
#
# WHY THE SHAPE CHANGED AND NOT JUST THE CONSTANT. MEASURED: no single decay
# can fix this. The live run (mass 32.76 / 20 = 1.638) and the catastrophic
# anchor of 25 critical contradictions (mass 50.0 / 27 = 1.852) have ratios
# only 13% apart, so under ANY function of the ratio alone they score within a
# few points of each other -- at DECAY 0.5 they are 44.09 and 39.62, a 4.5
# point gap. Lowering the decay far enough to lift the live run into 30-45
# therefore drags a genuinely catastrophic policy up with it. Separating them
# needs information the ratio does not carry: the SEVERITY MIX. The live run
# is 12 critical findings among 31 (average impact 1.06); the anchor is 25 of
# 25 (average impact 2.00).
#
# WHY THE CRITICAL MASS IS SPLIT OUT ADDITIVELY RATHER THAN AVERAGED. The
# obvious way to read the mix is mass/finding_count, i.e. scale a coverage
# term by average severity. MEASURED, that model is disqualified: it is NOT
# MONOTONE. Because dividing by the finding count lets a cheap finding pull
# the average down faster than it pushes the total up, adding one more minor
# absence, minor contradiction, or moderate absence to the live run RAISES its
# score. A scorer that rewards reporting one more problem is broken, whatever
# its calibration. Splitting the mass by tier and adding the parts keeps every
# finding strictly costly -- both terms only ever grow -- while still letting a
# uniformly critical report separate from a mixed one.
#
# CRITICAL_EXPONENT is why they separate. Critical breaches compound: a policy
# with one critical hole has a gap, a policy where most examined controls are
# critically breached has no posture at all, and the marginal harm of the
# twentieth is larger than the first. Non-critical findings accumulate
# linearly -- they are largely independent documentation defects. At 1.0 this
# reduces EXACTLY to the old strain = failure_mass / assessed, which is the
# backstop for reading the constant: 1.5 is a deliberate departure from a
# shape that is otherwise preserved.
#
# WHY EXPONENTIAL AND NOT LINEAR 100*(1-strain):
#   1. It cannot go negative, so no clamp is needed and no information is lost
#      to clamping. Linear coverage pins at 0 for every strain >= 1, which is
#      where badly failing documents actually land -- exactly the range the old
#      flat 100-5n model was replaced for flattening.
#   2. It never returns 0. A hard 0 asserts "nothing in this policy complies",
#      which this pipeline can no more support than a 100. See coverage_score.
#
# THE TRADEOFF THIS ACCEPTS, STATED. Pinning the live run at ~36 forces the
# mild end UP: 5 moderate contradictions over 27 controls moves from 57 to 89,
# and the synthetic 13-finding Acme-like case from 38 to 89. That is not a
# free choice, it is arithmetic. The live run's NON-critical mass alone is
# 10.96 over 20 controls, a density of 0.548, which is three times the whole
# 5-moderate case (0.185). Any monotone model that scores the live run 37
# therefore cannot score 5 moderate contradictions below about 87 -- the
# budget is spent before a single critical finding is counted. The curve is
# now forgiving of thin, low-severity finding sets and reserves its
# discrimination for the dense and the critical, which is the range real
# audits land in and the range this pipeline is asked to rank.
COVERAGE_DECAY = _env_float("LYTREX_COVERAGE_DECAY", 0.6)

# Super-linear cost of CRITICAL failure density only. 1.0 restores the
# previous pure-ratio model exactly. Not raised further than 1.5: at 2.0 the
# term shrinks sparse critical densities so hard (0.26 ** 2 = 0.07) that the
# 13-finding case with FOUR critical findings scores 92.2, above the
# 5-moderate case at 89.5 -- an ordering inversion the 1.5 value avoids.
COVERAGE_CRITICAL_EXPONENT = _env_float("LYTREX_COVERAGE_CRITICAL_EXPONENT", 1.5)

# Guard on the denominator. A section-starved run can assess two or three
# controls, and dividing by that turns one finding into a near-zero score on
# almost no evidence. Below this floor the ratio is computed as though this
# many controls were assessed, which damps the score rather than inflating it.
COVERAGE_MIN_ASSESSED = _env_float("LYTREX_COVERAGE_MIN_ASSESSED", 5.0)

# The lowest score the coverage model will report. exp() is asymptotic and
# never reaches zero, but rounding to two decimals does, so without this a
# sufficiently bad document still prints the 0.00 that this model exists to
# avoid claiming. Set far below any real audit -- it only binds past ratio
# ~3.07, i.e. a failure mass over three times the assessed scope -- so it
# costs no discrimination in the range real documents occupy.
COVERAGE_SCORE_FLOOR = _env_float("LYTREX_COVERAGE_FLOOR", 0.01)


# --- RETRIEVAL RELEVANCE FLOOR --------------------------------------------
# Each section used to be handed exactly k framework controls whether or not
# the reranker thought any of them were on topic, because the top-k slice was
# unconditional. A section about physical security therefore received
# cryptography controls, and the model correctly answered "these are not
# documented in this excerpt" -- an absence finding manufactured by retrieval
# rather than observed in the policy. That is why one control (SEC_0007) was
# reported missing from section after section.
#
# SCALE: self.reranker is a sentence-transformers CrossEncoder wrapping
# BAAI/bge-reranker-base. That checkpoint's config.json declares num_labels=1
# and sets no activation override, so CrossEncoder.get_default_activation_fn()
# returns nn.Sigmoid() and .predict() emits PROBABILITIES in (0, 1) -- NOT the
# unbounded ~-11..+11 logits that BAAI's own FlagReranker returns. The model's
# own relevant/not-relevant boundary is therefore 0.5, not 0. Any threshold
# copied from FlagReranker advice would be meaningless here.
#
# VALUE: 0.0 -- the floor is OFF by default, and this is a deliberate result
# rather than an oversight. Scored against known control/violation pairs from
# this very domain, the reranker rates real breaches as low as pure noise:
#
#   control                                   policy sentence                     score
#   "shared admin credentials prohibited"     "IT shares one admin account"      0.0024
#   "no admin rights by default"              "all employees get global admin"   0.0028
#   "backups encrypted with AES-256"          "backups written unencrypted"      0.0121
#   "MFA enforced for remote access"          "MFA is not required remotely"     0.0306
#   "logs retained twelve months"             "logs retained 30 days"            0.1951
#   (noise)                                   "cafeteria hours are 11:00-14:00"  0.0000
#
# Four of those five genuine violations fall below 0.05, and two fall below the
# score of a section whose best match was known-irrelevant noise (0.0076). There
# is no cut point that separates them, because the reranker measures semantic
# SIMILARITY while a violation is semantic OPPOSITION -- "MFA shall be enforced"
# and "MFA is not required" are near-antonyms, which is exactly the pair the
# audit exists to catch. Filtering on relevance therefore removes the findings
# that matter most while keeping bland, on-topic, compliant text.
#
# The duplicate-absence problem this was meant to solve is handled downstream
# instead, by classify_and_dedupe() collapsing absences per control -- a lossless
# fix, where this one silently drops evidence.
#
# The machinery is kept because it is sound and cheap to re-enable: set
# LYTREX_RERANK_MIN_SCORE to a small positive value to trade recall for a
# quieter report. Do not raise it without measuring against known violations
# first; in a compliance tool a missed breach is far worse than a noisy finding.
RERANK_MIN_SCORE = _env_float("LYTREX_RERANK_MIN_SCORE", 0.0)


# --- TOKEN ACCOUNTING: PRICES ---------------------------------------------
# Dollars per 1,000,000 tokens for the configured model. The defaults are the
# published claude-haiku-4-5 rates ($1.00 in / $5.00 out per MTok), which is
# what PROVIDER=anthropic resolves to today. A model or price change is one
# edit here, or an env override at deploy time -- nothing downstream carries a
# rate of its own.
#
# These are applied to the token counts the PROVIDER REPORTS, never to a
# chars/4 estimate. A provider that reports no usage contributes nothing to the
# cost line and is counted separately as an unmetered call, so "this run was
# free" can be told apart from "this provider does not report usage".
PRICE_INPUT_PER_MTOK = _env_float("LYTREX_PRICE_INPUT_PER_MTOK", 1.00)
PRICE_OUTPUT_PER_MTOK = _env_float("LYTREX_PRICE_OUTPUT_PER_MTOK", 5.00)
PRICE_MODEL_LABEL = (os.getenv("LYTREX_PRICE_MODEL_LABEL") or "claude-haiku-4-5").strip()


def _usage_from_message(message: Any) -> Optional[Dict[str, int]]:
    """
    Reads the per-call token counts off a LangChain response message.

    VERIFIED against the installed packages, not recalled. langchain-core
    declares the field on AIMessage as

        usage_metadata: UsageMetadata | None = None
        (site-packages/langchain_core/messages/ai.py:176)

    a TypedDict carrying input_tokens / output_tokens / total_tokens, and
    langchain-anthropic fills it in on the non-streaming path with

        msg.usage_metadata = _create_usage_metadata(data.usage)
        (site-packages/langchain_anthropic/chat_models.py:2212)

    _create_usage_metadata (same file, line 2932) adds cache_read and
    cache_creation back into input_tokens, because Anthropic's own
    input_tokens field EXCLUDES cached tokens. So the value read here is the
    true total input for the call, not the uncached remainder.

    The base chat model also merges the raw provider payload into
    .response_metadata, which is the fallback for any client that leaves
    usage_metadata unset: Anthropic writes 'usage' {input_tokens,
    output_tokens}, the OpenAI-compatible clients (Groq, Cerebras) write
    'token_usage' {prompt_tokens, completion_tokens}. Returns None when
    nothing is reported -- a normal outcome on some providers, and never a
    reason to raise inside an audit.
    """
    if message is None:
        return None

    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict):
        in_tok = meta.get("input_tokens")
        out_tok = meta.get("output_tokens")
        if isinstance(in_tok, int) or isinstance(out_tok, int):
            return {"input_tokens": int(in_tok or 0),
                    "output_tokens": int(out_tok or 0)}

    rm = getattr(message, "response_metadata", None)
    if isinstance(rm, dict):
        for key, in_key, out_key in (
            ("usage", "input_tokens", "output_tokens"),
            ("token_usage", "prompt_tokens", "completion_tokens"),
            ("usage_metadata", "prompt_token_count", "candidates_token_count"),
        ):
            raw = rm.get(key)
            if isinstance(raw, dict):
                in_tok = raw.get(in_key)
                out_tok = raw.get(out_key)
                if isinstance(in_tok, int) or isinstance(out_tok, int):
                    return {"input_tokens": int(in_tok or 0),
                            "output_tokens": int(out_tok or 0)}
    return None


class TokenLedger:
    """
    Per-audit token and cost accounting, split by phase.

    Three phases are tracked separately because they have three different cost
    SHAPES, and a single total hides which one to fix:

      gate   - one call per audit, one page of text. Flat cost.
      map    - one call PER SECTION, so it scales with document length. This
               is the term that dominates on a long document.
      reduce - one call for a document that fits in a single batch, and
               otherwise one call per batch plus the calls that merge the
               partials (see batch_reports_by_size). Every one of them is
               recorded here, so the reduce line is a call COUNT, not a
               presumed 1.
    """

    PHASE_ORDER = ("gate", "map", "reduce")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phases: Dict[str, Dict[str, int]] = {}
        self.unmetered_calls = 0

    def record(self, phase: str, message: Any) -> None:
        """
        Adds one call to a phase. Deliberately never raises: a bookkeeping bug
        must not be able to destroy an audit that already cost money to run.
        """
        try:
            bucket = self._phases.setdefault(
                phase, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            bucket["calls"] += 1
            usage = _usage_from_message(message)
            if usage is None:
                self.unmetered_calls += 1
                return
            bucket["input_tokens"] += usage["input_tokens"]
            bucket["output_tokens"] += usage["output_tokens"]
        except Exception as e:
            print(f"[LYTREX] token accounting skipped one {phase} call: {e}")

    @staticmethod
    def cost_usd(input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * PRICE_INPUT_PER_MTOK
                + output_tokens * PRICE_OUTPUT_PER_MTOK) / 1_000_000.0

    def summary(self) -> Dict[str, Any]:
        """The payload written into the report under 'token_usage'."""
        phases: Dict[str, Any] = {}
        total_in = total_out = total_calls = 0
        ordered = [p for p in self.PHASE_ORDER if p in self._phases]
        ordered += [p for p in sorted(self._phases) if p not in self.PHASE_ORDER]
        for name in ordered:
            b = self._phases[name]
            phases[name] = {
                "calls": b["calls"],
                "input_tokens": b["input_tokens"],
                "output_tokens": b["output_tokens"],
                "total_tokens": b["input_tokens"] + b["output_tokens"],
                "estimated_cost_usd": round(
                    self.cost_usd(b["input_tokens"], b["output_tokens"]), 6
                ),
            }
            total_in += b["input_tokens"]
            total_out += b["output_tokens"]
            total_calls += b["calls"]
        return {
            "model": PRICE_MODEL_LABEL,
            "price_input_per_mtok": PRICE_INPUT_PER_MTOK,
            "price_output_per_mtok": PRICE_OUTPUT_PER_MTOK,
            "calls": total_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": round(self.cost_usd(total_in, total_out), 6),
            "unmetered_calls": self.unmetered_calls,
            "phases": phases,
        }

    def render(self, summary: Optional[Dict[str, Any]] = None) -> str:
        """The console block. Exactly the numbers summary() archives."""
        s = summary if summary is not None else self.summary()
        bar = "=" * 74
        lines = [
            "",
            bar,
            f"[LYTREX] TOKEN & COST BREAKDOWN  (model: {s['model']} | "
            f"${s['price_input_per_mtok']:.2f}/MTok in, "
            f"${s['price_output_per_mtok']:.2f}/MTok out)",
            bar,
            f"  {'phase':<8}{'calls':>7}{'input tok':>14}{'output tok':>14}{'cost $':>12}",
        ]
        for name, p in s["phases"].items():
            lines.append(
                f"  {name:<8}{p['calls']:>7}{p['input_tokens']:>14,}"
                f"{p['output_tokens']:>14,}{p['estimated_cost_usd']:>12.4f}"
            )
        lines.append("  " + "-" * 70)
        lines.append(
            f"  {'TOTAL':<8}{s['calls']:>7}{s['input_tokens']:>14,}"
            f"{s['output_tokens']:>14,}{s['estimated_cost_usd']:>12.4f}"
        )
        in_cost = s["input_tokens"] * s["price_input_per_mtok"] / 1_000_000.0
        out_cost = s["output_tokens"] * s["price_output_per_mtok"] / 1_000_000.0
        lines.append(
            f"  input ${in_cost:.4f}  +  output ${out_cost:.4f}  "
            f"=  TOTAL ${s['estimated_cost_usd']:.4f}"
        )
        if s["unmetered_calls"]:
            lines.append(
                f"  NOTE: {s['unmetered_calls']} call(s) reported no usage, so the "
                f"figures above are a LOWER BOUND."
            )
        lines.append(bar)
        return "\n".join(lines)


# --- REDUCE PHASE: HIERARCHICAL MERGE -------------------------------------
# THE DEFECT THIS REPLACES. The reducer used to be handed
# prune_text(json.dumps(all_reports, indent=2), 60000), and prune_text is
# text[:max_chars]. A serialised section report measures ~713-1305 chars
# depending on finding density, so the cap was reached somewhere between 46
# and 84 sections -- roughly 9-15 pages. Past that the reducer always saw the
# SAME first N sections, cut mid-JSON, and every later section's findings never
# reached the merge at all. It was silent because the score is computed by
# score_basis_from_map_findings over ALL map findings and then written over the
# reducer's number, so the displayed narrative covered a fraction of the
# document while the score covered all of it. The repo's own 30-page PDF (162
# sections) is already well past the cliff.
#
# THE REPLACEMENT. Group the section reports into batches that each fit under
# the cap, reduce each batch into a partial master report, then reduce the
# partials together -- recursing when the partials themselves do not fit.
# Nothing is cut, and a document that fits in one batch takes exactly the one
# call it always took, with a byte-identical payload.
#
# The cap is on the WHOLE rendered prompt, not just the payload; the template's
# own length is measured at call time and subtracted, so editing a reduce
# prompt cannot silently eat the headroom.
REDUCE_PROMPT_MAX_CHARS = _env_int("LYTREX_REDUCE_PROMPT_MAX_CHARS", 50000)
# Floor for the payload budget, so a pathologically long template cannot drive
# the budget to zero and batch every report on its own.
REDUCE_MIN_PAYLOAD_CHARS = _env_int("LYTREX_REDUCE_MIN_PAYLOAD_CHARS", 8000)
# Used only if a prompt template refuses to render for measurement.
REDUCE_TEMPLATE_OVERHEAD_FALLBACK = 2000
# Levels of partial-merging allowed before the remainder is merged in one
# capped call. Each level at least halves its input (see batch_reports_by_size
# min_per_batch), so 8 covers 2**8 = 256 first-level batches -- on the order of
# 10,000 sections, or a 1,800-page document. It exists to make an unbounded
# spend impossible, not to be reached: a 140-page document needs 4.
REDUCE_MAX_DEPTH = _env_int("LYTREX_REDUCE_MAX_DEPTH", 8)
REDUCE_TRUNCATION_MARK = "\n... [TRUNCATED: reduce batch exceeds the payload budget] ..."


class ReduceBatch(NamedTuple):
    """
    One reduce call's worth of input.

    reports   - the reports in this batch, in order.
    payload   - exactly the string handed to the prompt's {reports} slot. Never
                longer than the budget, whatever else is true.
    truncated - the batch could not be made to fit and its payload was cut.
                Only reachable when a batch is forced to hold content larger
                than the budget: a single report bigger than the whole budget,
                or a min_per_batch group that has to stay together to make the
                recursion converge. Cut, but never dropped, and never silent --
                the flag is what puts it on the report.
    """
    reports: List[Any]
    payload: str
    truncated: bool


def _close_batch(reports: List[Any], payload: str, limit: int) -> ReduceBatch:
    """Seals one batch, cutting the payload only if it genuinely cannot fit."""
    if len(payload) <= limit:
        return ReduceBatch(reports, payload, False)
    body_limit = max(limit - len(REDUCE_TRUNCATION_MARK), 1)
    return ReduceBatch(reports, payload[:body_limit] + REDUCE_TRUNCATION_MARK, True)


def batch_reports_by_size(reports: List[Any], max_chars: int,
                          min_per_batch: int = 1) -> List[ReduceBatch]:
    """
    Groups reports so that every batch's serialised payload fits in max_chars.

    Sized by SERIALISED LENGTH, not by a fixed count, because report size
    varies ~2x with finding density: a fixed count that is safe for a
    finding-heavy document wastes most of the budget on a sparse one, and a
    count tuned for a sparse one overruns on a dense one.

    min_per_batch is the convergence lever. At 1 (the section-report level) a
    batch is closed as soon as the next report would overflow it. At 2 (every
    partial-merging level above it) a batch must take a second report even if
    that overflows, because a level that emits as many batches as it consumed
    makes no progress and the recursion would never terminate. Overflowing
    groups are cut and flagged; merging two partials with the tail of the
    second cut still beats one capped call over all of them, which is what the
    code did before batching existed.

    Guarantees, each of which has a test:
      - every input report appears in exactly one batch, in input order;
      - len(batch.payload) <= max_chars for every batch, always;
      - no payload is cut mid-JSON unless the batch is flagged truncated;
      - with min_per_batch >= 2 and len(reports) >= 2 the batch count is at
        most ceil(len(reports) / 2), so every level strictly shrinks;
      - a list that fits whole returns ONE batch whose payload is exactly
        json.dumps(reports, indent=2) -- byte-identical to the pre-batching
        call, so small documents do not change behaviour at all.
    """
    batches: List[ReduceBatch] = []
    if not reports:
        return batches
    limit = max(int(max_chars), 1)
    floor = max(int(min_per_batch), 1)

    current: List[Any] = []
    current_payload = ""
    for report in reports:
        candidate = current + [report]
        payload = json.dumps(candidate, indent=2)
        if len(payload) <= limit or len(current) < floor:
            current, current_payload = candidate, payload
            continue
        batches.append(_close_batch(current, current_payload, limit))
        current, current_payload = [report], json.dumps([report], indent=2)

    if current:
        batches.append(_close_batch(current, current_payload, limit))
    return batches


# Fields a partial master report keeps when it is compacted for the next merge
# up the tree. Everything else -- all_compliant_areas, master_recommendations,
# final_compliance_score -- is either regenerated by the merge above or, in the
# score's case, overwritten from the map phase regardless.
REDUCE_PARTIAL_SUMMARY_KEYS = ("master_executive_summary", "master_summary", "summary")


def payload_finding_survival(batches: List[ReduceBatch], master_key: str) -> tuple:
    """
    (kept, total) findings that actually fit inside the payloads about to be
    sent.

    This is the criterion compaction is judged on, and it is deliberately not
    "how many batches got cut". A cut takes the TAIL of the JSON, and the
    findings list is one of the last fields a reducer emits, so a cut batch and
    an uncut one can carry wildly different amounts of the thing the merge
    exists to carry. Counting findings measures that directly.
    """
    kept = total = 0
    for batch in batches:
        for report in batch.reports:
            if not isinstance(report, dict):
                continue
            for finding in (report.get(master_key) or []):
                if not isinstance(finding, str) or not finding.strip():
                    continue
                total += 1
                if finding in batch.payload:
                    kept += 1
    return kept, total


def compact_partial_report(report: Any, master_key: str) -> Any:
    """
    Projects a partial master report onto what the NEXT merge actually needs:
    its findings, and one line of summary.

    Only used when partials will not otherwise fit, and only above the section
    level. The alternative there is cutting a partial's payload mid-JSON, and
    because the findings list is one of the LAST fields the reducer emits, that
    cut lands squarely on the findings -- the one thing the merge exists to
    carry. Dropping a partial's recommendations to keep its findings whole is
    the right way round.
    """
    if not isinstance(report, dict):
        return report
    slim: Dict[str, Any] = {}
    for key in REDUCE_PARTIAL_SUMMARY_KEYS:
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            slim[key] = value
            break
    slim[master_key] = report.get(master_key) or []
    return slim


def _finding_kind(text: str) -> str:
    return "absence" if _ABSENCE_RE.search(text) else "contradiction"


def _finding_severity(text: str) -> str:
    """
    The tier the model assigned this finding, or DEFAULT_SEVERITY_TIER when it
    assigned none.

    UNVALIDATED AGAINST LIVE OUTPUT. No archived report contains the Severity
    field this reads, because no prompt asked for it until now, so every
    archived finding takes the default and this function has never yet returned
    anything else on real data. Whether the model populates the field, and
    whether its labels discriminate rather than collapsing to all-CRITICAL as
    the 2911fa67 run's volunteered markers did, can only be established by a
    live audit.
    """
    if not isinstance(text, str):
        return DEFAULT_SEVERITY_TIER
    m = _SEVERITY_RE.search(text)
    if not m:
        return DEFAULT_SEVERITY_TIER
    return _SEVERITY_SYNONYMS.get(m.group(1).lower(), DEFAULT_SEVERITY_TIER)


def finding_impact(text: str, kind: Optional[str] = None) -> float:
    """
    What one finding contributes to the failure mass: its severity tier scaled
    by whether it is a stated breach or an undocumented control.
    """
    tier = _finding_severity(text)
    k = kind or _finding_kind(text if isinstance(text, str) else "")
    kind_w = KIND_WEIGHT_ABSENCE if k == "absence" else KIND_WEIGHT_CONTRADICTION
    return SEVERITY_TIER_WEIGHTS.get(tier, SEVERITY_TIER_WEIGHTS["moderate"]) * kind_w


def severity_mass(contradictions: List[str], absences: List[str]) -> Dict[str, Any]:
    """
    Total weighted failure mass, plus the per-tier census that explains it.

    The census is returned rather than only the total because the total alone
    is unreadable: a mass of 8.79 could be three critical breaches or twenty-two
    minor gaps, and a reader checking the score needs to see which.
    """
    tally: Dict[str, Dict[str, int]] = {
        t: {"contradiction": 0, "absence": 0} for t in SEVERITY_TIER_WEIGHTS
    }
    total = 0.0
    # The critical share is split out here rather than recomputed downstream
    # because the transfer function charges it a different exponent; see
    # COVERAGE_CRITICAL_EXPONENT. Kept as mass, not a count, so a critical
    # ABSENCE still carries its kind discount into the amplified term.
    critical_total = 0.0
    for kind, items in (("contradiction", contradictions), ("absence", absences)):
        for text in items or []:
            tier = _finding_severity(text)
            if tier not in tally:
                tier = DEFAULT_SEVERITY_TIER
            tally[tier][kind] += 1
            impact = finding_impact(text, kind)
            total += impact
            if tier == "critical":
                critical_total += impact
    labelled = sum(
        1 for t in list(contradictions or []) + list(absences or [])
        if isinstance(t, str) and _SEVERITY_RE.search(t)
    )
    return {
        "failure_mass": round(total, 4),
        "critical_failure_mass": round(critical_total, 4),
        "severity_census": tally,
        "findings_with_explicit_severity": labelled,
        "findings_defaulted_to_%s" % DEFAULT_SEVERITY_TIER:
            len(contradictions or []) + len(absences or []) - labelled,
    }


def coverage_strain(
    failure_mass: float,
    controls_assessed: int,
    critical_mass: Optional[float] = None,
) -> float:
    """
    The dimensionless badness the score decays over: critical failure density
    charged super-linearly, everything else charged linearly.

    critical_mass: the part of failure_mass carried by CRITICAL findings. When
        omitted the whole mass is charged at the linear rate, which is the
        conservative reading -- a caller that cannot say which findings were
        critical must not have the amplified term applied on a guess. That
        degenerate case is exactly the old strain = failure_mass / assessed.

    MONOTONE BY CONSTRUCTION. Both terms are non-decreasing in their own mass
    and no finding count appears anywhere, so adding any finding can only
    raise the strain. This is the property that ruled out the
    average-severity formulation; see COVERAGE_CRITICAL_EXPONENT.
    """
    denom = max(float(controls_assessed), COVERAGE_MIN_ASSESSED)
    total = max(0.0, float(failure_mass))
    crit = 0.0 if critical_mass is None else max(0.0, float(critical_mass))
    # Never let rounding of the two independently-rounded masses push the
    # non-critical remainder negative.
    crit = min(crit, total)
    other = total - crit
    return (crit / denom) ** COVERAGE_CRITICAL_EXPONENT + (other / denom)


def coverage_score(
    failure_mass: float,
    controls_assessed: int,
    critical_mass: Optional[float] = None,
) -> float:
    """
    The primary model: weighted failure mass measured against the controls this
    audit actually examined, with critical density charged super-linearly.

    WHY THE SCORE NEVER REACHES 0. exp() is asymptotic, so a document whose
    failure mass exceeds its assessed scope lands near zero without arriving,
    and COVERAGE_SCORE_FLOOR stops rounding from finishing the job. That is
    deliberate. A literal 0 is the claim "no part of this policy complies with
    any examined control", and the pipeline cannot support it for the same
    reason it cannot support a 100: it observes failures, never successes. The
    residual fraction of a point is not generosity, it is the honest statement
    that nothing here was verified either way. Staying off the floor is also
    what keeps 25 findings distinguishable from 40, which the old flat
    100 - 5n model lost at twenty.
    """
    strain = coverage_strain(failure_mass, controls_assessed, critical_mass)
    return max(COVERAGE_SCORE_FLOOR, round(100.0 * math.exp(-COVERAGE_DECAY * strain), 2))


def _finding_control_key(text: str) -> Optional[str]:
    """
    The identifier an absence finding is about, so the same missing control
    collapses to one finding however many chunks reported it.

    Only the control IDs are kept, not the descriptive tail: the model words
    the same control differently per section ("SEC_0026 (6.2) KLM Processes"
    vs "SEC_0026 (6.2) Key Distribution"), and keying on the prose would let
    those survive as separate findings. Where several controls are cited the
    sorted set becomes the key, so the same group matches in any order.
    """
    m = _CONTROL_RE.search(text)
    if not m:
        return None
    cited = m.group(1)
    ids = _CONTROL_ID_RE.findall(cited.upper())
    if ids:
        # An alphanumeric id (SEC_0007, AC-03) names the control; a bare dotted
        # number (3.4) is the sub-clause it happens to sit under. Keep the
        # former when present so "SEC_0007" and "SEC_0007 (3.4)" share a key.
        alpha = {i.replace("-", "_") for i in ids if re.search(r"[A-Z]", i)}
        canonical = sorted(alpha) if alpha else sorted(set(ids))
        return "|".join(canonical)
    key = re.sub(r"[^a-z0-9]+", "", cited.lower())
    return key or None


def classify_and_dedupe(findings: List[Any]) -> Dict[str, Any]:
    """
    Splits findings into contradictions and absences, collapsing absences that
    name the same framework control so one undocumented control counts once
    however many chunks failed to mention it. Contradictions are only collapsed
    on exact text, since two genuine breaches can cite the same control.
    """
    contradictions: List[str] = []
    absences: List[str] = []
    seen_contradiction: set = set()
    seen_absence_control: set = set()
    absence_dupes = 0
    contradiction_dupes = 0

    for raw in findings or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip()
        if _finding_kind(text) == "absence":
            key = _finding_control_key(text) or re.sub(r"[^a-z0-9]+", "", text.lower())[:90]
            if key in seen_absence_control:
                absence_dupes += 1
                continue
            seen_absence_control.add(key)
            absences.append(text)
        else:
            key = re.sub(r"\s+", " ", text.lower())
            if key in seen_contradiction:
                contradiction_dupes += 1
                continue
            seen_contradiction.add(key)
            contradictions.append(text)

    return {
        "contradictions": contradictions,
        "absences": absences,
        "absence_duplicates_removed": absence_dupes,
        "contradiction_duplicates_removed": contradiction_dupes,
    }


def penalty_estimate_score(n_contradictions: int, n_absences: int) -> float:
    """
    The FALLBACK model, unchanged: pure penalty accumulation with no
    denominator. Kept verbatim so an archived report recomputes to the number
    it was filed with.

    Its defect is the reason the coverage model exists: with nothing to divide
    by, it can only say how much was found wrong, never out of how much. Two
    policies with thirteen breaches score the same whether the audit examined
    fifteen controls or fifty. Do not present its output as coverage.
    """
    penalty = (
        CONTRADICTION_WEIGHT * (max(0, n_contradictions) ** CONTRADICTION_EXPONENT)
        + ABSENCE_WEIGHT * (max(0, n_absences) ** ABSENCE_EXPONENT)
    )
    return round(max(0.0, 100.0 - penalty), 2)


def compliance_score(
    n_contradictions: int,
    n_absences: int,
    *,
    contradictions: Optional[List[str]] = None,
    absences: Optional[List[str]] = None,
    controls_assessed: Optional[int] = None,
) -> float:
    """
    Rates a policy by COVERAGE AND SEVERITY when the assessed scope is known,
    and falls back to the old penalty estimate when it is not.

    The two positional arguments keep their original meaning and, called with
    those alone, this returns exactly what it always returned. The coverage
    model is opt-in on controls_assessed being supplied, which is deliberate:
    a caller that cannot say how many controls were examined must not be handed
    a coverage figure, because there is nothing to divide by and inventing a
    denominator is the failure this whole change is about.

    controls_assessed: how many distinct framework controls were retrieved into
        a section prompt. NOT the framework's total control count -- see
        score_basis_from_map_findings for why that denominator is rejected.
    contradictions / absences: the finding texts, so per-finding severity is
        read. When omitted, the counts are scored at DEFAULT_SEVERITY_TIER.
    """
    if controls_assessed is None or controls_assessed <= 0:
        return penalty_estimate_score(n_contradictions, n_absences)

    if contradictions is None and absences is None:
        # No texts to read severity from, so the counts are scored at the
        # default tier. Same denominator, same transfer function -- only the
        # per-finding severity detail is missing. The critical mass is 0
        # unless the default tier IS critical, which is the honest reading:
        # an unlabelled finding is evidence of nothing about its severity and
        # must not be charged the amplified critical rate on a guess.
        unit = SEVERITY_TIER_WEIGHTS.get(
            DEFAULT_SEVERITY_TIER, SEVERITY_TIER_WEIGHTS["moderate"]
        )
        mass = (
            max(0, n_contradictions) * unit * KIND_WEIGHT_CONTRADICTION
            + max(0, n_absences) * unit * KIND_WEIGHT_ABSENCE
        )
        crit = mass if DEFAULT_SEVERITY_TIER == "critical" else 0.0
    else:
        detail = severity_mass(contradictions or [], absences or [])
        mass = detail["failure_mass"]
        crit = detail["critical_failure_mass"]

    return coverage_score(mass, controls_assessed, crit)


# --- SCORING BASIS: MAP PHASE, NOT THE REDUCER'S PROSE ---------------------
# The score used to be computed from the reducer's merged output. The reducer
# is a free-form LLM rewrite step: it prepends "CRITICAL:", rephrases findings,
# and compresses citations from
#     [Company Page: 2 | Company Section: ... | Framework Control: SEC_0000 (1-1-1)]
# down to
#     [SEC_0000 1-1-1]
# Every one of those is a wording choice, and every one of them feeds
# _ABSENCE_RE and _finding_control_key. The score was therefore a function of
# the reducer's prose style, which is not a property of the audited document.
#
# MEASURED, on the two archived ECC runs of the same policy and framework:
#   _finding_control_key() resolves on  0 of 30 reducer-reworded findings
#                          and on      32 of 32 map-phase-style findings
# So the reworded form does not merely shift classification -- it destroys the
# control identity that absence dedupe and corroboration ranking both key on.
#
# The map phase is the stable input: each section report is produced by the same
# prompt against the same citation template, so its phrasing is consistent by
# construction. Scoring from it means the reducer may reword however it likes
# without moving the number, while its merged output still drives the report's
# displayed findings and summary, which is what it is good at.
# --- THE DENOMINATOR --------------------------------------------------------
# A coverage score is met/assessed, and this pipeline never establishes that a
# control is MET. It reports failures and absences only. Silence about a
# control is not evidence of compliance. Three denominators were available and
# the choice between them is the whole design, so it is recorded here.
#
# (a) REJECTED -- total framework controls (27 for ECC). The intuitive one:
#     "13 failures out of 27 controls is 52%". It asserts the other 14 controls
#     were verified compliant when most were never placed in front of the model
#     at all. In a compliance tool that is a false assurance claim, and it is
#     the most dangerous kind, because it inflates the score precisely where the
#     audit did the least work -- adding controls the pipeline never reaches
#     raises the score. It is also not a stable number. MEASURED in the shipped
#     JSON: ECC.json carries 27 top-level `sections`, but their `text` fields
#     hold 111 inline sub-controls ("1-1-1:", "1-1-2:", ...), while NCA.json's
#     26 and SAMA.json's 36 sections carry none. So "total controls" silently
#     means 111 for ECC and 26 for NCA, and cross-framework scores would not be
#     comparable. Worse, the model cites sub-controls ("SEC_0000 (1-1-1)") while
#     the retrieval unit is the section id, so numerator and denominator would
#     be counting different things.
#
# (b) CHOSEN -- controls actually assessed: the distinct framework control ids
#     retrieved into at least one section prompt. The claim it supports is the
#     one a human auditor's scope statement makes: "of the controls this audit
#     examined, this share failed." It is computable, and cheaply: the section
#     loop in audit_large_document already reads fw_internal_section_id off
#     every parent chunk it forwards, so the union is collected there and is in
#     hand before the reducer is called.
#     ITS OWN BIAS, STATED: "assessed" means "retrieved into the prompt", not
#     "reasoned about". A control the model ignored still counts, which
#     over-counts the denominator and therefore inflates the score -- the same
#     direction of error as (a), though bounded by retrieval rather than by the
#     framework's size. This is why the score is labelled examined-scope
#     coverage and controls_assessed is published beside it: the reader can see
#     the scope the number is relative to. It is never a verified-compliance
#     claim.
#
# (c) REJECTED as primary -- all_compliant_areas + all_unique_violations. In
#     principle the best denominator, since both sides are observed. Three
#     defects. It exists only in detailed mode; MEASURED, all five archived ECC
#     reports ran concise mode and `all_compliant_areas` is absent from 5 of 5,
#     so in practice it is a denominator the corpus does not have. It is
#     REDUCER output, and routing the score back through reducer prose is
#     exactly the coupling score_basis_from_map_findings was built to sever.
#     And a "compliant area" is the model's unverified assertion, which would
#     import its optimism into numerator and denominator at once.
def score_basis_from_map_findings(
    map_findings: List[Any],
    assessed_control_ids: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Classifies the map-phase findings and derives the compliance score from
    them. Returns the score plus the full map-phase evidence, so an archived
    report carries everything needed to recompute its own score offline.

    assessed_control_ids: the distinct framework control ids this audit put in
        front of the model. Supply it and the score is COVERAGE-based; omit it
        and the score is the penalty ESTIMATE, because without a denominator
        there is no coverage to report and one is not invented.

    RETURN SHAPE. Called with map_findings alone this returns exactly the keys
    it always returned, so archived reports recompute to the number they were
    filed with. Passing assessed_control_ids -- even an empty collection --
    adds the score_model block that names which model produced the figure. The
    live pipeline always passes it, so a live report always states its basis.
    """
    buckets = classify_and_dedupe(map_findings)
    contradictions = buckets["contradictions"]
    absences = buckets["absences"]

    basis: Dict[str, Any] = {
        "scoring_basis": "map_phase",
        "final_compliance_score": compliance_score(len(contradictions), len(absences)),
        "map_phase_finding_count": len([
            f for f in (map_findings or []) if isinstance(f, str) and f.strip()
        ]),
        "map_phase_contradiction_count": len(contradictions),
        "map_phase_absence_count": len(absences),
        "map_phase_contradiction_findings": contradictions,
        "map_phase_absence_findings": absences,
        "map_phase_absence_duplicates_removed": buckets["absence_duplicates_removed"],
        "map_phase_contradiction_duplicates_removed": buckets["contradiction_duplicates_removed"],
    }
    if assessed_control_ids is None:
        # Legacy call: counts only, no scope. Shape and number unchanged.
        return basis

    ids = sorted({str(c).strip() for c in assessed_control_ids if str(c).strip()})
    n_assessed = len(ids)
    mass_detail = severity_mass(contradictions, absences)
    basis["assessed_control_ids"] = ids
    basis["controls_assessed"] = n_assessed
    basis.update(mass_detail)

    if n_assessed > 0:
        basis["final_compliance_score"] = compliance_score(
            len(contradictions), len(absences),
            contradictions=contradictions, absences=absences,
            controls_assessed=n_assessed,
        )
        basis["score_model"] = "coverage"
        basis["coverage_denominator"] = "controls_assessed"
        basis["coverage_ratio"] = round(
            mass_detail["failure_mass"] / max(float(n_assessed), COVERAGE_MIN_ASSESSED), 4
        )
        # coverage_ratio remains the plain mass-over-scope reading, which is
        # what the prose note quotes. coverage_strain is what the score
        # actually decays over -- they differ whenever critical findings are
        # present, so both are published rather than leaving a reader to
        # wonder why exp(-DECAY * ratio) does not reproduce the score.
        basis["coverage_strain"] = round(
            coverage_strain(
                mass_detail["failure_mass"],
                n_assessed,
                mass_detail["critical_failure_mass"],
            ), 4
        )
        basis["coverage_critical_exponent"] = COVERAGE_CRITICAL_EXPONENT
        basis["coverage_decay"] = COVERAGE_DECAY
        basis["score_is_verified_coverage"] = False
    else:
        # Retrieval recorded no control ids, so there is nothing to divide by.
        # The penalty estimate already sits in final_compliance_score above.
        basis["score_model"] = "penalty_estimate"
        basis["coverage_denominator"] = None
        basis["coverage_ratio"] = None
        basis["score_model_fallback_reason"] = (
            "No framework control ids were recorded for this run, so the number "
            "of controls assessed is unknown and no coverage fraction can be "
            "computed. The score is a penalty estimate, not a coverage figure."
        )
        basis["score_is_verified_coverage"] = False
    return basis


def score_basis_note(basis: Dict[str, Any], n_displayed: int) -> str:
    """
    One sentence reconciling the scored counts with the displayed list, so a
    reader who notices the two disagree finds the reason in the report itself
    rather than concluding the report is broken.
    """
    note = (
        f"Score is computed from the {basis['map_phase_finding_count']} finding(s) the "
        f"section-level (map) phase produced -- {basis['map_phase_contradiction_count']} "
        f"contradiction(s) and {basis['map_phase_absence_count']} coverage gap(s) after "
        f"deduplication. The {n_displayed} finding(s) listed in this report are the "
        f"merge step's consolidated view of those same findings; its wording and grouping "
        f"do not affect the score."
    )
    # Which MODEL turned those findings into a number. A reader must never have
    # to guess whether they are looking at a measured coverage fraction or a
    # penalty estimate wearing the same units.
    model = basis.get("score_model")
    if model == "coverage":
        note += (
            f" MODEL: coverage. The weighted severity of those findings "
            f"({basis['failure_mass']}) is measured against the "
            f"{basis['controls_assessed']} distinct framework control(s) this audit "
            f"actually retrieved into a section prompt, giving a failure ratio of "
            f"{basis['coverage_ratio']}. Of that mass, "
            f"{basis['critical_failure_mass']} is carried by findings marked "
            f"critical, and that share is charged super-linearly (exponent "
            f"{basis['coverage_critical_exponent']}), so the figure the score "
            f"actually decays over is {basis['coverage_strain']} rather than the "
            f"ratio itself. This is examined-scope coverage, NOT verified "
            f"compliance: a control that produced no finding was examined without an "
            f"exception being raised, which is not the same as having been tested and "
            f"passed. Controls the audit never retrieved are outside this denominator "
            f"and are neither credited nor penalised."
        )
    elif model == "penalty_estimate":
        note += (
            " MODEL: penalty estimate, NOT coverage. "
            + basis.get("score_model_fallback_reason", "")
        )
    else:
        note += (
            " MODEL: penalty estimate, NOT coverage -- this basis carries no record "
            "of how many controls were assessed, so the figure states how much was "
            "found wrong, never out of how much."
        )
    return note


# --- CONTRADICTION RANKING -------------------------------------------------
# Findings arrive as plain strings. No prompt in this file asks the model for a
# severity field, so there is no model-assigned severity to sort on and none is
# invented here. Only two signals actually exist in the data at reduce time:
#
#   1. STATED PENALTY - the detailed prompt's template asks for a trailing
#      "(-15 pts)", so in detailed mode the model often emits its own point
#      deduction for that breach. It is absent in concise mode and absent
#      whenever the model ignores the template, in which case it reads 0.
#   2. CORROBORATION - how many separate section reports independently raised a
#      contradiction against the same framework control. This is confidence,
#      not severity: a breach three sections agree on is less likely to be a
#      one-off hallucination than one raised once.
#
# Neither is a severity taxonomy, so this orders the list rather than scoring
# it, and both signals are returned alongside so a reader can see why a finding
# sits where it does. Ordering is stable: equal signals keep their input order.
_PENALTY_RE = re.compile(r"\(\s*-\s*(\d{1,3})\s*(?:pts?|points?)\s*\)", re.IGNORECASE)


def _stated_penalty(text: str) -> int:
    m = _PENALTY_RE.search(text)
    return int(m.group(1)) if m else 0


def rank_contradictions(
    contradictions: List[str],
    section_findings: Optional[List[List[Any]]] = None,
) -> tuple:
    """
    Orders contradictions by the model's own stated point deduction, then by
    how many section reports corroborated the same control, then by input
    order. Returns (ordered_findings, ranking_signals) where ranking_signals is
    a list parallel to ordered_findings.
    """
    corroboration: Dict[str, int] = {}
    for findings in section_findings or []:
        seen_here: set = set()
        for raw in findings or []:
            if not isinstance(raw, str) or not raw.strip():
                continue
            if _finding_kind(raw) != "contradiction":
                continue
            key = _finding_control_key(raw)
            if key and key not in seen_here:
                seen_here.add(key)
                corroboration[key] = corroboration.get(key, 0) + 1

    decorated = []
    for idx, text in enumerate(contradictions):
        key = _finding_control_key(text)
        support = corroboration.get(key, 0) if key else 0
        # Negated so a plain ascending sort puts the strongest signals first.
        decorated.append((-_stated_penalty(text), -support, idx, text))
    decorated.sort()

    ordered = [t for _, _, _, t in decorated]
    signals = [
        {
            "finding": t,
            "stated_penalty": -penalty,
            "corroborating_sections": -support,
        }
        for penalty, support, _, t in decorated
    ]
    return ordered, signals


class ComplianceRAG:
    """
    Elite Lytrex Compliance RAG with Structural Hybrid Retrieval.
    Architecture: JSON-Structured Parents -> Child Chunks -> FAISS+BM25 -> Reranker -> LLM (provider set by PROVIDER: groq / cerebras / gemini).
    """

    def __init__(self,
                 # Directory containing the source framework files (PDFs/JSONs)
                 pdf_source_dir: str = "frameworks",
                 # Base path where the FAISS vector database will be saved/loaded
                 vector_db_base_path: str = "LytrexDB_Groq", 
                 # BGE Large is highly accurate and free (runs locally via HuggingFace)
                 embedding_model: str = "BAAI/bge-large-en-v1.5", 
                 # Left as None so the model follows PROVIDER: each provider in
                 # PROVIDER_CONFIG carries its own default and env override,
                 # because the same model is published under different ids
                 # (Groq "openai/gpt-oss-120b" vs Cerebras "gpt-oss-120b").
                 # Pass a string here to pin one regardless of provider.
                 model_name: Optional[str] = None,
                 # The Cross-Encoder model used to rerank retrieved candidates
                 reranker_model: str = "BAAI/bge-reranker-base", 
                 # Character size for the larger parent context chunks
                 parent_chunk_size: int = 2500,   
                 # Character overlap for the larger parent context chunks
                 parent_chunk_overlap: int = 250,
                 # Character size for the smaller child chunks stored in the vector DB
                 child_chunk_size: int = 500,     
                 # Character overlap for the smaller child chunks
                 child_chunk_overlap: int = 100,
                 # Character size for initially splitting the user's uploaded target document into sections.
                 # Default and env override live on MAP_CHUNK_SIZE; an explicit argument still wins.
                 map_chunk_size: int = MAP_CHUNK_SIZE,
                 # Character overlap for the target document sections
                 map_chunk_overlap: int = MAP_CHUNK_OVERLAP,
                 # Character size for breaking down target document sections into micro-queries for retrieval
                 section_chunk_size: int = 200,    
                 # Character overlap for the target document micro-queries
                 section_chunk_overlap: int = 20,  
                 # Optional explicit API key; if not provided, it will look for GROQ_API_KEY in the .env file
                 groq_api_key: Optional[str] = ''):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_base_path = os.path.join(base_dir, vector_db_base_path)

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.map_chunk_size = map_chunk_size
        self.map_chunk_overlap = map_chunk_overlap
        self.section_chunk_size = section_chunk_size
        self.section_chunk_overlap = section_chunk_overlap

        # Provider is resolved before the local models load, so a missing key
        # fails in a second rather than after a multi-GB model download.
        self.provider = _resolve_provider()
        cfg = PROVIDER_CONFIG[self.provider]

        api_key = groq_api_key or os.getenv(cfg["key_env"])
        if not api_key:
            raise ValueError(
                f"{cfg['key_env']} missing. PROVIDER is set to '{self.provider}', "
                f"so that key must be present in .env. Get one at {cfg['console']}."
            )

        # Explicit argument wins, then the provider's env override, then default.
        self.model_name = model_name or os.getenv(cfg["model_env"]) or cfg["default_model"]

        # Request pacing. LLM_RPM overrides the provider default; 0 disables
        # throttling entirely. Tune this when changing model or tier.
        rpm_raw = (os.getenv("LLM_RPM") or "").strip()
        try:
            rpm = int(rpm_raw) if rpm_raw else int(cfg.get("rpm", 0))
        except ValueError:
            print(f"[INIT] LLM_RPM={rpm_raw!r} is not an integer; using provider default.")
            rpm = int(cfg.get("rpm", 0))
        self.request_rpm = max(0, rpm)
        self._min_call_interval = 60.0 / self.request_rpm if self.request_rpm else 0.0
        self._last_call_at = 0.0

        # Per-audit token accounting. One ledger per instance, reset at the
        # start of each run; see TokenLedger for why the phases are split.
        self._token_ledger = TokenLedger()

        print(f"[INIT] Loading Local HuggingFace Embeddings ({embedding_model})...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        print(f"[INIT] Loading BGE Cross-Encoder Reranker ({reranker_model})...")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        pace = (f"{self.request_rpm} req/min ({self._min_call_interval:.1f}s apart)"
                if self.request_rpm else "unthrottled")
        print(f"[INIT] LLM provider: {self.provider} | model: {self.model_name} | pacing: {pace}")
        if self.provider == "groq":
            # Unchanged from before: the native client, still the default path.
            self.llm = ChatGroq(
                temperature=0,
                groq_api_key=api_key,
                model_name=self.model_name,
            )
        elif self.provider == "gemini":
            # Gemini has its own protocol, so it needs its own client.
            # response_mime_type pins the reply to raw JSON: without it Gemini
            # tends to wrap the object in ```json fences and sometimes adds a
            # sentence of preamble, both of which the parser would have to
            # strip back off.
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0,
                response_mime_type="application/json",
                timeout=120,
                max_retries=2,
            )
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # max_tokens is required by the Anthropic API and langchain-anthropic
            # defaults it to a low value; the audit returns a JSON report of
            # roughly a thousand tokens, so this is generous headroom rather
            # than a limit anything should reach.
            self.llm = ChatAnthropic(
                model=self.model_name,
                api_key=api_key,
                temperature=0,
                max_tokens=8192,
                timeout=120,
                max_retries=2,
            )
        else:
            # Cerebras speaks the OpenAI wire format, so the OpenAI client
            # works against their base URL with no extra dependency.
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                temperature=0,
                model=self.model_name,
                api_key=api_key,
                base_url=cfg["base_url"],
                timeout=120,
                max_retries=2,
            )
        self.output_parser = JsonOutputParser()
        self._setup_prompts()

    def _setup_prompts(self):
        self.relevance_prompt = ChatPromptTemplate.from_template(
            """
            You are a strict Document Classification AI.
            Analyze this excerpt from an uploaded document. Is this document a corporate policy, cybersecurity framework, technical procedure, or business manual?
            If it is a menu, novel, random article, or irrelevant, reject it.
            
            Excerpt: {text}
            
            Respond ONLY with a JSON object:
            {{
                "is_relevant": true/false,
                "reasoning": "1 sentence explaining why."
            }}
            """
        )

        self.detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite, strict Compliance Auditor AI developed by the Lytrex Team.
            Evaluate the <company_document> against the <framework_context>.

            CRITICAL INSTRUCTIONS:
            1. DO NOT lazily copy the JSON template. Actively evaluate.
            2. If a control is not mentioned, do NOT say it's a violation. Silence is ignored.
            3. Explicit Contradictions Only: Flag violations where the document explicitly contradicts the framework.
            4. Use 'internal_audit_reasoning' to map your logic BEFORE scoring.
            5. Base score is 100. Deduct 10-25 points for each explicit violation found.
            6. Traceability: For EVERY single comparison, sentence, compliant area, and violation, you MUST explicitly cite it in this exact format: [Company Page: X | Company Section: Y | Framework Control: Z | Severity: S]. Extract the page from the metadata provided or the printed text.
            7. Severity: S is EXACTLY one of CRITICAL, MODERATE, MINOR. Assign it per finding using these definitions, and do NOT default everything to CRITICAL - a report where every finding is critical carries no information and will be treated as unrated:
               - CRITICAL: the breach directly exposes systems or data now. Unenforced or absent MFA on remote/privileged access, shared or default administrator credentials, unencrypted backups or data at rest, no security monitoring or logging at all, plaintext credential storage.
               - MODERATE: a real control weakness that raises risk but is not directly exploitable on its own. Untested disaster recovery, review cycles that are defined but not performed, retention periods shorter than the framework requires, missing approval steps.
               - MINOR: documentation and completeness gaps. A control area the excerpt simply does not mention, a missing definition, an unstated owner or interval, wording that is vague rather than wrong.
               A finding that says a control is "not documented in this excerpt" is MINOR unless the control itself is one of the CRITICAL items above, because an excerpt not mentioning something is usually an artifact of how the document was split.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a comprehensive analysis.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "internal_audit_reasoning": "Step-by-step logic. I checked [Company Section: X]. Z was missing (ignored). Found violation in [Company Section: W].",
                "compliance_score": 1-100,
                "executive_summary": "A detailed 3-4 sentence summary.",
                "compliant_areas": ["[Company Page: X | Company Section: Y (Never mention Section Title) | Framework section: Z (never mention framework ID)] precise detail of what they did right"],
                "violations": ["[Company Page: X | Company Section: Y (Never mention Section Title) | Framework section: Z (never mention framework ID) | Severity: CRITICAL] specific breach (-15 pts)"],
                "recommendations": ["[Company Page: X | Company Section: Y (Never mention Section Title) | Framework section: Z (never mention framework ID)] Detailed actionable steps to fix the specific violation"]
            }}
            """
        )

        self.concise_prompt = ChatPromptTemplate.from_template(
            """
            You are a fast Compliance Auditor AI developed by the Lytrex Team.
            Evaluate the <company_document> against the <framework_context>.

            CRITICAL INSTRUCTIONS:
            1. If a control is not mentioned, ignore it. Do NOT invent violations.
            2. Explicit Contradictions Only.
            3. Use 'internal_audit_reasoning' to do math. Base score 100, deduct for explicit violations.
            4. Traceability: You MUST explicitly format the citation exactly like this: [Company Page: X | Company Section: Y (Never mention Section Title) | Framework Control: Z (never mention framework ID) | Severity: S] for EVERY key issue and comparison sentence. Get the page from the metadata provided or printed text.
            5. Severity: S is EXACTLY one of CRITICAL, MODERATE, MINOR. Do NOT mark everything CRITICAL - a report where every finding is critical carries no information and will be treated as unrated.
               - CRITICAL: directly exposes systems or data now (unenforced/absent MFA, shared or default admin credentials, unencrypted backups or data at rest, no monitoring or logging at all, plaintext credentials).
               - MODERATE: a real control weakness that is not directly exploitable alone (untested disaster recovery, reviews defined but not performed, retention shorter than required, missing approvals).
               - MINOR: documentation gaps - a control area this excerpt does not mention, a missing definition, an unstated owner or interval.
               An issue that says something is "not documented in this excerpt" is MINOR unless it is one of the CRITICAL items above, because an excerpt not mentioning a control is usually an artifact of how the document was split.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a strictly brief, top-level overview. Do not over-explain.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "internal_audit_reasoning": "Brief check of contradictions for scoring based on [Company Section: X].",
                "compliance_score": 1-100,
                "summary": "A strict 1-sentence summary.",
                "key_issues": ["[Company Page: X | Company Section: Y | Framework section: Z | Severity: CRITICAL] Top critical explicit issue"]
            }}
            """
        )

        self.reduce_detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are the Chief Auditor. Merge these section-based detailed JSON reports into one master audit.
            Deduplicate findings and synthesize the final compliance score based on all unique violations.
            Ignore empty sections. Keep the page/section citations intact, INCLUDING the "| Severity: ..."
            field exactly as the section report wrote it. Do not re-rate, upgrade, or drop a severity:
            the section that read the source text assigned it. Where you merge duplicate findings, keep
            the highest severity among them.

            <raw_reports>
            {reports}
            </raw_reports>

            Return ONLY valid JSON matching this structure:
            {{
                "final_compliance_score": 1-100,
                "master_executive_summary": "A comprehensive summary of the entire document's compliance posture.",
                "all_compliant_areas": ["..."],
                "all_unique_violations": ["..."],
                "master_recommendations": ["..."]
            }}
            """
        )

        self.reduce_concise_prompt = ChatPromptTemplate.from_template(
            """
            You are the Chief Auditor. Merge these section-based summary JSON reports into one master overview.
            Deduplicate key issues and calculate the final score based on unique critical issues. Keep citations
            intact, INCLUDING the "| Severity: ..." field exactly as the section report wrote it. Do not re-rate,
            upgrade, or drop a severity. Where you merge duplicates, keep the highest severity among them.

            <raw_reports>
            {reports}
            </raw_reports>

            Return ONLY valid JSON matching this structure:
            {{
                "final_compliance_score": 1-100,
                "master_summary": "A strict 1-2 sentence overall summary.",
                "all_unique_key_issues": ["Merged list of top critical issues"]
            }}

            hey don't be strict, we don't want to miss anything
            """
        )

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_framework_section(self, sec: Dict[str, Any]) -> Dict[str, str]:
        # Pull cleanly from structured JSON
        internal_section_id = self._clean_text(sec.get("section_id")) or self._clean_text(sec.get("control_id")) or "Unknown"
        chapter = self._clean_text(sec.get("chapter"))
        section = self._clean_text(sec.get("section"))
        control = self._clean_text(sec.get("control"))
        subsection = self._clean_text(sec.get("subsection"))
        
        # Build the best available display number
        display_number = (
            self._clean_text(sec.get("section_number"))
            or self._clean_text(sec.get("control_number"))
            or control
            or section
            or chapter
        )
        
        title = self._clean_text(sec.get("title")) or "Untitled Section"
        start_page = self._clean_text(sec.get("start_page")) or "N/A"
        end_page = self._clean_text(sec.get("end_page")) or start_page
        page_label = start_page if start_page == end_page or end_page in {"", "N/A"} else f"{start_page}-{end_page}"

        # Combine for a full display title
        full_title_parts = []
        if display_number:
            full_title_parts.append(display_number)
        if title and title != "Untitled Section":
            full_title_parts.append(title)
        full_title = " ".join(full_title_parts).strip() or "Untitled Section"

        return {
            "internal_section_id": internal_section_id,
            "display_number": display_number,
            "chapter": chapter,
            "section": section,
            "control": control,
            "subsection": subsection,
            "title": title,
            "raw_title": self._clean_text(sec.get("title")),
            "full_title": full_title,
            "start_page": start_page,
            "end_page": end_page,
            "page_label": page_label,
            "text": self._clean_text(sec.get("text")),
        }

    def _extract_company_section_label(self, text: str) -> str:
        # Regex is still needed here *only* to parse the raw unstructured text from the user's uploaded PDF
        patterns = [
            r'(?im)^\s*((?:section|chapter|control|policy)\s+\d+(?:\.\d+)*)\b',
            r'(?im)^\s*(\d+(?:\.\d+){1,5})\s*[-–:.]?\s+(.{3,120})$',
        ]

        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if not match:
                continue
            if idx == 0:
                return match.group(1).strip()
            number = match.group(1).strip()
            title = match.group(2).strip()
            return f"{number} {title}"

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[0][:120]
        return "Unknown Section"

    # =========================================================================
    # RATE-LIMIT AWARE INVOCATION
    # =========================================================================

    @staticmethod
    def _retry_after_seconds(message: str) -> Optional[float]:
        """
        Providers state the wait in the 429 body. Groq writes
        'try again in 967.5ms'; Gemini writes a RetryInfo of 'retryDelay: 14s'.
        """
        m = re.search(r"try again in\s*([\d.]+)\s*(ms|s)\b", message, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return value / 1000.0 if m.group(2).lower() == "ms" else value
        m = re.search(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    def _throttle(self) -> None:
        """
        Spaces outbound calls so a burst of sections cannot trip a
        requests-per-minute cap. An audit fires one request per section plus the
        merge, back to back, which on a per-minute quota means the later sections
        are the ones that get rejected. Pacing up front is cheaper than paying
        for those rejections in backoff afterwards.
        """
        if self._min_call_interval <= 0:
            return
        wait = self._min_call_interval - (time.monotonic() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()

    def _invoke_with_retry(self, chain, payload: Dict[str, Any], what: str, attempts: int = 6,
                           phase: Optional[str] = None):
        """
        Free tiers cap both tokens and requests per minute, and one audit fires a
        request per section plus a merge. Without this, a 429 propagates as an
        ordinary exception: the caller drops that section and carries on, so rate
        limiting quietly removes evidence from the report instead of failing
        loudly. A document is then scored on whichever sections got through.

        _throttle() keeps the request rate under the cap; the backoff below stays
        as the safety net for whatever slips past it.
        """
        delay = 2.0
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                self._throttle()
                result = chain.invoke(payload)
                # Recorded here rather than at the call sites so every retried
                # attempt that actually reached the model is counted once, and
                # a failed attempt (which raises below) is not counted at all.
                if phase:
                    self._token_ledger.record(phase, result)
                return result
            except Exception as e:
                last_error = e
                text = str(e)
                rate_limited = "429" in text or "rate limit" in text.lower()
                if not rate_limited or attempt == attempts:
                    raise
                wait = max(self._retry_after_seconds(text) or 0.0, delay)
                print(
                    f"  [rate-limit] {what}: waiting {wait:.1f}s then retrying "
                    f"(attempt {attempt}/{attempts})"
                )
                time.sleep(wait)
                delay = min(delay * 2, 30.0)
        if last_error:
            raise last_error

    def _attach_token_usage(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prints the cost breakdown and writes the SAME numbers into the returned
        report, so an archived JSON can be costed months later without paying
        for the audit again. Applied to every terminal return of an audit,
        including the failure paths -- a run that broke after 40 map calls still
        spent the money for those 40 calls, and a report that omits that is
        worse than no report.
        """
        try:
            summary = self._token_ledger.summary()
            print(self._token_ledger.render(summary))
            if isinstance(report, dict):
                report["token_usage"] = summary
                report["estimated_cost_usd"] = summary["estimated_cost_usd"]
        except Exception as e:
            print(f"[LYTREX] token accounting summary unavailable: {e}")
        return report

    @staticmethod
    def _message_text(message: Any) -> str:
        """
        A LangChain message carries its content either as a plain string or as
        a list of content blocks. Groq returns the string form; Gemini 3.x
        returns the list form, e.g.

            [{"type": "text", "text": "{...}", "extras": {...}}]

        so reading .content directly yields a list there and every downstream
        string operation fails. .text flattens both shapes -- it is a property
        on current langchain-core and was a method on older releases, so both
        are handled, with a manual join as the last resort.
        """
        text = getattr(message, "text", None)
        if callable(text):
            text = text()
        if isinstance(text, str) and text.strip():
            return text

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)
        return str(content or "")

    def _parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        """
        JsonOutputParser already unwraps ```json fences. This adds one more
        fallback for the case it does not cover: a model that prefixes or
        suffixes the object with prose. Rather than lose the whole section to
        a stray sentence, the outermost balanced {...} is extracted and parsed.
        """
        try:
            return self.output_parser.parse(raw_text)
        except Exception:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(raw_text[start:end + 1])

    def prune_text(self, text: str, max_chars: int) -> str:
        if len(text) > max_chars: return text[:max_chars] + "\n... [TRUNCATED] ..."
        return text

    # =========================================================================
    # REDUCE PHASE: BATCHED / HIERARCHICAL MERGE
    # =========================================================================

    @staticmethod
    def reduce_payload_budget(prompt_template: Any) -> int:
        """
        How many characters of section reports fit in one reduce call.

        The cap is on the rendered prompt, so the template's own length is
        MEASURED here rather than assumed -- rendering it with an empty
        {reports} gives its exact overhead (885 chars for the concise reducer
        as written today). Editing a reduce prompt therefore moves the budget
        automatically instead of quietly overrunning the cap.
        """
        overhead = REDUCE_TEMPLATE_OVERHEAD_FALLBACK
        try:
            rendered = prompt_template.format(reports="")
            if isinstance(rendered, str):
                overhead = len(rendered)
        except Exception as e:
            print(f"[LYTREX] reduce template size unmeasurable ({e}); "
                  f"assuming {REDUCE_TEMPLATE_OVERHEAD_FALLBACK} chars of overhead.")
        return max(REDUCE_PROMPT_MAX_CHARS - overhead, REDUCE_MIN_PAYLOAD_CHARS)

    def _reduce_once(self, payload: str, chain: Any, master_key: str, what: str,
                     stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        One reduce call: invoke, parse, dump. Every call routes through here so
        every call lands in the ledger's 'reduce' phase -- including each batch
        and each merge-of-partials, which is what keeps estimated_cost_usd
        honest once a document needs more than one.
        """
        stats["calls"] = stats.get("calls", 0) + 1
        message = self._invoke_with_retry(
            chain, {"reports": payload}, what=what, phase="reduce",
        )
        raw = self._message_text(message)
        try:
            parsed = self._parse_json_response(raw)
        except Exception as parse_error:
            _debug_dump("REDUCE", raw, None, master_key)
            print(f"[LYTREX] PARSER REJECTED the reduce response ({what}) - {parse_error}")
            raise
        _debug_dump("REDUCE", raw, parsed, master_key)
        if not isinstance(parsed, dict):
            raise ValueError(f"reduce response ({what}) parsed to {type(parsed).__name__}, not an object")
        return parsed

    def _reduce_hierarchical(self, reports: List[Any], chain: Any, master_key: str,
                             budget: int, stats: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """
        Merges any number of section reports into one master report.

        One batch  -> one call, exactly as before batching existed.
        N batches  -> N partial merges, then this same function over the N
                      partials, recursing until one batch remains.

        FAILURE POLICY, stated because it is a real choice: if one batch fails
        the others still merge, and the report carries reduce_batches_failed
        plus a findings_basis of 'reducer_merged_partial'. Losing one batch's
        narrative beats losing every batch's. If ALL batches fail this raises,
        which drops into the caller's existing map-phase fallback -- so a total
        reducer outage still returns the map findings, as it always did. The
        SCORE is untouched either way: it was fixed from the map phase before
        this function was called and is written over the merge afterwards.

        TERMINATION. Every level above the first batches with min_per_batch=2,
        so it emits at most ceil(n/2) batches and the input strictly halves --
        4 partials -> 2 -> 1, and 21 -> 11 -> 6 -> 3 -> 2 -> 1. That holds even
        when the partials are individually enormous, which is the case a plain
        size-only rule cannot converge on: it would emit one batch per partial
        forever. REDUCE_MAX_DEPTH is a backstop for the impossible, not a
        working limit; reaching it merges the remainder in one capped call.
        """
        floor = 1 if depth == 0 else 2
        batches = batch_reports_by_size(reports, budget, min_per_batch=floor)
        truncated = sum(1 for b in batches if b.truncated)

        # Partials that will not fit get compacted rather than cut. Tried only
        # when a cut is otherwise unavoidable, and only above the section
        # level: section reports are the evidence and are never stripped.
        compacted = False
        if truncated and depth > 0:
            slim = [compact_partial_report(r, master_key) for r in reports]
            slim_batches = batch_reports_by_size(slim, budget, min_per_batch=floor)
            kept, total = payload_finding_survival(batches, master_key)
            slim_kept, _ = payload_finding_survival(slim_batches, master_key)
            if slim_kept > kept:
                print(f"[LYTREX] reduce depth {depth}: {len(reports)} partial(s) over budget; "
                      f"compacting to findings + summary lifts surviving findings "
                      f"{kept} -> {slim_kept} of {total}.")
                reports, batches, compacted = slim, slim_batches, True
                truncated = sum(1 for b in batches if b.truncated)
        stats["compacted_levels"] += 1 if compacted else 0
        # Kept apart: a cut at depth 0 means one SECTION REPORT is bigger than
        # the whole budget, which is a document problem. A cut above that is a
        # partial-merge that stayed too big even after compaction, which is a
        # narrative problem. They deserve different alarm.
        if depth == 0:
            stats["oversized_reports"] += truncated
        else:
            stats["truncated_merge_batches"] += truncated
        stats["levels"].append({
            "depth": depth,
            "inputs": len(reports),
            "batches": len(batches),
            "batch_sizes": [len(b.reports) for b in batches],
            "payload_chars": [len(b.payload) for b in batches],
            "truncated_batches": truncated,
            "compacted": compacted,
        })

        if len(batches) == 1:
            # The unchanged path. For a document that fits, batches[0].payload
            # is json.dumps(reports, indent=2) verbatim.
            return self._reduce_once(
                batches[0].payload, chain, master_key,
                what="reduce" if depth == 0 else f"reduce merge (depth {depth})",
                stats=stats,
            )

        if depth >= REDUCE_MAX_DEPTH:
            # Unreachable on any real document; see TERMINATION above. Kept so
            # that a future change to the batching rule cannot turn a bug into
            # an unbounded spend.
            stats["depth_limited"] = True
            print(f"[LYTREX] reduce: depth limit {REDUCE_MAX_DEPTH} reached with "
                  f"{len(reports)} report(s); merging in one capped call.")
            capped = _close_batch(reports, json.dumps(reports, indent=2), budget)
            stats["truncated_merge_batches"] += 1 if capped.truncated else 0
            return self._reduce_once(
                capped.payload, chain, master_key,
                what=f"reduce (capped, depth {depth})", stats=stats,
            )

        print(f"[LYTREX] reduce level {depth}: {len(reports)} report(s) -> {len(batches)} batch(es) "
              f"of {[len(b.reports) for b in batches]} (payload cap {budget:,} chars).")
        partials: List[Dict[str, Any]] = []
        for idx, batch in enumerate(batches, 1):
            try:
                partials.append(self._reduce_once(
                    batch.payload, chain, master_key,
                    what=f"reduce batch {idx}/{len(batches)} (depth {depth})",
                    stats=stats,
                ))
            except Exception as e:
                stats["failed_batches"] += 1
                stats["failed_batch_detail"].append({
                    "depth": depth, "batch": idx,
                    "reports_in_batch": len(batch.reports),
                    "error": str(e)[:300],
                })
                print(f"[LYTREX] reduce batch {idx}/{len(batches)} failed ({e}). "
                      f"{len(batch.reports)} section report(s) will be missing from the "
                      f"displayed narrative; the score is unaffected.")

        if not partials:
            raise RuntimeError(
                f"all {len(batches)} reduce batches failed at depth {depth}"
            )
        if len(partials) == 1:
            return partials[0]
        return self._reduce_hierarchical(partials, chain, master_key, budget, stats, depth + 1)

    @staticmethod
    def _reduce_batching_summary(stats: Dict[str, Any], budget: int,
                                 section_report_count: int) -> Dict[str, Any]:
        """
        The audit trail for the merge: how the reports were grouped, how many
        LLM calls that cost, and -- the number this whole change exists to make
        true -- how many section reports actually reached a merge.

        sections_merged counts the reports in every batch whose merge SUCCEEDED
        at depth 0. Under the old single capped call this figure was whatever
        fitted in 60,000 chars and no more; it should now equal
        sections_audited on any document, of any length.
        """
        level0 = next((lv for lv in stats["levels"] if lv["depth"] == 0), None)
        batches_at_level0 = level0["batches"] if level0 else 0
        merged = section_report_count
        if level0 and stats["failed_batches"]:
            lost = sum(d["reports_in_batch"] for d in stats["failed_batch_detail"]
                       if d["depth"] == 0)
            merged = max(section_report_count - lost, 0)
        return {
            "payload_budget_chars": budget,
            "batches": batches_at_level0,
            "levels": stats["levels"],
            "merge_levels": len(stats["levels"]),
            "batches_failed": stats["failed_batches"],
            "reduce_calls_attempted": stats["calls"],
            "failed_batch_detail": stats["failed_batch_detail"],
            "oversized_section_reports": stats["oversized_reports"],
            "compacted_levels": stats["compacted_levels"],
            "truncated_merge_batches": stats["truncated_merge_batches"],
            "depth_limited": stats["depth_limited"],
            "section_reports_merged": merged,
            "section_reports_total": section_report_count,
            "all_sections_merged": merged == section_report_count,
        }

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace("\n", " ").split(" ")

    def _get_fw_db_path(self, framework_name: str) -> str:
        return os.path.join(self.vector_db_base_path, framework_name.upper())

    def _load_fw_vectorstore(self, framework_name: str):
        path = self._get_fw_db_path(framework_name)
        if os.path.exists(os.path.join(path, "index.faiss")):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    # =========================================================================
    # STRUCTURAL INGESTION (JSON-DRIVEN)
    # =========================================================================
    def ingest_single_framework(self, framework_name: str):
        fw_name = framework_name.upper()
        json_path = os.path.join(self.pdf_source_dir, fw_name, f"{fw_name}.json")
        
        if not os.path.exists(json_path):
            print(f"[!] {json_path} not found. Falling back to blind PDF loading.")
            return self._legacy_ingest_pdf(framework_name)
        
        print(f"[STRUCTURAL INGEST] Loading sections from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict) and "sections" in raw_data:
            sections = raw_data["sections"]
        elif isinstance(raw_data, list):
            sections = raw_data
        else:
            sections = [raw_data]

        c_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size, 
            chunk_overlap=self.child_chunk_overlap
        )

        final_chunks = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue

            normalized = self._normalize_framework_section(sec)
            full_text = normalized["text"]
            if not full_text:
                continue

            children = c_splitter.split_text(full_text)

            for idx, c in enumerate(children):
                doc = Document(
                    page_content=c,
                    metadata={
                        "parent_content": full_text,
                        "fw_internal_section_id": normalized["internal_section_id"],
                        "fw_section_number": normalized["display_number"],
                        "control_id": normalized["control"],
                        "fw_title": normalized["title"],
                        "fw_raw_title": normalized["raw_title"],
                        "domain": normalized["full_title"],
                        "chapter": normalized["chapter"],
                        "section": normalized["section"],
                        "subsection": normalized["subsection"],
                        "source": f"{fw_name}_Framework",
                        "page": normalized["page_label"],
                        "page_start": normalized["start_page"],
                        "page_end": normalized["end_page"],
                        "child_index": idx,
                    }
                )
                final_chunks.append(doc)

        if not final_chunks:
            print("[!] No text could be parsed from the JSON. Check the format.")
            return None

        vectorstore = FAISS.from_documents(final_chunks, self.embeddings)
        out = self._get_fw_db_path(framework_name)
        os.makedirs(out, exist_ok=True)
        vectorstore.save_local(out)
        return vectorstore

    def _legacy_ingest_pdf(self, framework_name: str):
        print("legaccylegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnn runnn")
        pdf_paths = sorted(glob.glob(os.path.join(self.pdf_source_dir, framework_name.upper(), "*.pdf")))
        if not pdf_paths: return None
        all_docs = []
        for path in pdf_paths: all_docs.extend(PyPDFLoader(path).load())
        
        p_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size, chunk_overlap=250)
        c_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size, chunk_overlap=self.child_chunk_overlap)
        
        p_docs = p_splitter.split_documents(all_docs)
        final_chunks = []
        for p in p_docs:
            children = c_splitter.split_text(p.page_content)
            for c in children:
                doc = p.copy()
                doc.page_content = c
                doc.metadata["parent_content"] = p.page_content
                final_chunks.append(doc)
        
        vectorstore = FAISS.from_documents(final_chunks, self.embeddings)
        out = self._get_fw_db_path(framework_name)
        os.makedirs(out, exist_ok=True)
        vectorstore.save_local(out)
        return vectorstore

    # =========================================================================
    # CORE EVALUATION LOGIC
    # =========================================================================
    def audit_large_document(self, target_pdf_path: str, framework_name: str, k: int = 10, summary_mode: bool = False, evaluate_llm: bool = True, reset_token_ledger: bool = True) -> Dict[str, Any]:
        # check_compliance() owns the ledger when it is the caller, because its
        # relevance-gate call is billed against the same audit and happens
        # before this method is entered. A direct call owns the ledger itself.
        if reset_token_ledger:
            self._token_ledger.reset()

        vectorstore = self._load_fw_vectorstore(framework_name) or self.ingest_single_framework(framework_name)
        if not vectorstore: return {"error": "Framework not found."}

        all_docs = sorted(list(vectorstore.docstore._dict.values()), key=lambda x: x.page_content)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)

        documents = PyPDFLoader(target_pdf_path).load()
        section_splitter = RecursiveCharacterTextSplitter(chunk_size=self.map_chunk_size, chunk_overlap=self.map_chunk_overlap)
        sections = section_splitter.split_documents(documents)
        
        # USE CONSTRUCTOR PARAMETERS HERE
        small_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.section_chunk_size, 
            chunk_overlap=self.section_chunk_overlap
        )
        
        all_reports = []
        # Which section produced each entry of all_reports, parallel to it.
        # Kept as a separate list rather than a field on the report dict because
        # all_reports is serialised verbatim into the reducer prompt, and adding
        # a key there would change what the reducer sees.
        audited_section_ids: List[int] = []
        raw_retrieval_log = {}
        failed_sections = 0
        # Sections whose retrieval produced nothing above the relevance floor.
        # Kept apart from failed_sections, which means a parse failure.
        skipped_sections = 0
        skipped_section_details: List[Dict[str, Any]] = []
        forwarded_counts: List[int] = []
        # THE COVERAGE DENOMINATOR. The distinct framework control ids that were
        # retrieved into a section prompt AND whose section report came back
        # parseable. Ids from a section that failed to parse are deliberately
        # NOT counted: nothing was examined there, and crediting them would
        # enlarge the denominator without any matching chance of a finding,
        # which inflates the score exactly where the audit broke.
        assessed_control_ids: set = set()
        
        mode_str = "SUMMARY" if summary_mode else "DETAILED"
        if not evaluate_llm: mode_str = "RETRIEVAL ONLY (Bypassing LLM)"
            
        print(f"\n[LYTREX] Processing {len(sections)} sections ({mode_str})...")

        for i, section in enumerate(sections, 1):
            
            # --- NEW PIPELINE: Split -> small chunks -> BM25+FAISS for each chunk ---
            section_small_chunks = small_chunk_splitter.split_text(section.page_content)
            
            combined_children = []
            
            for small_chunk in section_small_chunks:
                # Retrieve candidates per chunk (using dynamic multiplier to manage load)
                chunk_faiss = vectorstore.similarity_search(small_chunk, k=k*2)
                
                tokenized_chunk = self._tokenize(small_chunk)
                chunk_bm25 = bm25.get_top_n(tokenized_chunk, all_docs, n=k*2)
                
                combined_children.extend(chunk_faiss)
                combined_children.extend(chunk_bm25)
            # -------------------------------------------------------------------------

            # Merge all candidates and Deduplicate
            unique_children_map = {}
            for child in combined_children:
                if child.page_content not in unique_children_map:
                    unique_children_map[child.page_content] = child

            unique_children = list(unique_children_map.values())
            if not unique_children: continue

            # Cross-Encoder (Section vs Candidate)
            pairs = [[section.page_content, child.page_content] for child in unique_children]
            scores = self.reranker.predict(pairs)
            
            # Top-K results, but only among candidates that clear the
            # relevance floor. The slice used to be unconditional, so every
            # section received k controls whether or not any were on topic and
            # the model reported the off-topic ones as "not documented here".
            # See RERANK_MIN_SCORE for the scale and the chosen value.
            scored_children = sorted(zip(scores, unique_children), key=lambda x: x[0], reverse=True)
            best_score = float(scored_children[0][0])
            relevant = [(float(sc), ch) for sc, ch in scored_children if float(sc) >= RERANK_MIN_SCORE]
            top_k_children = [child for score, child in relevant[:k]]

            if not top_k_children:
                # No framework control is topically related to this section.
                # Auditing it against the k least-irrelevant controls is what
                # manufactured the repeated absence findings, so the section is
                # recorded and skipped rather than fed noise. It is deliberately
                # NOT counted in sections_audited (nothing was audited) nor in
                # sections_failed (nothing failed) -- it gets its own counter.
                skipped_sections += 1
                skipped_section_details.append({
                    "section": i,
                    "best_score": round(best_score, 4),
                    "candidates_considered": len(unique_children),
                })
                print(f"  -> Section {i}: SKIPPED, no framework control cleared the "
                      f"relevance floor (best {best_score:.4f} < {RERANK_MIN_SCORE}).")
                if not evaluate_llm:
                    raw_retrieval_log[f"Section_{i}"] = {
                        "query": section.page_content,
                        "context": "",
                        "skipped_reason": "no framework control cleared the relevance floor",
                        "best_score": round(best_score, 4),
                    }
                continue

            forwarded_counts.append(len(top_k_children))

            # Context Assembly (Parent chunks)
            unique_parents = []
            parent_meta_map = {}
            # Controls forwarded to THIS section. Merged into the audit-wide
            # denominator only if the section's report parses; see
            # assessed_control_ids above.
            section_control_ids: set = set()
            for child in top_k_children:
                parent_text = child.metadata.get("parent_content", child.page_content)
                if parent_text not in parent_meta_map:
                    fw_internal_section_id = child.metadata.get("fw_internal_section_id", "N/A")
                    # Same identifier space as _finding_control_key(), which
                    # reads SEC_0007 out of a finding's citation, so numerator
                    # and denominator count the same unit.
                    if fw_internal_section_id and fw_internal_section_id != "N/A":
                        section_control_ids.add(str(fw_internal_section_id).strip())
                    fw_section_number = child.metadata.get("fw_section_number", "")
                    fw_title = child.metadata.get("fw_title") or child.metadata.get("fw_raw_title") or child.metadata.get("domain", "N/A")
                    fw_page = child.metadata.get("page", "N/A")
                    
                    meta_bits = [f"FRAMEWORK ID: {fw_internal_section_id}", f"FRAMEWORK PAGE(S): {fw_page}"]
                    if fw_section_number:
                        meta_bits.insert(1, f"FRAMEWORK SECTION/CONTROL: {fw_section_number}")
                    if fw_title:
                        meta_bits.append(f"FRAMEWORK TITLE: {fw_title}")

                    tagged_fw_text = f"--- {' | '.join(meta_bits)} ---\n{parent_text}"
                    parent_meta_map[parent_text] = tagged_fw_text
                    unique_parents.append(parent_text)

            formatted_context = "\n\n---\n\n".join([parent_meta_map[p] for p in unique_parents])

            actual_doc_page = int(section.metadata.get("page", 0)) + 1
            doc_source = os.path.basename(section.metadata.get("source", "Company Document"))
            company_section_label = self._extract_company_section_label(section.page_content)
            
            chunk_with_metadata = (
                f"--- COMPANY DOCUMENT | FILE: {doc_source} | COMPANY PAGE: {actual_doc_page} | "
                f"COMPANY SECTION: {company_section_label} ---\n{section.page_content}"
            )

            if not evaluate_llm:
                raw_retrieval_log[f"Section_{i}"] = {
                    "query": chunk_with_metadata,
                    "context": formatted_context
                }
                print(f"  -> Section {i}: Context retrieved and reranked successfully.")
                continue
                
            # LLM (Micro-Evaluation for this section)
            active_prompt = self.concise_prompt if summary_mode else self.detailed_prompt
            # The parser is applied separately rather than piped in, so the raw
            # model text can be inspected before it is consumed. TEMPORARY.
            chain = active_prompt | self.llm
            section_findings_key = "key_issues" if summary_mode else "violations"

            try:
                message = self._invoke_with_retry(
                    chain,
                    {"context": self.prune_text(formatted_context, 12000),
                     "company_doc": self.prune_text(chunk_with_metadata, 4000)},
                    what=f"section {i}",
                    phase="map",
                )

                raw_text = self._message_text(message)
                if DEBUG_RAW_RESPONSES:
                    reasoning = ""
                    try:
                        reasoning = (getattr(message, "additional_kwargs", {}) or {}).get("reasoning", "") or ""
                    except Exception:
                        reasoning = ""
                    print(f"\n[debug] section {i}: content={len(raw_text)} chars, "
                          f"separate reasoning field={len(reasoning)} chars")

                # Parsing happens here, on the exact string logged above.
                try:
                    report = self._parse_json_response(raw_text)
                except Exception as parse_error:
                    _debug_dump(f"SECTION {i}", raw_text, None, section_findings_key)
                    print(f"  -> Section {i}: PARSER REJECTED the response - {parse_error}")
                    raise

                _debug_dump(f"SECTION {i}", raw_text, report, section_findings_key)

                # --- NEW COMPLIANCE SCORING LOGIC COMPUTED WITHOUT LLM INTERVENTION ---
                violations_list = report.get("key_issues", []) if summary_mode else report.get("violations", [])
                calc_score = 100.0
                for _ in violations_list:
                    # Decrease by ~5% randomized slightly to prevent duplicate scores like 75 75
                    # calc_score -= (5.0 * random.uniform(0.4, 0.6))
                    calc_score -= (5.0)
                
                report["compliance_score"] = round(max(0.0, calc_score), 2)
                # -----------------------------------------------------------------------

                all_reports.append(report) # Store JSON result
                audited_section_ids.append(i)
                # The section was examined, so its controls join the denominator.
                assessed_control_ids.update(section_control_ids)
                score = report.get("compliance_score", "N/A")
                print(f"  -> Section {i}: Audited successfully. [Score: {score}]")
            except Exception as e:
                failed_sections += 1
                print(f"  -> Section {i}: Failed parsing - {str(e)}")

        if not evaluate_llm: return {"raw_retrieval_results": raw_retrieval_log}
        if not all_reports:
            return self._attach_token_usage(
                {"error": "Failed to generate any valid section reports."}
            )

        # Every finding the map phase actually produced, in order, deduplicated
        # on exact text across sections. This is the SCORING INPUT: see
        # score_basis_from_map_findings for why the reducer's output is not.
        findings_key = "key_issues" if summary_mode else "violations"
        map_findings: List[str] = []
        # Per-section provenance. Not used for scoring; persisted so a finding
        # can be traced back to the chunk that raised it, which is what makes
        # retrieval and chunking artifacts measurable from an archived report
        # instead of only from a live console.
        map_findings_by_section: List[Dict[str, Any]] = []
        for section_id, report in zip(audited_section_ids, all_reports):
            section_items: List[str] = []
            for item in (report.get(findings_key) or []):
                if not isinstance(item, str) or not item.strip():
                    continue
                text = item.strip()
                section_items.append(text)
                if text not in map_findings:
                    map_findings.append(text)
            map_findings_by_section.append({
                "section": section_id,
                "findings": section_items,
            })

        # The score is fixed here, before the reducer is ever called, so nothing
        # the reducer does to the wording can move it.
        score_basis = score_basis_from_map_findings(map_findings, assessed_control_ids)
        score_basis["map_phase_findings"] = map_findings
        score_basis["map_phase_findings_by_section"] = map_findings_by_section

        if failed_sections:
            print(f"[LYTREX] WARNING: {failed_sections} section(s) failed to parse and were dropped.")
        if skipped_sections:
            avg_fwd = (sum(forwarded_counts) / len(forwarded_counts)) if forwarded_counts else 0.0
            print(f"[LYTREX] {skipped_sections} section(s) skipped: no framework control scored "
                  f">= {RERANK_MIN_SCORE} against them. Audited sections received "
                  f"{avg_fwd:.1f} controls on average (cap k={k}).")
        print(f"[LYTREX] Map phase produced {len(map_findings)} unique finding(s) across {len(all_reports)} section report(s).")
        print(
            f"[LYTREX] SCORED from map phase: {score_basis['map_phase_contradiction_count']} "
            f"contradiction, {score_basis['map_phase_absence_count']} absence "
            f"-> score {score_basis['final_compliance_score']}. "
            f"The merge step below changes the displayed findings, not this number."
        )
        if score_basis.get("score_model") == "coverage":
            _cen = score_basis["severity_census"]
            print(
                f"[LYTREX] MODEL=coverage: failure mass {score_basis['failure_mass']} over "
                f"{score_basis['controls_assessed']} control(s) assessed "
                f"(ratio {score_basis['coverage_ratio']}, strain "
                f"{score_basis['coverage_strain']}). Severity labelled by the model on "
                f"{score_basis['findings_with_explicit_severity']} finding(s); the rest "
                f"defaulted to '{DEFAULT_SEVERITY_TIER}'. Census "
                f"critical={_cen['critical']}, moderate={_cen['moderate']}, minor={_cen['minor']}. "
                f"This is examined-scope coverage, not verified compliance."
            )
        else:
            print(
                f"[LYTREX] MODEL=penalty_estimate: no control ids were recorded, so there is "
                f"no denominator and this score is NOT a coverage figure."
            )

        # Deduplicate findings -> Aggregate scores -> LLM Final Evaluation -> Final JSON Output
        print(f"\n[LYTREX] Reducing {len(all_reports)} section reports into Master Report...")
        active_reduce_prompt = self.reduce_concise_prompt if summary_mode else self.reduce_detailed_prompt
        # Parser applied separately so the reducer's raw text is inspectable. TEMPORARY.
        chain = active_reduce_prompt | self.llm
        master_key = "all_unique_key_issues" if summary_mode else "all_unique_violations"
        # Batching bookkeeping, published on the report so the shape of the
        # merge is inspectable offline rather than only in a console log.
        reduce_stats: Dict[str, Any] = {
            "levels": [], "failed_batches": 0, "failed_batch_detail": [],
            "oversized_reports": 0, "depth_limited": False, "calls": 0,
            "compacted_levels": 0, "truncated_merge_batches": 0,
        }
        reduce_budget = self.reduce_payload_budget(active_reduce_prompt)
        try:
            # The reducer used to see every section report in ONE call, capped
            # at 60,000 chars by a straight text[:n]. Past ~46-84 sections that
            # cap silently discarded every later section and handed the model a
            # payload cut mid-JSON. Now the reports are grouped into batches
            # that fit, merged into partials, and the partials merged in turn --
            # so every section report reaches a merge no matter how long the
            # document is. See batch_reports_by_size.
            final_report = self._reduce_hierarchical(
                all_reports, chain, master_key, reduce_budget, reduce_stats,
            )

            master_violations = final_report.get(master_key) or []
            if not isinstance(master_violations, list):
                master_violations = []
            # Where the DISPLAYED list came from. Distinct from scoring_basis,
            # which is always the map phase. '..._partial' means at least one
            # batch's merge failed, so the narrative below covers fewer section
            # reports than the score does -- the same asymmetry the 60,000-char
            # cap used to create silently, now named on the report.
            displayed_basis = (
                "reducer_merged_partial" if reduce_stats["failed_batches"] else "reducer_merged"
            )

            # A reducer that drops findings is indistinguishable from a clean
            # policy once only the score survives: both render as 100/100. The
            # map phase already did the auditing, so when the merge comes back
            # empty its result is not trusted over evidence already in hand.
            if not master_violations and map_findings:
                print(
                    f"[LYTREX] WARNING: reducer returned 0 findings under '{master_key}' "
                    f"but the map phase found {len(map_findings)}. "
                    f"Falling back to the map findings so they are not silently discarded."
                )
                master_violations = map_findings
                final_report[master_key] = map_findings
                final_report["reducer_fallback_used"] = True
                displayed_basis = "map_phase_unmerged"

            # Split contradictions from absences and collapse the absences that
            # name the same control, then score the two classes separately.
            buckets = classify_and_dedupe(master_violations)
            # Contradictions are ordered strongest-signal first; see
            # rank_contradictions for what the signals are and are not.
            contradictions, contradiction_ranking = rank_contradictions(
                buckets["contradictions"],
                [r.get(findings_key) or [] for r in all_reports],
            )
            absences = buckets["absences"]
            # master_key stays contradictions-first, then coverage gaps, so the
            # existing frontend leads with real breaches without any change.
            kept = contradictions + absences

            final_report[master_key] = kept
            # The same findings, split so a consumer can render genuine
            # contradictions apart from coverage gaps instead of parsing the
            # flat list back apart. master_key above is kept populated and
            # unchanged in membership for backward compatibility.
            final_report["contradiction_findings"] = contradictions
            final_report["coverage_gap_findings"] = absences
            final_report["contradiction_ranking"] = contradiction_ranking

            # DISPLAYED counts. These keep their original meaning: they describe
            # the list rendered above, so a consumer that prints a count next to
            # its own list never shows a number that disagrees with it. They are
            # deliberately NOT the scored counts -- the score comes from the map
            # phase, and score_basis_note below states that in words.
            final_report["contradiction_count"] = len(contradictions)
            final_report["absence_count"] = len(absences)
            final_report["absence_duplicates_removed"] = buckets["absence_duplicates_removed"]
            final_report["contradiction_duplicates_removed"] = buckets["contradiction_duplicates_removed"]
            # Explicit aliases, so nothing has to rely on the unprefixed names
            # meaning "displayed" by convention.
            final_report["displayed_contradiction_count"] = len(contradictions)
            final_report["displayed_absence_count"] = len(absences)
            final_report["displayed_finding_count"] = len(kept)
            final_report["findings_basis"] = displayed_basis

            # SCORED basis, applied last so it owns final_compliance_score
            # outright -- the reducer states a score of its own in that field and
            # it is replaced here, as it was before this change. The invariant a
            # reader can check offline is:
            #   compliance_score(map_phase_contradiction_count,
            #                    map_phase_absence_count) == final_compliance_score
            final_report.update(score_basis)
            final_report["score_basis_note"] = score_basis_note(score_basis, len(kept))
            final_report["sections_audited"] = len(all_reports)
            final_report["sections_failed"] = failed_sections
            final_report["sections_skipped_no_relevant_controls"] = skipped_sections
            final_report["skipped_sections_detail"] = skipped_section_details
            final_report["retrieval_relevance_threshold"] = RERANK_MIN_SCORE
            final_report["reduce_batching"] = self._reduce_batching_summary(
                reduce_stats, reduce_budget, len(all_reports)
            )

            print(
                f"[LYTREX] Findings in: {len(master_violations)} -> kept {len(kept)} "
                f"({len(contradictions)} contradiction, {len(absences)} absence; "
                f"removed {buckets['absence_duplicates_removed']} duplicate absence, "
                f"{buckets['contradiction_duplicates_removed']} duplicate contradiction)"
            )
            print(
                f"[LYTREX] Master report score {final_report['final_compliance_score']} "
                f"(from the map phase; the merge step's {len(kept)} displayed "
                f"finding(s) did not set it)."
            )
            _rb = final_report["reduce_batching"]
            print(
                f"[LYTREX] MERGE COVERAGE: {_rb['section_reports_merged']}/"
                f"{_rb['section_reports_total']} section report(s) reached a merge, in "
                f"{_rb['batches']} batch(es) over {_rb['merge_levels']} level(s), "
                f"{_rb['reduce_calls_attempted']} reduce call(s)."
            )
            if _rb["batches_failed"]:
                print(
                    f"[LYTREX] WARNING: {_rb['batches_failed']} reduce batch(es) failed. The "
                    f"displayed narrative is PARTIAL (findings_basis="
                    f"'{final_report['findings_basis']}'); the score still covers all "
                    f"{_rb['section_reports_total']} section report(s)."
                )
            if _rb["oversized_section_reports"]:
                print(
                    f"[LYTREX] WARNING: {_rb['oversized_section_reports']} section report(s) "
                    f"exceed the {reduce_budget:,}-char batch cap on their own and were sent "
                    f"truncated rather than dropped."
                )
            if _rb["truncated_merge_batches"]:
                print(
                    f"[LYTREX] NOTE: {_rb['truncated_merge_batches']} partial-merge batch(es) "
                    f"were still over budget after compaction and were cut. Every section "
                    f"report still reached a merge; some merged wording did not survive the "
                    f"final synthesis. Raise LYTREX_REDUCE_PROMPT_MAX_CHARS to avoid this."
                )

            return self._attach_token_usage(final_report)
        except Exception as e:
            # The map phase already found real issues; losing them to a reducer
            # failure would report a perfect score for a document known to be bad.
            print(f"[LYTREX] Reducer failed ({e}). Falling back to map-phase findings.")
            if map_findings:
                summary_field = "master_summary" if summary_mode else "master_executive_summary"
                # The score was already fixed from the map phase above, so this
                # branch does not recompute it -- a reducer failure now returns
                # exactly the score a successful reduce would have returned for
                # the same document. The classification is reused from
                # score_basis rather than recomputed, so the two cannot drift.
                fb_contradictions, fb_ranking = rank_contradictions(
                    score_basis["map_phase_contradiction_findings"],
                    [r.get(findings_key) or [] for r in all_reports],
                )
                fb_absences = score_basis["map_phase_absence_findings"]
                fb_kept = fb_contradictions + fb_absences
                fallback_report = {
                    master_key: fb_kept,
                    "contradiction_findings": fb_contradictions,
                    "coverage_gap_findings": fb_absences,
                    "contradiction_ranking": fb_ranking,
                    summary_field: (
                        f"The section-level audit found {len(fb_kept)} issue(s): "
                        f"{len(fb_contradictions)} contradiction(s) and "
                        f"{len(fb_absences)} coverage gap(s). "
                        f"The final merge step failed, so these are reported unmerged."
                    ),
                    # Here the displayed list IS the map-phase list, so the
                    # displayed and scored counts coincide by construction.
                    "contradiction_count": len(fb_contradictions),
                    "absence_count": len(fb_absences),
                    "absence_duplicates_removed":
                        score_basis["map_phase_absence_duplicates_removed"],
                    "contradiction_duplicates_removed":
                        score_basis["map_phase_contradiction_duplicates_removed"],
                    "displayed_contradiction_count": len(fb_contradictions),
                    "displayed_absence_count": len(fb_absences),
                    "displayed_finding_count": len(fb_kept),
                    "findings_basis": "map_phase_unmerged",
                    "sections_audited": len(all_reports),
                    "sections_failed": failed_sections,
                    "sections_skipped_no_relevant_controls": skipped_sections,
                    "skipped_sections_detail": skipped_section_details,
                    "retrieval_relevance_threshold": RERANK_MIN_SCORE,
                    "reducer_fallback_used": True,
                    # Records what the merge managed before it gave out --
                    # including the batches that did succeed and were then
                    # discarded because no partial survived. Costed calls are
                    # already in the ledger, so the report should say what they
                    # bought.
                    "reduce_batching": self._reduce_batching_summary(
                        reduce_stats, reduce_budget, len(all_reports)
                    ),
                }
                fallback_report.update(score_basis)
                fallback_report["score_basis_note"] = score_basis_note(
                    score_basis, len(fb_kept)
                )
                return self._attach_token_usage(fallback_report)
            return self._attach_token_usage({"error": f"Reducer LLM Error: {str(e)}"})

    # =========================================================================
    # THE API BRIDGE METHOD 
    # =========================================================================
    def check_compliance(self, target_pdf_path: str, framework_name: str, k: int = 10, detailed: bool = False) -> Dict[str, Any]:
        # One audit, one ledger. Reset here rather than inside
        # audit_large_document because the relevance gate below is already a
        # billable call against this same audit.
        self._token_ledger.reset()
        try:
            docs = PyPDFLoader(target_pdf_path).load()
            if not docs: return {"error": "Uploaded PDF is empty or unreadable."}
            
            # Gatekeeper (Relevance Check)
            first_page_text = docs[0].page_content[:3000]
            # The parser is applied separately rather than piped in, for the
            # same reason the map and reduce chains do it: piping it in replaces
            # the response message with a plain dict, and the token counts on
            # that message are gone before anything can read them. The parse
            # itself is identical -- JsonOutputParser accepts a message.
            gate_chain = self.relevance_prompt | self.llm
            gate_message = gate_chain.invoke({"text": first_page_text})
            self._token_ledger.record("gate", gate_message)
            gate_result = self.output_parser.invoke(gate_message)
            
            if not gate_result.get("is_relevant", True):
                return {
                    "error": "Document Rejected: The file does not appear to be a corporate policy or procedure.",
                    "llm_reasoning": gate_result.get("reasoning", "Irrelevant document type.")
                }
        except Exception as e:
            print(f"[Warning] Relevance Gate failed, proceeding to audit: {str(e)}")

        summary_mode = not detailed
        return self.audit_large_document(
            target_pdf_path=target_pdf_path,
            framework_name=framework_name,
            k=k,
            summary_mode=summary_mode,
            evaluate_llm=True,
            reset_token_ledger=False,
        )

    def get_framework_full_text(self, framework_name: str) -> str:
        fw_name = framework_name.upper()
        json_path = os.path.join(self.pdf_source_dir, fw_name, f"{fw_name}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            if isinstance(raw_data, dict) and "sections" in raw_data:
                sections = raw_data["sections"]
            elif isinstance(raw_data, list):
                sections = raw_data
            else:
                sections = [raw_data]
                
            return "\n\n".join([s.get("text", "") for s in sections if isinstance(s, dict)])
        
        pdf_paths = sorted(glob.glob(os.path.join(self.pdf_source_dir, fw_name, "*.pdf")))
        full_text = []
        for p in pdf_paths: full_text.extend(doc.page_content for doc in PyPDFLoader(p).load())
        return "\n\n".join(full_text)