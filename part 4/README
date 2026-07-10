# Part 4 — LLM-Powered Feature: Structured JSON Extraction (Track A)

## Chosen Track
Track A — Structured JSON Extraction

## Overview
This feature extracts structured customer-support data (customer name, product name,
issue type, sentiment, refund requested) from free-text customer messages using an LLM,
validates the output against a JSON schema, and applies a PII guardrail before every call.

## LLM Provider
OpenRouter (https://openrouter.ai), model: `openrouter/free` — OpenRouter's auto-router,
which selects from currently available free-tier models per request to avoid single-model
rate limits. Accessed via Python `requests` with a JSON body containing `model` and
`messages` fields, matching the OpenAI-compatible chat completions format.

## Setup
1. Create a free OpenRouter account and API key at https://openrouter.ai/keys.
2. Set the environment variable `LLM_API_KEY` (never hardcoded in code).
3. Install dependencies: `pip install requests jsonschema`

## System Prompt (verbatim)
You are a structured data extractor. Given a raw customer support message, extract exactly these fields and output ONLY valid JSON, with no extra text, no markdown formatting, and no explanation:
{
"customer_name": string,
"product_name": string,
"issue_type": string,
"sentiment": "positive" | "neutral" | "negative",
"refund_requested": true | false
}
## Few-Shot User Prompt Template
Example 1
Input: "Hi, this is Priya. My Bluetooth headphones (SoundMax Pro) stopped charging after two days. Very disappointed, I want my money back."
Output: {"customer_name": "Priya", "product_name": "SoundMax Pro", "issue_type": "charging failure", "sentiment": "negative", "refund_requested": true}
Example 2
Input: "Hello, I'm Arjun. Just wanted to say the AeroFit Running Shoes I bought are fantastic, great cushioning and fit true to size."
Output: {"customer_name": "Arjun", "product_name": "AeroFit Running Shoes", "issue_type": "none", "sentiment": "positive", "refund_requested": false}
Now extract from this input:
Input: "<actual customer message>"
Output:
## Why temperature=0
Temperature=0 makes the model always pick its highest-probability token at each step,
producing deterministic, repeatable output — important for structured data extraction
where the same input should ideally yield the same JSON. Temperature=0.7 introduces
sampling randomness, useful for creative tasks but undesirable here since it can vary
field wording (e.g. "leaking" vs "packaging failure") for the same input, even when the
underlying facts are unchanged.

## Temperature A/B Comparison

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Neha / GlowSkin Serum | `{"customer_name":"Neha","product_name":"GlowSkin Face Serum","issue_type":"packaging failure","sentiment":"negative","refund_requested":true}` | `{"customer_name":"Neha","product_name":"GlowSkin Face Serum","issue_type":"leaking","sentiment":"negative","refund_requested":true}` | Same facts, different wording for `issue_type` |
| Kabir / ChronoWatch X1 | `{"customer_name":"Kabir","product_name":"ChronoWatch X1","issue_type":"none","sentiment":"positive","refund_requested":false}` | `{"customer_name":"Kabir","product_name":"ChronoWatch X1","issue_type":"none","sentiment":"positive","refund_requested":false}` | Identical output at both temperatures |
| Simran / UrbanTrek Backpack | `{"customer_name":"Simran","product_name":"UrbanTrek Backpack","issue_type":"delayed shipment and broken zipper","sentiment":"negative","refund_requested":true}` | `{"customer_name":"Simran","product_name":"UrbanTrek Backpack","issue_type":"delayed delivery","sentiment":"negative","refund_requested":true}` | Different phrasing/completeness of `issue_type` |

## Structured Output Handling
- JSON Schema defined with 5 required fields: `customer_name` (string), `product_name`
  (string), `issue_type` (string), `sentiment` (string enum: positive/neutral/negative),
  `refund_requested` (boolean).
- Each LLM response is parsed with `json.loads()` inside a `try/except JSONDecodeError`
  block.
- The parsed dict is validated with `jsonschema.validate()` inside a
  `try/except jsonschema.ValidationError` block.
- On any parsing or validation failure, a fallback dict with all 5 required fields set
  to `None`/`null` is returned, the error is printed, and processing continues without
  crashing.

## PII Guardrail
Before every `call_llm(...)`, the raw user input is checked with a regex for email
addresses and 10-digit / formatted phone numbers. If PII is detected, the LLM is never
called; `"Input blocked: PII detected."` is printed and `None` is returned instead.

Guardrail test results:
- Input containing an email (`rahul.k@email.com`) → **Blocked**, LLM not called.
- Input with no PII → **Passed through**, LLM called normally.

## End-to-End Demonstration

| Input | LLM Output | Valid JSON (pass/fail) | Pass/Block (guardrail) |
|---|---|---|---|
| Neha / GlowSkin Serum | `{"customer_name":"Neha","product_name":"GlowSkin Face Serum","issue_type":"packaging failure","sentiment":"negative","refund_requested":true}` | pass | Pass |
| Kabir / ChronoWatch X1 | `{"customer_name":"Kabir","product_name":"ChronoWatch X1","issue_type":"none","sentiment":"positive","refund_requested":false}` | pass | Pass |
| Simran / UrbanTrek Backpack | `{"customer_name":"Simran","product_name":"UrbanTrek Backpack","issue_type":"delayed shipment and broken zipper","sentiment":"negative","refund_requested":true}` | pass | Pass |
| Rahul (contains email) | — | — | Blocked |

## Findings
Across the three test inputs, all final outputs were valid, schema-conforming JSON with
correct field values. The `issue_type` field showed minor wording variation between
temp=0 and temp=0.7 runs, confirming that temperature=0 reduces but does not fully
guarantee identical wording across separate calls, especially when routed through
`openrouter/free`, which may select a different underlying model per request. During
development, two transient failure modes were also observed and correctly handled: one
call returned a non-JSON safety-classifier response instead of structured output, and
one call returned malformed JSON. Both were caught by the `try/except` JSON parsing and
schema validation, and a safe fallback (`None`/null-filled dict) was returned instead of
the pipeline crashing — demonstrating graceful degradation under real-world LLM
unreliability.

## How to Run
1. Open the notebook in Google Colab.
2. Run all cells top to bottom.
3. When prompted, paste your OpenRouter API key (get one free at openrouter.ai/keys).
4. All outputs (guardrail test, extraction demo, temperature comparison) print inline.
