# Improving LLM via KG integration and prompt engineering
Create a new file called `openai_api_key.txt` and insert your OpenAI API key there.

## Preliminary Results
| Model                        | EM (MMLU Geography) | EM (MMLU Government & Politics) |
|------------------------------|---------------------|---------------------------------|
| GPT 3.5                      | 0.6565656565656566  | 0.6839378238341969              |
| GPT 3.5 + DBPedia (baseline) | 0.5202020202020202  | 0.694300518134715               |
| GPT 3.5 + Wikidata           | 0.5050505050505051  | 0.6683937823834197              |