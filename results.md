## Preliminary Results
| Model                                                        | EM (MMLU Geography) | EM (MMLU Government & Politics) |
|--------------------------------------------------------------|---------------------|---------------------------------|
| GPT 3.5                                                      | 0.6565656565656566  | 0.6839378238341969              |
| GPT 3.5 + DBPedia (baseline)                                 | 0.5202020202020202  | 0.694300518134715               |
| GPT 3.5 + Wikidata (baseline)                                | 0.5050505050505051  | 0.6683937823834197              |
| GPT 3.5 + Wikidata (baseline, 2nd Trial after fixing errors) | 0.48484848484848486 | 0.6632124352331606              |
| GPT 3.5 + Wikidata (Evaluation using similarity measure)     | 0.5707070707070707  | 0.7409326424870466              |

# Filtered Dataset Results

## Number of questions
| Dataset                                  | Number of questions (unfiltered) | Number of questions (filtered) |
|------------------------------------------|----------------------------------|--------------------------------|
| MMLU Geography                           | 198                              | 114                            |
| MMLU Government & Politics               | 193                              | 90                             |
| MMLU Miscellaneous                       | 783                              | 512                            |

## EM Scores
| Model                                   | EM (MMLU Geography) | EM (MMLU Government & Politics) | EM (MMLU Miscellaneous) |
|-----------------------------------------|---------------------|---------------------------------|-------------------------|
| GPT 3.5                                 | 0.7192982456140351  | 0.7888888888888889              | 0.884765625             |
| GPT 3.5 + Wikidata (Cos^2 Sim. Measure) | 0.7192982456140351  | 0.8                             |                         |

# Run times
| Model                                   | Run time (MMLU Geography) | Run time (MMLU Government & Politics) | Run time (MMLU Miscellaneous) |
|-----------------------------------------|---------------------------|---------------------------------------|-------------------------------|
| GPT 3.5                                 | 4 min 10 s                | 4 min 2 s                             | 23 min 44 s                   |
| GPT 3.5 + Wikidata (Cos^2 Sim. Measure) | 2 h 42 min 40 s           | 2 h 16 min 7 s                        |                               |
