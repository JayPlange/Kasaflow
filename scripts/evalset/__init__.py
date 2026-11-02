"""
Conversation eval set for KasaFlow: capture, anonymise, label, replay, score.

    python -m scripts.evalset.anonymise LOGFILE [LOGFILE ...]   # traces -> evalset/conversations.jsonl
    python -m scripts.evalset.label                             # add new conversations to evalset/labels.json
    python -m scripts.evalset.run --mode recorded|live [--repeat N]
    python -m scripts.evalset.dashboard                         # results -> eval_reports/*.html

See evalset/README.md for the full workflow and what each metric means.
"""
