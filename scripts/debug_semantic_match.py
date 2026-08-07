"""
One-off diagnostic: show the top 5 semantic-search candidates (with
scores) for a query, instead of just the top_k=1 that get_product_price
actually uses. Not a permanent part of the pipeline -- just to see how
close "Golden Necklace, 10g" is actually scoring against what a query
should really match, and by how much it's winning.

Usage:
    python scripts/debug_semantic_match.py "gold ring"
    python scripts/debug_semantic_match.py "gold bracelet"
    python scripts/debug_semantic_match.py "silver bracelet"
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from services.product_search import get_product_index

query = " ".join(sys.argv[1:]) or "gold ring"

print(f"Query: {query!r}\n")
matches = get_product_index().search(query, top_k=5, min_score=0.0)  # no cutoff, see everything
for i, m in enumerate(matches, 1):
    print(f"{i}. score={m['score']:.4f}  {m['product']!r}  material={m['material']!r}  price={m['price']}")
