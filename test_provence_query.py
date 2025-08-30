#!/usr/bin/env python3
"""
Test different Provence queries on Utah social media act sections
"""

from pathlib import Path
from provence_integration import ProvenceProcessor

# Read the Utah document
ROOT = Path(__file__).resolve().parents[0]
utah_doc = ROOT / "data" / "laws" / "us_ut_social_media_act.md"
content = utah_doc.read_text()

# Extract just Section 13-2c-301 content for testing
lines = content.split('\n')
section_301_start = None
section_301_end = None

for i, line in enumerate(lines):
    if "Section 13-2c-301 - Minor Access Restrictions" in line:
        section_301_start = i
    elif section_301_start and line.strip().startswith("### _Section 13-2c-302"):
        section_301_end = i
        break

if section_301_start and section_301_end:
    section_301_content = '\n'.join(lines[section_301_start:section_301_end])
else:
    print("Could not extract Section 13-2c-301")
    exit(1)

print("ORIGINAL SECTION 13-2c-301 CONTENT:")
print("=" * 60)
print(section_301_content)
print("\n" + "=" * 60)

# Initialize Provence
provence = ProvenceProcessor(threshold=0.2)

# Test different queries
queries = [
    "What regulations must providers comply with",  # Current query
    "What are the requirements for platforms or providers to comply",  # Your suggestion
    "What must platforms do to comply with curfew requirements",  # More specific
    "What are the access restriction requirements for Utah",  # Geographic specific
]

for query in queries:
    print(f"\n🔍 TESTING QUERY: '{query}'")
    print("-" * 60)
    
    result = provence.prune_context(
        question=query,
        context=section_301_content,
        custom_threshold=0.2
    )
    
    print(f"Compliance Score: {result['reranking_score']:.3f}")
    print(f"Compression Ratio: {result['compression_ratio']:.3f}")
    print(f"Words: {len(result['pruned_context'].split())}")
    print("\nPRUNED CONTENT:")
    print(result['pruned_context'])
    print()