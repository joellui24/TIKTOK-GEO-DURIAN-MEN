# TikTok Geo-Compliance Classification System

🚀 **TikTok Hackathon Entry** - An AI-powered system that automatically flags product features requiring geo-specific compliance logic.

## 🎯 Problem Statement

TikTok operates globally and must comply with dozens of geographic regulations. This system provides automated visibility into:
- "Does this feature require dedicated logic to comply with region-specific legal obligations?"
- "How many features have we rolled out to ensure compliance with this regulation?"

Without automated compliance screening, TikTok faces:
- ⚖️ Legal exposure from undetected compliance gaps
- 🛑 Reactive firefighting when regulators inquire
- 🚧 Manual overhead in scaling global feature rollouts

## 🌟 Solution Highlights

- 🔍 **Semantic Legal Search**: Vector-based search through regulation documents
- 🤖 **Intelligent Classification**: Determines if features need geo-compliance logic
- 📊 **Risk Assessment**: Critical/High/Medium/Low risk scoring
- 🏷️ **Terminology Handling**: Expands TikTok-specific jargon (ASL, GH, CDS, etc.)
- ⚖️ **Regulation Mapping**: Links features to specific laws
- 🔄 **Uncertainty Detection**: Flags ambiguous cases for human review
- 💾 **Production Ready**: CSV in → CSV out, fully explainable results

## ⚡ Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build legal knowledge base
python3 index/build.py

# Run demo
python3 simple_demo.py
```

## 📊 Demo Results

The system correctly classifies test features:

| Feature | Classification | Risk | Regulation |
|---------|---------------|------|------------|
| Utah Curfew System | ✅ Compliance Required | Medium | Utah Social Media Regulation Act |
| California Teen PF Controls | ✅ Compliance Required | Medium | California SB976 |
| CSAM Detection | ✅ Compliance Required | **Critical** | 18 USC 2258A |
| Creator Leaderboard | ❌ No Compliance | Low | None |
| Universal PF Changes | ❓ Uncertain | Medium | Human Review |

## 🏗️ Architecture

```
Feature Description
        ↓
[Terminology Expansion] ← TikTok jargon dictionary  
        ↓
[Legal Document Search] ← Vector DB of regulations
        ↓
[Pattern Classification] ← Rule-based compliance detection
        ↓
[Risk & Regulation ID] ← Critical/High/Medium/Low + specific laws
        ↓
[CSV Output + Reasoning] ← Business-friendly results
```

## 🔧 Technologies

- **FastEmbed** - Semantic embeddings for legal search
- **ChromaDB** - Vector database for regulation documents
- **Python** - Classification logic and pattern matching
- **Regex** - Compliance indicator detection
- **CSV** - Production-ready input/output format

## 📁 Project Structure

```
├── compliance/              # Core classification system
│   ├── simple_classifier.py   # Rule-based classifier (no LLM needed)
│   └── classifier.py          # Advanced LLM-based classifier  
├── data/laws/              # Legal documents with metadata
├── index/                  # Search and indexing tools
├── ingest/
│   ├── terms.json         # TikTok terminology dictionary
│   └── manifest.json      # Document registry
├── simple_demo.py         # Main demo (works offline)
└── outputs/               # Generated compliance reports
```

## 🎯 Key Innovations

### 1. **Terminology-Aware Classification**
Handles TikTok's internal jargon:
- `ASL` → Age-sensitive logic
- `GH` → Geo-handler routing
- `CDS` → Compliance detection system
- `Jellybean` → Parental control framework

### 2. **Uncertainty-Aware Decisions**
Flags ambiguous cases like "For compliance concerns, we limit feature Y in SEA, EU and JP" for human review rather than guessing.

### 3. **Multi-Stage RAG Pipeline**
Combines semantic search with rule-based patterns for explainable, accurate classifications.

### 4. **Production-Ready Output**
CSV format with detailed reasoning, confidence scores, and source citations ready for compliance teams.

## 🚀 Business Impact

- **Reduce Legal Exposure**: Catch compliance gaps before global deployment
- **Audit-Ready Evidence**: Generate traceable compliance screening records
- **Scale Global Rollouts**: Automate geo-compliance checks for rapid iteration
- **Lower Governance Costs**: Reduce manual legal review overhead

## 🔜 Future Enhancements

- 📱 **Code Analysis**: Scan actual feature code for geo-handlers
- 🌍 **Expanded Coverage**: Add more jurisdictions (GDPR, Brazil, India)
- 🤖 **LLM Integration**: Advanced reasoning with GPT-4/Claude
- 📊 **Web Dashboard**: Compliance team interface
- 🔄 **Learning Loop**: Improve from human feedback

---

*Built for TikTok Hackathon 2024 - Automating geo-compliance to enable safer, faster global feature rollouts.* 🌍
