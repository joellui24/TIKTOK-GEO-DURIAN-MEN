# TikTok Geo-Compliance JSON Generator

🚀 **Sonar Integration Tool** - Generate structured JSON for automated compliance analysis with TikTok feature context and legal article retrieval.

## 🎯 Overview

This tool generates structured JSON templates for Sonar to analyze whether TikTok features require geo-specific compliance. It automatically:
- Retrieves relevant legal articles using semantic search
- Expands TikTok internal jargon for proper context
- Formats everything into Sonar's required JSON structure

## ⚡ Quick Setup

### Prerequisites
- Python 3.8+
- 2-3 GB free disk space (for embedding models)

### Installation
```bash
# 1. Clone and setup virtual environment
git clone <repository-url>
cd TIKTOK-GEO-DURIAN-MEN
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the legal knowledge base (one-time setup, ~2-3 minutes)
python index/build.py
```

### Verify Setup
```bash
# Test search functionality
python index/smoke_test.py "Utah curfew system"

# Should return relevant legal articles about curfew requirements
```

## 🚀 Using the JSON Generator

### Basic Usage
```bash
python generate_sonar_compliance_json.py "Feature Title" "Feature Description"
```

### Examples
```bash
# Curfew system example
python generate_sonar_compliance_json.py "Curfew login blocker with ASL and GH for Utah minors" "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries."

# Age verification example  
python generate_sonar_compliance_json.py "Age verification system" "A system that verifies user age before allowing platform access"

# Data handling example
python generate_sonar_compliance_json.py "T5 data handling with Spanner and Redline for CDS compliance" "Implementing T5 sensitive data processing using Spanner rule engine with Redline legal review flags for CDS compliance monitoring"
```

### Output
The script generates a complete JSON structure with:
- **TikTok jargon expansions** (ASL, GH, T5, etc.)
- **Top 5 relevant legal articles** from the knowledge base
- **Structured format** ready for Sonar analysis
- **Clear constraints** requiring article-based reasoning

## 🏗️ How It Works

```
Feature Title + Description
        ↓
[TikTok Jargon Expansion] ← ingest/tiktok_jargons.json
        ↓
[Semantic Legal Search] ← ChromaDB vector database
        ↓
[Article Retrieval] ← Top 5 most relevant legal articles
        ↓
[JSON Formatting] ← Structured output for Sonar
        ↓
[Sonar Analysis] ← Determines geo-compliance requirements
```

## 🔧 Technologies

- **FastEmbed** - BAAI/bge-small-en-v1.5 embedding model for semantic search
- **ChromaDB** - Vector database for legal article storage and retrieval
- **Provence AI** - Content pruning for compliance-focused embeddings
- **Python** - Core logic and JSON generation
- **JSON** - Structured output format for Sonar integration

## 📁 Project Structure

```
├── data/
│   ├── laws/                   # Legal document sources
│   │   ├── us_ut_social_media_act.md  # Utah minor protection laws
│   │   ├── us_ca_sb976.md             # California SB976 restrictions  
│   │   ├── eu_dsa.md                  # EU Digital Services Act
│   │   ├── us_fl_hb3.md               # Florida minor protections
│   │   └── us_18usc_2258a.md          # Federal CSAM reporting
│   └── index/chroma/           # ChromaDB vector index
├── index/
│   ├── build.py               # Builds the legal knowledge base
│   └── smoke_test.py          # Interactive search testing
├── ingest/
│   ├── tiktok_jargons.json    # TikTok internal terminology
│   └── manifest.json          # Document registry and metadata
├── generate_sonar_compliance_json.py  # 🚀 Main JSON generator
├── provence_integration.py    # Provence AI integration
└── requirements.txt           # Python dependencies
```

## 🏷️ TikTok Jargon Support

The system automatically expands TikTok internal terminology:

| Jargon | Expansion |
|--------|-----------|
| `ASL` | Age-sensitive logic |
| `GH` | Geo-handler; module for routing features based on user region |
| `T5` | Tier 5 sensitivity data; more critical than T1-T4 |
| `Spanner` | Synthetic name for a rule engine |
| `Redline` | Flag for legal review |
| `ShadowMode` | Deploy feature in non-user-impact way to collect analytics |
| `EchoTrace` | Log tracing mode to verify compliance routing |
| `CDS` | Compliance Detection System |

*Full list: 20+ terms in `ingest/tiktok_jargons.json`*

## 📋 Example JSON Output

```json
{
  "task": "Based solely on the provided articles, determine whether the described feature requires geo-specific compliance.",
  "feature_title": "Curfew login blocker with ASL and GH for Utah minors",
  "feature_description": "System using ASL to detect minor accounts and GH for Utah enforcement...",
  "Expanded_tiktok_jargons": {
    "ASL": "Age-sensitive logic",
    "GH": "Geo-handler; module responsible for routing features based on user region"
  },
  "articles": [
    {
      "Law": "US-Utah",
      "Article": "Minor Access Restrictions", 
      "Section": "Section 13-2c-301",
      "Content": "Social media platforms must restrict access for users under 18 between 10:30 PM and 6:00 AM..."
    }
  ],
  "output_format_json": {
    "verdict": "yes | no | uncertain",
    "reasoning": "Concise, article-based justification, maximum 500 characters",
    "references": ["List of relevant laws"]
  }
}
```

## 🚀 Legal Coverage

Current knowledge base includes:
- **🇺🇸 Utah Social Media Regulation Act** - Minor curfews and parental controls
- **🇺🇸 California SB976** - Personalized feed restrictions for teens  
- **🇺🇸 Florida HB3** - Social media age restrictions
- **🇺🇸 18 USC 2258A** - Federal CSAM reporting requirements
- **🇪🇺 Digital Services Act** - Content moderation and transparency

## 🛠️ Troubleshooting

### Common Issues

**IndexError: vector index not found**
```bash
# Rebuild the index
python index/build.py
```

**ModuleNotFoundError**  
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

**Slow first run**
- First-time embedding model download (~1GB)
- ChromaDB index creation (~2-3 minutes)  
- Subsequent runs are fast (<1 second)

### Debug Mode
```bash
# Test search functionality
python index/smoke_test.py

# Interactive mode - try different queries
python index/smoke_test.py
```

---

*Streamlined for Sonar integration - Generate compliance-aware JSON for any TikTok feature.* 🎯
