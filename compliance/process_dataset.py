#!/usr/bin/env python3

import csv
import json
import os
from pathlib import Path
from typing import List, Dict
from compliance.classifier import ComplianceClassifier, ComplianceResult

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "test_dataset.csv"  # Input test dataset
OUTPUT_CSV = ROOT / "outputs" / "compliance_results.csv"  # Output results
OUTPUT_JSON = ROOT / "outputs" / "compliance_results.json"  # Detailed JSON results

def process_test_dataset(input_file: Path, output_csv: Path, output_json: Path):
    """Process the test dataset and generate compliance classifications"""
    
    if not input_file.exists():
        raise FileNotFoundError(f"Test dataset not found: {input_file}")
    
    # Initialize classifier
    print("Initializing compliance classifier...")
    classifier = ComplianceClassifier()
    
    # Read input CSV
    features = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            features.append({
                "feature_id": f"FEAT-{i+1:03d}",
                "feature_name": row.get("feature_name", "").strip(),
                "feature_description": row.get("feature_description", "").strip()
            })
    
    print(f"Processing {len(features)} features...")
    
    # Analyze features
    results = classifier.analyze_batch(features)
    
    # Create output directories
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV output (required format for submission)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "feature_id",
            "feature_name", 
            "feature_description",
            "needs_geo_compliance",
            "confidence_score",
            "risk_level",
            "reasoning",
            "applicable_regulations",
            "top_legal_matches"
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Format regulations and legal matches
            regulations = "; ".join(result.applicable_regulations) if result.applicable_regulations else "None"
            
            top_matches = []
            for passage in result.retrieved_passages[:3]:  # Top 3 matches
                match_info = f"{passage['jurisdiction']} {passage['law']} (relevance: {passage['relevance_score']:.3f})"
                top_matches.append(match_info)
            
            writer.writerow({
                "feature_id": result.feature_id,
                "feature_name": result.title,
                "feature_description": features[int(result.feature_id.split('-')[1])-1]["feature_description"],
                "needs_geo_compliance": result.needs_compliance,
                "confidence_score": f"{result.confidence_score:.3f}",
                "risk_level": result.risk_level,
                "reasoning": result.reasoning,
                "applicable_regulations": regulations,
                "top_legal_matches": "; ".join(top_matches)
            })
    
    # Write detailed JSON output for analysis
    json_results = []
    for i, result in enumerate(results):
        json_results.append({
            "feature_id": result.feature_id,
            "feature_name": result.title,
            "feature_description": features[i]["feature_description"],
            "classification": {
                "needs_geo_compliance": result.needs_compliance,
                "confidence_score": result.confidence_score,
                "risk_level": result.risk_level
            },
            "reasoning": result.reasoning,
            "applicable_regulations": result.applicable_regulations,
            "legal_analysis": {
                "retrieved_passages": [
                    {
                        "law": f"{p['jurisdiction']} {p['law']}",
                        "relevance_score": p["relevance_score"],
                        "text_snippet": p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"],
                        "source_url": p["source_url"]
                    }
                    for p in result.retrieved_passages
                ]
            }
        })
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results written to:")
    print(f"  CSV: {output_csv}")
    print(f"  JSON: {output_json}")
    
    # Print summary statistics
    classifications = [r.needs_compliance for r in results]
    print(f"\n📊 Summary Statistics:")
    print(f"  Total features analyzed: {len(results)}")
    print(f"  Needs compliance: {classifications.count('yes')}")
    print(f"  No compliance needed: {classifications.count('no')}")
    print(f"  Uncertain cases: {classifications.count('uncertain')}")
    
    confidence_scores = [r.confidence_score for r in results]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    print(f"  Average confidence: {avg_confidence:.3f}")
    
    risk_levels = [r.risk_level for r in results]
    print(f"  Risk distribution:")
    for risk in ["critical", "high", "medium", "low"]:
        count = risk_levels.count(risk)
        print(f"    {risk}: {count}")

def create_sample_dataset():
    """Create sample test dataset from the hackathon examples"""
    
    sample_data = [
        ("Curfew login blocker with ASL and GH for Utah minors", "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries. The feature activates during restricted night hours and logs activity using EchoTrace for auditability. This allows parental control to be enacted without user-facing alerts, operating in ShadowMode during initial rollout."),
        
        ("PF default toggle with NR enforcement for California teens", "As part of compliance with California's SB976, the app will disable PF by default for users under 18 located in California. This default setting is considered NR to override, unless explicit parental opt-in is provided. Geo-detection is handled via GH, and rollout is monitored with FR logs. The design ensures minimal disruption while meeting the strict personalization requirements imposed by the law."),
        
        ("Child abuse content scanner using T5 and CDS triggers", "In line with the US federal law requiring providers to report child sexual abuse content to NCMEC, this feature scans uploads and flags suspected materials tagged as T5. Once flagged, the CDS auto-generates reports and routes them via secure channel APIs. The logic runs in real-time, supports human validation, and logs detection metadata for internal audits. Regional thresholds are governed by LCP parameters in the backend."),
        
        ("Content visibility lock with NSP for EU DSA", "To meet the transparency expectations of the EU Digital Services Act, we are introducing a visibility lock for flagged user-generated content labeled under NSP. When such content is detected, a soft Softblock is applied and GH ensures enforcement is restricted to the EU region only. EchoTrace supports traceability, and Redline status can be triggered for legal review. This feature enhances accountability and complies with Article 16's removal mechanisms."),
        
        ("Jellybean-based parental notifications for Florida regulation", "To support Florida's Online Protections for Minors law, this feature extends the Jellybean parental control framework. Notifications are dispatched to verified parent accounts when a minor attempts to access restricted features. Using IMT, the system checks behavioral anomalies against BB models. If violations are detected, restrictions are applied in ShadowMode with full audit logging through CDS. Glow flags ensure compliance visibility during rollout phases."),
        
        ("Universal PF deactivation on guest mode", "By default, PF will be turned off for all users browsing in guest mode."),
        
        ("Story resharing with content expiry", "Enable users to reshare stories from others, with auto-expiry after 48 hours. This feature logs resharing attempts with EchoTrace and stores activity under BB."),
        
        ("Leaderboard system for weekly creators", "Introduce a creator leaderboard updated weekly using internal analytics. Points and rankings are stored in FR metadata and tracked using IMT."),
        
        ("Regional trial of autoplay behavior", "Enable video autoplay only for users in US. GH filters users, while Spanner logs click-through deltas."),
        
        ("Age-specific notification controls with ASL", "Notifications will be tailored by age using ASL, allowing us to throttle or suppress push alerts for minors. EchoTrace will log adjustments, and CDS will verify enforcement across rollout waves."),
    ]
    
    # Write sample dataset
    sample_file = ROOT / "data" / "test_dataset.csv"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "feature_description"])
        writer.writerows(sample_data)
    
    print(f"✅ Created sample dataset: {sample_file}")
    return sample_file

def main():
    """Main processing function"""
    
    # Create sample dataset if it doesn't exist
    if not INPUT_CSV.exists():
        print("Creating sample test dataset...")
        create_sample_dataset()
    
    # Process the dataset
    process_test_dataset(INPUT_CSV, OUTPUT_CSV, OUTPUT_JSON)

if __name__ == "__main__":
    main()