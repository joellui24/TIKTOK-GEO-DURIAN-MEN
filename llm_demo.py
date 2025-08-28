#!/usr/bin/env python3

import sys
import csv
from pathlib import Path
from compliance.llm_classifier import LLMComplianceClassifier

ROOT = Path(__file__).resolve().parent

def run_llm_demo():
    """Run the LLM-enhanced compliance classification demo"""
    
    print("🚀 TikTok Geo-Compliance Classification System (LLM-Enhanced)")
    print("=" * 70)
    print()
    print("This system uses LLM + semantic search to classify features")
    print("without hardcoded patterns, providing nuanced legal reasoning.")
    print()
    
    # Initialize classifier
    try:
        print("🔧 Initializing LLM classifier...")
        classifier = LLMComplianceClassifier()
        print("✅ System ready!")
        print()
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        print("\nSetup instructions:")
        print("1. Copy .env.example to .env")
        print("2. Add your API key: PERPLEXITY_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY")
        print("3. Set LLM_PROVIDER: 'perplexity', 'anthropic', or 'openai'")
        sys.exit(1)
    
    # Test with enhanced examples
    test_features = [
        {
            "feature_id": "LLM-001",
            "feature_name": "Utah minor curfew system with ASL enforcement",
            "feature_description": "Implement time-based access restrictions for Utah residents under 18, automatically logging out users between 10:30 PM and 6:00 AM. Uses ASL (age-sensitive logic) to identify minors and GH (geo-handler) to apply restrictions only to Utah-located users. Includes parental override capability."
        },
        {
            "feature_id": "LLM-002", 
            "feature_name": "California teen personalized feed controls",
            "feature_description": "Default disable personalized algorithmic feeds for California users under 18. Requires explicit parental consent to enable PF (personalized feed). Uses NR (non-revocable) setting unless parent provides verified opt-in through Jellybean parental control system."
        },
        {
            "feature_id": "LLM-003",
            "feature_name": "CSAM detection and NCMEC reporting pipeline", 
            "feature_description": "Automated content scanning system using T5 classification to identify suspected child sexual abuse material. Automatically reports flagged content to NCMEC as required by federal law. Integrates with CDS (compliance detection system) for audit trail."
        },
        {
            "feature_id": "LLM-004",
            "feature_name": "EU DSA illegal content reporting mechanism",
            "feature_description": "Notice-and-action system for EU users to report illegal content. Provides transparent reporting flow with automated acknowledgment and review timeline. Implements Article 24 requirements for Digital Services Act compliance."
        },
        {
            "feature_id": "LLM-005",
            "feature_name": "Florida parental notification system",
            "feature_description": "Automated alerts to parents when minor accounts in Florida change privacy settings, add new contacts, or receive direct messages from unknown users. Uses Jellybean framework for secure parent-child account linking."
        },
        {
            "feature_id": "LLM-006",
            "feature_name": "Creator monetization leaderboard",
            "feature_description": "Weekly ranking system for content creators based on engagement metrics. Shows top performers globally with analytics dashboard. Integrates with creator fund eligibility and payment systems."
        },
        {
            "feature_id": "LLM-007",
            "feature_name": "Global autoplay optimization experiment",
            "feature_description": "A/B testing different autoplay behaviors across regions to optimize user engagement. Tracks metrics like watch time, scroll rate, and session duration. Uses ML models to personalize autoplay timing."
        }
    ]
    
    print("📊 ANALYZING TEST FEATURES WITH LLM")
    print("=" * 70)
    
    results = classifier.analyze_batch(test_features)
    
    # Display results
    for result in results:
        status_emoji = {"yes": "🔴", "no": "🟢", "uncertain": "🟡"}
        risk_emoji = {"critical": "🚨", "high": "⚠️", "medium": "🟨", "low": "🟦"}
        
        print(f"\n{'-'*70}")
        print(f"🏷️  {result.title}")
        print(f"{status_emoji.get(result.needs_compliance, '❓')} Needs Compliance: {result.needs_compliance.upper()}")
        print(f"📊 Confidence: {result.confidence_score:.2f}/1.0")
        print(f"{risk_emoji.get(result.risk_level, '❓')} Risk: {result.risk_level.upper()}")
        
        if result.applicable_regulations:
            print(f"📋 Regulations: {', '.join(result.applicable_regulations)}")
        
        print(f"💭 LLM Reasoning: {result.reasoning}")
        
        if result.retrieved_passages:
            top_match = result.retrieved_passages[0]
            print(f"⚖️  Top Legal Match: {top_match['jurisdiction']} {top_match['law']} (relevance: {top_match['relevance_score']:.3f})")
    
    # Generate enhanced statistics
    print(f"\n{'='*70}")
    print("📈 ENHANCED ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    classifications = [r.needs_compliance for r in results]
    print(f"Total features analyzed: {len(results)}")
    print(f"  🔴 Needs compliance: {classifications.count('yes')}")
    print(f"  🟢 No compliance needed: {classifications.count('no')}")
    print(f"  🟡 Uncertain cases: {classifications.count('uncertain')}")
    
    confidence_scores = [r.confidence_score for r in results]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    print(f"Average confidence: {avg_confidence:.3f}")
    
    risk_levels = [r.risk_level for r in results]
    print(f"Risk distribution:")
    for risk in ["critical", "high", "medium", "low"]:
        count = risk_levels.count(risk)
        emoji = {"critical": "🚨", "high": "⚠️", "medium": "🟨", "low": "🟦"}
        print(f"  {emoji[risk]} {risk}: {count}")
    
    # Save enhanced results
    output_file = ROOT / "outputs" / "llm_demo_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "feature_id", "feature_name", "needs_geo_compliance",
            "confidence_score", "risk_level", "llm_reasoning", 
            "applicable_regulations", "top_legal_match"
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            regulations = "; ".join(result.applicable_regulations) if result.applicable_regulations else "None"
            top_match = ""
            if result.retrieved_passages:
                top = result.retrieved_passages[0]
                top_match = f"{top['jurisdiction']} {top['law']} (relevance: {top['relevance_score']:.3f})"
            
            writer.writerow({
                "feature_id": result.feature_id,
                "feature_name": result.title,
                "needs_geo_compliance": result.needs_compliance,
                "confidence_score": f"{result.confidence_score:.3f}",
                "risk_level": result.risk_level,
                "llm_reasoning": result.reasoning,
                "applicable_regulations": regulations,
                "top_legal_match": top_match
            })
    
    print(f"\n💾 Enhanced results saved to: {output_file}")
    
    print(f"\n🎯 LLM ANALYSIS INSIGHTS:")
    print("• Utah curfew and California PF features should be flagged as compliance-required")
    print("• CSAM detection should be identified as critical federal compliance")
    print("• EU DSA reporting should be recognized as medium-risk geo-specific requirement")
    print("• Business features should be correctly classified as non-compliance")
    print("• Reasoning should cite specific legal provisions and requirements")
    
    print(f"\n🚀 LLM SYSTEM ADVANTAGES:")
    print("• No hardcoded patterns - adapts to new regulations dynamically")
    print("• Nuanced legal reasoning with specific law citations")
    print("• Handles complex compliance scenarios and edge cases")
    print("• Explainable decisions with detailed justification")
    print("• Semantic understanding of legal context and requirements")

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        print("Usage: python3 llm_demo.py")
        print("This demo runs LLM-enhanced compliance analysis.")
        print("Make sure to set up API keys in .env file first.")
        sys.exit(1)
    
    run_llm_demo()

if __name__ == "__main__":
    main()