#!/usr/bin/env python3
"""
Debug script to check ML system integration
"""

def debug_ml_system():
    """Debug the ML system integration"""
    print("🔍 Debugging ML System Integration")
    print("=" * 50)
    
    try:
        from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
        engine = AdaptiveDiagnosticEngine()
        
        print(f"✅ Adaptive Diagnostic Engine initialized")
        print(f"📊 Medical Rule Engine available: {engine.medical_rule_engine is not None}")
        print(f"📊 Learning Collector available: {engine.learning_collector is not None}")
        print(f"📊 Performance Monitor available: {engine.performance_monitor is not None}")
        
        if engine.medical_rule_engine:
            print("\n🧪 Testing Medical Rule Engine directly...")
            result = engine.medical_rule_engine.get_enhanced_similarity(
                'left side of my abdomen',
                'RIGHT LOWER QUADRANT (RLQ) pain',
                'Acute Appendicitis'
            )
            print(f"✅ Direct Medical Rule Engine result: {result}")
            
            print("\n🧪 Testing through Adaptive Diagnostic Engine...")
            similarity = engine._compute_enhanced_location_similarity(
                'left side of my abdomen',
                'RIGHT LOWER QUADRANT (RLQ) pain',
                'Acute Appendicitis'
            )
            print(f"✅ Through Adaptive Diagnostic Engine: {similarity}")
            
        else:
            print("❌ Medical Rule Engine not available - checking imports...")
            try:
                from ml.medical_rule_engine import MedicalRuleEngine
                print("✅ Medical Rule Engine import successful")
                print("❌ But not initialized in Adaptive Diagnostic Engine")
            except Exception as e:
                print(f"❌ Medical Rule Engine import failed: {e}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ml_system()
