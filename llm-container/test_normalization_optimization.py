#!/usr/bin/env python3
"""
Test script to verify normalization optimization:
- Normalization should happen ONCE per patient answer
- Pre-normalized text should be reused for all guidelines
"""

import requests
import json
import sys

def test_normalization_optimization():
    """Test that normalization happens once per patient answer"""
    
    base_url = "http://localhost:5001"
    
    # Test case: vague location that needs clarification
    test_queries = [
        "I have abdominal pain",
        "right side",
        "right upper quadrant"
    ]
    
    session_id = "test_normalization_optimization"
    
    print("🧪 TESTING NORMALIZATION OPTIMIZATION")
    print("=" * 60)
    print(f"📋 Session ID: {session_id}")
    print(f"🎯 Testing: Normalization happens ONCE per patient answer\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"📝 Query {i}: '{query}'")
        print(f"{'='*60}\n")
        
        try:
            response = requests.post(
                f"{base_url}/chat-tg",
                json={
                    "prompt": query,
                    "session_id": session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract debug info
                debug_info = data.get('debug', {})
                engine_debug = debug_info.get('engine_debug_output', [])
                
                # Count normalization messages
                normalization_count = 0
                for line in engine_debug:
                    if '🔄 Normalization:' in line or '📍 Clarification normalization' in line:
                        normalization_count += 1
                        print(f"  {line}")
                
                print(f"\n📊 Normalization count: {normalization_count}")
                
                if normalization_count == 1:
                    print("✅ OPTIMIZATION WORKING: Normalization happened ONCE per answer")
                elif normalization_count == 0:
                    print("⚠️  No normalization debug output found (may be using cached normalization)")
                else:
                    print(f"❌ ISSUE: Normalization happened {normalization_count} times (should be 1)")
                
                # Show response
                response_text = data.get('response', '')
                if response_text:
                    print(f"\n💬 Response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
                
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"Response: {response.text[:500]}")
                
        except requests.exceptions.ConnectionError:
            print("❌ ERROR: Cannot connect to container")
            print("💡 Make sure the container is running:")
            print("   docker compose -f setup/docker-compose.yml up -d llm")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🧪 TEST COMPLETE")
    print(f"{'='*60}")
    print("\n💡 Expected behavior:")
    print("   - Each patient answer normalizes ONCE")
    print("   - That normalized text is reused for all guidelines")
    print("   - Debug output shows '🔄 Normalization:' once per answer")

if __name__ == "__main__":
    test_normalization_optimization()

