#!/usr/bin/env python3
"""
Test script to verify that all LLM models can generate output properly using Ollama.
This helps debug model-specific issues.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("✅ Testing Ollama-based LLM models")

def test_model_generation(model_name, test_prompt="Hello, how are you?"):
    """Test if a model can generate text properly using Ollama"""
    print(f"\n🧪 Testing model: {model_name}")
    print(f"📝 Test prompt: {test_prompt}")
    
    try:
        import subprocess
        
        # Check if Ollama is available
        print("Checking Ollama availability...")
        try:
            subprocess.run(["ollama", "list"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("❌ Ollama is not available")
            return False
        
        print("Using Ollama to generate response...")
        
        # Use Ollama to generate response
        cmd = ["ollama", "run", model_name, test_prompt]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout for testing
        )
        
        if result.returncode != 0:
            print(f"❌ Ollama error: {result.stderr}")
            return False
        
        response = result.stdout.strip()
        
        if not response:
            print("❌ Empty response from model")
            return False
        
        print(f"✅ Response: {response[:200]}...")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout waiting for model response")
        return False
    except Exception as e:
        print(f"❌ Error with {model_name}: {str(e)}")
        return False

def main():
    """Test all available models"""
    print("🤖 LLM Model Test Suite")
    print("=" * 50)
    
    models_to_test = [
        "qwen2.5:1.5b",
        "gemma2:2b"
    ]
    
    test_prompt = "Summarize the following text: The meeting discussed quarterly results and future plans."
    
    results = {}
    for model_name in models_to_test:
        try:
            success = test_model_generation(model_name, test_prompt)
            results[model_name] = success
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error testing {model_name}: {e}")
            results[model_name] = False
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name.split('/')[-1]:20} {status}")
    
    # Overall result
    passed = sum(results.values())
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} models working correctly")

if __name__ == "__main__":
    main()