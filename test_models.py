#!/usr/bin/env python3
"""
Test script to verify that all LLM models can generate output properly.
This helps debug model-specific issues.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    LLM_AVAILABLE = True
except ImportError:
    print("❌ LLM dependencies not available. Install transformers and torch.")
    sys.exit(1)

def test_model_generation(model_name, test_prompt="Hello, how are you?"):
    """Test if a model can generate text properly"""
    print(f"\n🧪 Testing model: {model_name}")
    print(f"📝 Test prompt: {test_prompt}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle tokenizer configuration
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True
        )
        
        # Resize embeddings if needed
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # Format prompt based on model
        if "phi-2" in model_name.lower():
            formatted_prompt = f"Instruction: {test_prompt}\nResponse:"
        elif "tinyllama" in model_name.lower():
            formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{test_prompt}</s>\n<|assistant|>\n"
        elif "qwen" in model_name.lower():
            formatted_prompt = f"Human: {test_prompt}\nAssistant:"
        else:
            formatted_prompt = test_prompt
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"✅ Response: {response.strip()[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error with {model_name}: {str(e)}")
        return False

def main():
    """Test all available models"""
    print("🤖 LLM Model Test Suite")
    print("=" * 50)
    
    models_to_test = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "Qwen/Qwen2-0.5B-Instruct"
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