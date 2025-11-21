# cuda_diagnostic.py
import torch
import os
import subprocess

print("=== CUDA Diagnostic Report ===")

# System info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# GPU info
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name}")
        print(f"  Compute Capability: {prop.major}.{prop.minor}")
        print(f"  Total Memory: {prop.total_memory / 1e9:.1f} GB")
        print(f"  Multi Processor Count: {prop.multi_processor_count}")

# Test CUDA step by step
print("\n=== CUDA Functionality Test ===")
if torch.cuda.is_available():
    try:
        print("1. Testing CUDA device...")
        device = torch.device('cuda')
        print(f"   ✓ Device created: {device}")
        
        print("2. Testing CUDA context...")
        torch.cuda.init()
        print("   ✓ CUDA initialized")
        
        print("3. Testing current device...")
        current_device = torch.cuda.current_device()
        print(f"   ✓ Current device: {current_device}")
        
        print("4. Testing memory allocation...")
        memory_allocated = torch.cuda.memory_allocated()
        print(f"   ✓ Memory allocated: {memory_allocated} bytes")
        
        print("5. Testing simple tensor creation...")
        # Try with a very simple tensor first
        x = torch.tensor([1.0])
        print(f"   ✓ CPU tensor created")
        
        print("6. Testing tensor transfer to GPU...")
        x_gpu = x.cuda()
        print(f"   ✓ Tensor moved to GPU: {x_gpu.device}")
        
        print("7. Testing GPU computation...")
        result = x_gpu + 1.0
        print(f"   ✓ GPU computation successful: {result.item()}")
        
        print("🎉 ALL CUDA TESTS PASSED! CUDA is working correctly.")
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
else:
    print("❌ CUDA not available")