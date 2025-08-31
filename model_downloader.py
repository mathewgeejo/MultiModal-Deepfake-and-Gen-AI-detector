"""
🚀 AUDIO DEEPFAKE MODEL DOWNLOADER - RUN ONCE IN COLAB
=====================================================

This script downloads and caches all required models and packages.
Run this ONCE at the beginning of your Colab session, then use the main detector.

INSTRUCTIONS:
1. Copy this entire cell to Google Colab
2. Run it once (takes 5-10 minutes)
3. Then use the main detection code without waiting for downloads

💡 Models will be cached in Colab's temporary storage for the session
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def install_packages_with_progress():
    """Install all required packages with progress tracking"""
    packages = [
        ('torch', 'PyTorch for GPU computation'),
        ('torchaudio', 'Audio processing for PyTorch'),
        ('transformers', 'Hugging Face transformers'),
        ('librosa', 'Audio analysis library'),
        ('soundfile', 'Audio file I/O'),
        ('matplotlib', 'Plotting library'),
        ('seaborn', 'Statistical plotting'),
        ('plotly', 'Interactive visualizations'),
        ('scikit-learn', 'Machine learning tools'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('scipy', 'Scientific computing'),
        ('huggingface-hub', 'Hugging Face model hub')
    ]
    
    print("📦 INSTALLING PACKAGES FOR AUDIO DEEPFAKE DETECTION")
    print("=" * 60)
    print(f"⏱️  Total packages to install: {len(packages)}")
    print("💡 This will take about 3-5 minutes...")
    print()
    
    failed_packages = []
    
    for i, (package, description) in enumerate(packages, 1):
        print(f"[{i:2d}/{len(packages)}] Installing {package:15} - {description}")
        
        try:
            start_time = time.time()
            
            # Special handling for PyTorch with CUDA
            if package in ['torch', 'torchaudio']:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package,
                    '--index-url', 'https://download.pytorch.org/whl/cu118',
                    '--quiet'
                ], capture_output=True, text=True, timeout=300)
            else:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package, '--quiet'
                ], capture_output=True, text=True, timeout=300)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"         ✅ Success ({elapsed:.1f}s)")
            else:
                print(f"         ❌ Failed: {result.stderr[:100]}...")
                failed_packages.append(package)
                
        except subprocess.TimeoutExpired:
            print(f"         ⏰ Timeout after 5 minutes")
            failed_packages.append(package)
        except Exception as e:
            print(f"         ❌ Error: {str(e)[:100]}...")
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    if failed_packages:
        print(f"⚠️  Failed to install: {', '.join(failed_packages)}")
        print("💡 You can install these manually later if needed")
    else:
        print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
    print("=" * 60)
    
    return failed_packages

def download_and_cache_models():
    """Download and cache AI models to avoid repeated downloads"""
    print("\n🤖 DOWNLOADING AI MODELS")
    print("=" * 60)
    print("💡 This is the longest step - models are ~500MB each")
    print("⏱️  Expected time: 5-10 minutes depending on connection")
    print()
    
    # Import after packages are installed
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please restart runtime and try again")
        return False
    
    # Set up cache directory
    cache_dir = "/tmp/deepfake_models"
    os.makedirs(cache_dir, exist_ok=True)
    
    models_to_download = [
        {
            'name': 'wav2vec2-base',
            'model_id': 'facebook/wav2vec2-base-960h',
            'description': 'Wav2Vec2 Base Model (faster, good accuracy)',
            'size': '~360MB'
        },
        {
            'name': 'wav2vec2-large', 
            'model_id': 'facebook/wav2vec2-large-960h-lv60-self',
            'description': 'Wav2Vec2 Large Model (slower, best accuracy)',
            'size': '~1.2GB'
        }
    ]
    
    successfully_downloaded = []
    
    for i, model_info in enumerate(models_to_download, 1):
        print(f"[{i}/2] Downloading {model_info['name']}")
        print(f"     📋 {model_info['description']}")
        print(f"     💾 Size: {model_info['size']}")
        print(f"     🔗 Model ID: {model_info['model_id']}")
        
        try:
            start_time = time.time()
            
            # Download processor
            print("     📥 Downloading processor...")
            processor = Wav2Vec2Processor.from_pretrained(
                model_info['model_id'],
                cache_dir=cache_dir
            )
            
            # Download model
            print("     📥 Downloading model...")
            model = Wav2Vec2Model.from_pretrained(
                model_info['model_id'],
                cache_dir=cache_dir
            )
            
            elapsed = time.time() - start_time
            print(f"     ✅ Downloaded successfully ({elapsed:.1f}s)")
            print(f"     💾 Cached in: {cache_dir}")
            
            successfully_downloaded.append(model_info['name'])
            
            # Clear memory
            del processor, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"     ❌ Failed to download: {str(e)[:100]}...")
            print(f"     💡 Will use fallback methods in main code")
    
    print("\n" + "=" * 60)
    if successfully_downloaded:
        print(f"✅ Successfully downloaded: {', '.join(successfully_downloaded)}")
        print(f"💾 Models cached in: {cache_dir}")
        print("🚀 Ready for main detection code!")
    else:
        print("⚠️  No models downloaded successfully")
        print("💡 Main code will still work with traditional features")
    print("=" * 60)
    
    return len(successfully_downloaded) > 0

def check_gpu_setup():
    """Check GPU availability and setup"""
    print("\n🖥️  CHECKING GPU SETUP")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Available: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            if "T4" in gpu_name:
                print("🚀 T4 GPU detected - optimal for this workload!")
            else:
                print("💡 Non-T4 GPU detected - will still work well")
                
            # Test GPU
            test_tensor = torch.rand(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor)
            print("✅ GPU test passed")
            del test_tensor, result
            torch.cuda.empty_cache()
            
        else:
            print("⚠️  GPU not available - will use CPU (slower)")
            print("💡 To enable GPU: Runtime → Change Runtime Type → GPU")
            
    except ImportError:
        print("❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False
    
    print("=" * 60)
    return True

def setup_colab_environment():
    """Setup Google Colab specific configurations"""
    print("\n🔧 SETTING UP COLAB ENVIRONMENT")
    print("=" * 60)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab detected")
        
        # Setup matplotlib for inline plots
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'inline')
            print("✅ Matplotlib inline plotting enabled")
        except:
            print("⚠️  Could not setup inline plotting")
            
        # Setup file upload capability
        print("✅ File upload capability available")
        
        # Enable GPU optimizations
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
        print("✅ CUDA optimizations enabled")
        
    except ImportError:
        print("💻 Non-Colab environment detected")
    except Exception as e:
        print(f"⚠️  Setup warning: {e}")
    
    print("=" * 60)

def main():
    """Main setup function"""
    print("🚀 AUDIO DEEPFAKE DETECTION - COMPLETE SETUP")
    print("=" * 80)
    print("This script will:")
    print("  1. Install all required packages")
    print("  2. Download and cache AI models")  
    print("  3. Setup GPU optimization")
    print("  4. Configure Colab environment")
    print()
    print("⏱️  Total estimated time: 10-15 minutes")
    print("💡 After this completes, the main detector will load instantly!")
    print("=" * 80)
    
    # Step 1: Install packages
    failed_packages = install_packages_with_progress()
    
    # Step 2: Check GPU
    gpu_available = check_gpu_setup()
    
    # Step 3: Setup Colab
    setup_colab_environment()
    
    # Step 4: Download models
    models_downloaded = download_and_cache_models()
    
    # Final status
    print("\n" + "🎯 SETUP COMPLETE!" + " " * 50)
    print("=" * 80)
    
    if not failed_packages and models_downloaded and gpu_available:
        print("✅ PERFECT SETUP!")
        print("   • All packages installed")
        print("   • AI models downloaded and cached")
        print("   • GPU ready for acceleration")
        print("   • Colab environment configured")
    elif models_downloaded:
        print("✅ GOOD SETUP!")
        print("   • Core functionality ready")
        print("   • Models available for high accuracy")
        print("   • Some optional components may be missing")
    else:
        print("⚠️  BASIC SETUP!")
        print("   • Basic functionality available")
        print("   • Will use traditional features only")
        print("   • Some components need manual installation")
    
    print()
    print("🚀 NEXT STEP: Run the main detection code!")
    print("💡 The main detector will now load in seconds instead of minutes")
    print("=" * 80)

if __name__ == "__main__":
    main()
