"""
Quick script to test dataset loading
Run this to verify your dataset configuration works
"""
from datasets import load_dataset
import sys

def test_dataset(dataset_name, split='train', image_key='image'):
    """Test loading a dataset"""
    print(f"Testing dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Image key: {image_key}")
    print("="*60)

    try:
        # Try loading dataset
        print("\n1. Loading dataset...")
        ds = load_dataset(dataset_name, split=split, streaming=True)
        print("   ✓ Dataset loaded successfully (streaming mode)")

        # Get first sample
        print("\n2. Fetching first sample...")
        sample = next(iter(ds))
        print(f"   ✓ Sample keys: {list(sample.keys())}")

        # Check image
        print("\n3. Checking image...")
        if image_key in sample:
            img = sample[image_key]
            print(f"   ✓ Image found!")
            print(f"   - Image type: {type(img)}")
            if hasattr(img, 'size'):
                print(f"   - Image size: {img.size}")
            if hasattr(img, 'mode'):
                print(f"   - Image mode: {img.mode}")
        else:
            print(f"   ✗ Image key '{image_key}' not found!")
            print(f"   Available keys: {list(sample.keys())}")
            return False

        # Check for labels
        print("\n4. Checking for labels...")
        label_keys = ['label', 'fine_label', 'coarse_label', 'labels']
        found_labels = [k for k in label_keys if k in sample]
        if found_labels:
            print(f"   ✓ Label keys found: {found_labels}")
            for key in found_labels:
                print(f"     - {key}: {sample[key]}")
        else:
            print("   ℹ No standard label keys found (unsupervised dataset)")

        print("\n" + "="*60)
        print("✅ Dataset test PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n❌ Dataset test FAILED!")
        print(f"Error: {e}")
        print("="*60)
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Dataset Loading Test Script")
    print("="*60 + "\n")

    # Test main competition dataset
    print("\n" + "="*60)
    print("TEST 1: Main Competition Dataset")
    print("="*60)
    success1 = test_dataset(
        dataset_name='tsbpp/fall2025_deeplearning',
        split='train',
        image_key='image'
    )

    # Test CIFAR-10
    print("\n\n" + "="*60)
    print("TEST 2: CIFAR-10 (Debug/Sanity Check)")
    print("="*60)
    success2 = test_dataset(
        dataset_name='cifar10',
        split='train',
        image_key='img'
    )

    # Test CIFAR-100
    print("\n\n" + "="*60)
    print("TEST 3: CIFAR-100 (Full Baseline)")
    print("="*60)
    success3 = test_dataset(
        dataset_name='cifar100',
        split='train',
        image_key='img'
    )

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Main dataset (tsbpp/fall2025_deeplearning): {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"CIFAR-10: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"CIFAR-100: {'✅ PASS' if success3 else '❌ FAIL'}")
    print("="*60)

    if success1 and success2 and success3:
        print("\n🎉 All datasets are ready to use!")
        sys.exit(0)
    else:
        print("\n⚠️  Some datasets failed. Check errors above.")
        sys.exit(1)

