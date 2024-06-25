import torch
import tensorflow as tf
import onnxruntime as ort

def test_pytorch_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("PyTorch is using GPU!")
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("PyTorch is using CPU.")

def test_tensorflow_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("TensorFlow is using GPU!")
        print("GPU Name:", tf.test.gpu_device_name())
    else:
        print("TensorFlow is using CPU.")

def test_onnx_gpu():
    available_providers = ort.get_device()
    if 'GPU' in available_providers:
        return "GPU is available and can be used"
    else:
        return "GPU is not available or cannot be used"

if __name__ == "__main__":
    print("Testing PyTorch CUDA capability...")
    test_pytorch_cuda()

    print("\nTesting TensorFlow GPU capability...")
    test_tensorflow_gpu()

    print("\nTesting ONNX GPU capability...")
    print(test_onnx_gpu())