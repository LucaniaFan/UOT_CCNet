import torch
import platform
import sys

def check_device_compatibility():
    """检查系统设备兼容性并返回最佳可用设备"""
    
    print("系统信息：")
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查MPS可用性
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_available:
        print("\nMPS (Metal Performance Shaders) 可用")
        print("推荐使用 MPS 进行加速")
        device = torch.device("mps")
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("\nCUDA 可用")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"可用的GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda")
    
    if not (mps_available or cuda_available):
        print("\n没有可用的GPU加速设备，将使用CPU")
        device = torch.device("cpu")
    
    # 测试设备性能
    print("\n执行简单的性能测试...")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    # 预热
    for _ in range(10):
        _ = torch.matmul(x, y)
    
    # 计时
    start = torch.cuda.Event(enable_timing=True) if cuda_available else torch.tensor([])
    end = torch.cuda.Event(enable_timing=True) if cuda_available else torch.tensor([])
    
    if cuda_available:
        start.record()
    
    for _ in range(100):
        _ = torch.matmul(x, y)
    
    if cuda_available:
        end.record()
        torch.cuda.synchronize()
        print(f"矩阵乘法性能测试耗时: {start.elapsed_time(end)/100:.2f} ms/iter")
    
    return device

def test_model_on_device(model, device):
    """测试模型在指定设备上的运行情况"""
    try:
        model = model.to(device)
        print(f"模型已成功移至 {device} 设备")
        
        # 创建测试输入
        x = torch.randn(1, 3, 224, 224).to(device)
        
        # 测试前向传播
        with torch.no_grad():
            _ = model(x)
        print("模型前向传播测试成功")
        
        return True
    except Exception as e:
        print(f"模型测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    device = check_device_compatibility()
    print(f"\n推荐使用的设备: {device}")