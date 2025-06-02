import torch
import platform
import sys
import numpy as np
import time

def check_m3_compatibility():
    """检查M3芯片兼容性和性能"""
    print("=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.machine()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Numpy版本: {np.__version__}")
    
    print("\n=== MPS (Metal Performance Shaders) 检查 ===")
    print(f"MPS是否可用: {torch.backends.mps.is_available()}")
    print(f"MPS是否内置: {torch.backends.mps.is_built()}")
    
    # 性能测试
    print("\n=== 性能测试 ===")
    
    # 测试设备列表
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    # 矩阵运算测试
    size = 1000
    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n在 {device_name} 上进行测试:")
        
        # 创建测试张量
        try:
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)
            
            # 预热
            for _ in range(5):
                _ = torch.matmul(x, y)
            
            # 计时测试
            start_time = time.time()
            for _ in range(10):
                _ = torch.matmul(x, y)
            end_time = time.time()
            
            print(f"矩阵乘法 ({size}x{size}) 平均时间: {(end_time-start_time)/10*1000:.2f} ms")
            
        except Exception as e:
            print(f"测试失败: {str(e)}")
    
    print("\n=== 内存测试 ===")
    try:
        # 测试不同大小的张量分配
        sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
        for s in sizes:
            torch.empty(s, device='mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"成功分配 {s[0]}x{s[1]} 张量")
    except Exception as e:
        print(f"内存测试失败: {str(e)}")
    
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def to_device(tensor, device):
    try:
        return tensor.to(device)
    except Exception as e:
        print(f"警告：无法将张量移动到 {device}，将使用CPU: {str(e)}")
        return tensor.to('cpu')

# 示例：处理不支持的操作
def process_tensor(tensor, device):
    try:
        # 尝试在MPS上运行
        result = some_operation(tensor)
    except RuntimeError:
        # 如果失败，转到CPU执行
        cpu_tensor = tensor.to('cpu')
        result = some_operation(cpu_tensor)
        result = result.to(device)
    return result

if __name__ == "__main__":
    device = check_m3_compatibility()
    print(f"\n推荐使用的设备: {device}")
    
    if device.type == 'mps':
        print("\n提示：")
        print("1. 使用MPS后端时，某些PyTorch操作可能尚未实现")
        print("2. 如果遇到操作不支持的错误，可以临时将特定操作转移到CPU上执行")
        print("3. 建议定期更新PyTorch以获得更好的MPS支持")