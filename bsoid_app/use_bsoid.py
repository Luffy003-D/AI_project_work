# 创建一个使用示例 use_bsoid.py
try:
    from bsoid import BSOID
    
    # 初始化B-SOID
    bsoid_model = BSOID()
    
    # 如果有数据，可以尝试处理
    print("B-SOID 初始化成功")
    print("下一步需要准备小鼠姿态数据...")
    
except ImportError:
    print("无法导入B-SOID，尝试替代方法...")
    
    # 尝试其他导入方式
    try:
        import bsoid_py
        print("找到 bsoid_py 模块")
    except:
        print("需要检查项目结构并正确安装")