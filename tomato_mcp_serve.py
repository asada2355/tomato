import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from fastmcp import FastMCP
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# ============ 验证环境变量是否加载成功（测试用，可删除） ============
print("正在验证环境变量...")
print("TUYA_ACCESS_ID:", os.getenv("TUYA_ACCESS_ID")[:10] + "..." if os.getenv("TUYA_ACCESS_ID") else "未找到")
print("TUYA_ENDPOINT:", os.getenv("TUYA_ENDPOINT") if os.getenv("TUYA_ENDPOINT") else "未找到")

# ============ 初始化MCP服务（不传入任何认证参数） ============
mcp = FastMCP("番茄糖度预测专家")

# ============ 加载模型 ============
print("正在加载模型...")
model_package = joblib.load('best_model.pkl')

model = model_package['model']
scaler = model_package['scaler']
selected_features = model_package['features']
n_components = model_package['n_components']

print(f"模型加载成功！")
print(f"  主成分数: {n_components}")
print(f"  特征数量: {len(selected_features)}")
print(f"  训练样本: {model_package['performance']['n_samples']}")
print(f"  模型R²: {model_package['performance']['r2']:.4f}")


# ============ 定义预处理函数 ============
def preprocess_spectrum(raw_spectrum: list) -> np.ndarray:
    """对原始光谱数据进行预处理：SG平滑 + SNV标准化 + 特征选择"""
    X = np.array(raw_spectrum).reshape(1, -1)
    X_smooth = savgol_filter(X, window_length=9, polyorder=2, axis=1)
    X_snv = (X_smooth - X_smooth.mean(axis=1, keepdims=True)) / (X_smooth.std(axis=1, keepdims=True) + 1e-8)
    X_selected = X_snv[:, selected_features]
    return X_selected


# ============ 注册工具函数 ============
@mcp.tool()
def predict_tomato_brix(spectrum: list) -> dict:
    """
    根据近红外光谱数据预测番茄糖度（Brix值）。

    Args:
        spectrum: 包含228个波长点光谱强度的列表（900-1700nm范围）

    Returns:
        包含预测结果的字典
    """
    try:
        if len(spectrum) != 228:
            return {
                "status": "error",
                "message": f"输入数据长度应为228，实际为{len(spectrum)}"
            }

        X_processed = preprocess_spectrum(spectrum)
        X_scaled = scaler.transform(X_processed)
        prediction = model.predict(X_scaled)
        brix_value = float(prediction[0])

        # 品质分级
        if brix_value >= 9.0:
            grade, description = "优质", "糖度高，口感甜美，适合鲜食"
        elif brix_value >= 7.0:
            grade, description = "良好", "糖度适中，口感均衡"
        elif brix_value >= 5.0:
            grade, description = "一般", "糖度偏低，适合烹饪使用"
        else:
            grade, description = "偏低", "糖度较低，建议用于加工"

        return {
            "status": "success",
            "predicted_brix": round(brix_value, 2),
            "unit": "°Brix",
            "grade": grade,
            "description": description,
            "confidence": "高" if brix_value >= 6.0 else "中等"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"预测失败: {str(e)}"
        }


# ============ 可选：获取模型信息 ============
@mcp.tool()
def get_model_info() -> dict:
    """获取当前加载的模型信息"""
    perf = model_package['performance']
    return {
        "model_name": "番茄糖度PLS预测模型",
        "preprocessing": "SG平滑 + SNV标准化",
        "n_components": n_components,
        "n_features": len(selected_features),
        "training_samples": perf['n_samples'],
        "r2_score": perf['r2'],
        "rmse": perf['rmse']
    }


# ============ 启动服务 ============
if __name__ == "__main__":
    print("\n启动MCP服务...")
    print("服务名称: 番茄糖度预测专家")
    print("等待涂鸦平台连接...")
    mcp.run(transport="http", host="0.0.0.0", port=8000)