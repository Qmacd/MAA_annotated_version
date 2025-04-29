# Time : 2025/4/29 16:16
# Tong ji Marcus
# FileName: UI_construction.py

from flask import Flask, request, jsonify
import joblib  # 或者使用TensorFlow/Keras/PyTorch等库

app = Flask(__name__)

# 加载已训练的模型
model = joblib.load('your_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # 获取输入数据
    input_data = data['input']  # 假设输入是一个列表
    prediction = model.predict([input_data])  # 进行预测
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
