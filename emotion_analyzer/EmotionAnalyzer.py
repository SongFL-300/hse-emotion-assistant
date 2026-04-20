import cv2
import numpy as np
import os
from emotion_analyzer.common import classify_affect, compute_valence_arousal, normalize_score_dict
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError:
    try:
        from keras.models import load_model
        from keras.preprocessing.image import img_to_array
    except ImportError:
        raise ImportError("Could not import Keras or Tensorflow. Please install them.")

class EmotionAnalyzer:
    """
    基于预训练 CNN (Mini-Xception) 的人脸情感分析器
    适用于实时视频流或静态图像
    """
    
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    
    def __init__(self, model_path=None):
        self.last_fallback_notice = None
        if model_path is None:
            # 默认尝试加载项目内的模型
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 原路径包含中文 "情感分析项目"，直接使用可能导致编码问题
            # 尝试使用相对路径或将路径转为 unicode 兼容格式
            model_path = os.path.join(base_path, 'Emotion-recognition', 'models', '_mini_XCEPTION.102-0.66.hdf5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Emotion model not found at: {model_path}")
            
        print(f"[INFO] Loading emotion model...") # 避免打印含中文的路径，防止控制台编码错误
        try:
            self.model = load_model(model_path, compile=False)
        except Exception as e:
            # 如果是编码错误，尝试复制模型到临时目录（纯英文路径）再加载
            if "codec" in str(e) or "decode" in str(e):
                print("[WARN] Loading from chinese path failed. Trying temporary copy...")
                import shutil
                import tempfile
                
                temp_dir = tempfile.gettempdir()
                temp_model_path = os.path.join(temp_dir, '_mini_XCEPTION.102-0.66.hdf5')
                shutil.copy2(model_path, temp_model_path)
                self.last_fallback_notice = "因为中文路径加载失败，Mini-Xception 已回退到临时英文路径副本加载。"
                
                self.model = load_model(temp_model_path, compile=False)
                # 可选：加载完删除临时文件
                # os.remove(temp_model_path)
            else:
                raise e

        # 预热模型
        # 预热模型
        self.model.predict(np.zeros((1, 64, 64, 1)))
        print("[INFO] Model loaded successfully.")

    def analyze(self, aligned_face_rgb):
        """
        分析对齐人脸的情感
        
        Args:
            aligned_face_rgb: (H, W, 3) RGB 图像，来自 FaceMeshDetector.align_face
            
        Returns:
            dict: {
                'emotion': str (最可能的情感),
                'probability': float (置信度),
                'all_scores': dict (所有情感的得分)
            }
        """
        if aligned_face_rgb is None:
            return None
            
        # 1. 预处理：转灰度
        gray = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2GRAY)
        
        # 2. 调整大小为 64x64 (模型输入要求)
        roi = cv2.resize(gray, (64, 64))
        
        # 3. 归一化 (0-1)
        roi = roi.astype("float") / 255.0
        
        # 4. 增加维度 (Batch, H, W, Channels)
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # 5. 推理
        preds = self.model.predict(roi, verbose=0)[0]

        # 不再做人为偏置放大。先保留原始输出，再用保守规则做 neutral/unknown 保护。
        raw_scores = {emotion: float(score) for emotion, score in zip(self.EMOTIONS, preds)}
        all_scores = normalize_score_dict(raw_scores)
        label, prob, uncertain = classify_affect(all_scores)
        valence, arousal = compute_valence_arousal(all_scores)
        
        return {
            'emotion': label,
            'probability': float(prob),
            'all_scores': all_scores,
            'input_roi': roi[0],
            'valence': float(valence),
            'arousal': float(arousal),
            'uncertain': bool(uncertain),
        }

    def draw_result(self, frame, result, position=(10, 30)):
        """在图像上绘制分析结果"""
        if result is None:
            return frame
            
        label = result['emotion']
        prob = result['probability']
        text = f"{label}: {prob:.2f}"
        
        # 绘制背景框以提高可读性
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (position[0]-5, position[1]-text_h-5), 
                      (position[0]+text_w+5, position[1]+5), (0,0,0), -1)
        
        # 绘制文本
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        return frame
