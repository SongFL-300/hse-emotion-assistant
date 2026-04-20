import cv2
import numpy as np
import time
from collections import deque
from emotion_analyzer.common import classify_affect, compute_valence_arousal, normalize_score_dict

class DeepFaceAnalyzer:
    """
    基于 DeepFace 库的情感分析器 (Wrapper)
    使用混合模式: 外部 FaceMeshDetector 进行对齐 -> DeepFace 进行纯推理
    集成时序平滑和平滑修正策略
    """
    
    def __init__(self, model_name='VGG-Face', emotion_model_name='Race', smooth_window=5): 
        self.last_fallback_notice = None
        # DeepFace 的 analyze 函数通常会自动加载模型
        # 这里我们主要是为了检查 deepface 是否安装
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
        except ImportError:
            raise ImportError("DeepFace is not installed. Please run: pip install deepface --no-deps")
            
        print("[INFO] DeepFace Analyzer initialized.")
        print("[INFO] Note: First run will download model weights (~500MB+), please wait...")
        
        # 预定义情感标签映射 (DeepFace 的输出转为我们统一的格式)
        # DeepFace return keys: 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        self.EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        # 时序平滑队列
        self.smooth_window = smooth_window
        self.history_scores = deque(maxlen=smooth_window)

    def analyze(self, aligned_face_rgb):
        """
        分析对齐人脸的情感
        
        Args:
            aligned_face_rgb: (H, W, 3) RGB 图像，来自 FaceMeshDetector.align_face
            
        Returns:
            dict: 标准化结果格式
        """
        if aligned_face_rgb is None:
            return None
            
        try:
            # 输入已经经过 FaceMesh 对齐，优先避免二次检测；失败时再回退到 opencv。
            img_bgr = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR)

            try:
                results = self.DeepFace.analyze(
                    img_path=img_bgr,
                    actions=['emotion'],
                    detector_backend='skip',
                    enforce_detection=False,
                    silent=True
                )
            except Exception:
                self.last_fallback_notice = "因为 DeepFace 的 skip 检测后端失败，已回退到 opencv 检测后端。"
                results = self.DeepFace.analyze(
                    img_path=img_bgr,
                    actions=['emotion'],
                    detector_backend='opencv',
                    enforce_detection=False,
                    silent=True
                )
            
            # DeepFace 返回的是一个 list (虽然我们只传了一张图)
            if not results:
                return None
                
            result = results[0] # 取第一个结果
            
            # 提取情感数据
            emotion_dict = result['emotion'] # e.g. {'angry': 0.02, 'happy': 99.9, ...}
            
            current_scores = np.array([emotion_dict.get(e, 0.0) for e in self.EMOTIONS]) / 100.0
            
            self.history_scores.append(current_scores)
            
            avg_scores = np.mean(self.history_scores, axis=0)
            all_scores_dict = normalize_score_dict(
                {k: float(v) for k, v in zip(self.EMOTIONS, avg_scores)}
            )
            dominant_emotion, probability, uncertain = classify_affect(all_scores_dict)
            valence, arousal = compute_valence_arousal(all_scores_dict)

            with open("emotion_debug.log", "a") as f:
                f.write(f"RAW: {emotion_dict}\n")
                f.write(f"OUT: {dominant_emotion} ({probability:.2f})\n")
                f.write("-" * 20 + "\n")
            
            return {
                'emotion': dominant_emotion,
                'probability': float(probability),
                'all_scores': all_scores_dict,
                'input_roi': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                'valence': float(valence),
                'arousal': float(arousal),
                'uncertain': bool(uncertain),
            }
            
        except Exception as e:
            print(f"[ERROR] DeepFace inference failed: {e}")
            return None

    def draw_result(self, frame, result, position=(10, 30)):
        """在图像上绘制分析结果"""
        if result is None:
            return frame
            
        label = result['emotion']
        prob = result['probability']
        text = f"{label}: {prob:.2f}"
        
        # 绘制背景框
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (position[0]-5, position[1]-text_h-5), 
                      (position[0]+text_w+5, position[1]+5), (0,0,0), -1)
        
        # 绘制文本 (DeepFace 用黄色区分)
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        return frame
