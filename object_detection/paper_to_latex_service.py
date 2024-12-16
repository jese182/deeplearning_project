import gradio as gr
from ultralytics import YOLO
from pix2tex.cli import LatexOCR
import cv2
import numpy as np
from PIL import Image
import base64
import io
import logging
import datetime
import os
from pathlib import Path
import json
import uuid
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# 설정 관련 상수들
CONFIDENCE_THRESHOLD = 0.5
IMAGE_TARGET_SIZE = (768, 1024)
MAX_WORKERS = 3

@dataclass
class ProcessingConfig:
    """처리 관련 설정을 담는 데이터 클래스"""
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    target_size: Tuple[int, int] = IMAGE_TARGET_SIZE
    log_dir: Path = Path("./logs")  # 상대 경로로 수정
    image_dir: Path = Path("./uploaded_images")  # 상대 경로로 수정

class LoggerSetup:
    """로깅 설정을 관리하는 클래스"""
    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
            
        # 파일 핸들러
        file_handler = logging.FileHandler(
            ProcessingConfig.log_dir / f'app_{datetime.datetime.now().strftime("%Y%m%d")}.log',
            mode='a',
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # 스트림 핸들러
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False
        
        return logger

class ImageProcessor:
    """이미지 처리를 담당하는 클래스"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.yolo_model = YOLO('best.pt')
        self.ocr_model = LatexOCR()
        self.logger = LoggerSetup.setup_logger(__name__)
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """이미지 전처리"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, self.config.target_size)

        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

        return resized_image, (x, y, w, h)
    
    def convert_coordinates(self, 
                          box: List[float], 
                          crop_info: Tuple[int, int, int, int], 
                          original_size: Tuple[int, int]) -> List[int]:
        """좌표 변환"""
        x, y, crop_w, crop_h = crop_info
        target_w, target_h = self.config.target_size
        
        x1, y1, x2, y2 = box
        x1 = (x1 * crop_w) / target_w + x
        x2 = (x2 * crop_w) / target_w + x
        y1 = (y1 * crop_h) / target_h + y
        y2 = (y2 * crop_h) / target_h + y
        
        return list(map(int, [x1, y1, x2, y2]))

    def process_equation(self, 
                        equation_data: Tuple[np.ndarray, List[int], float]) -> Dict:
        """개별 수식 처리"""
        equation_image, coords, conf = equation_data
        x1, y1, x2, y2 = coords
        
        pil_equation = Image.fromarray(cv2.cvtColor(equation_image, cv2.COLOR_BGR2RGB))
        latex = self.ocr_model(pil_equation)
        
        return {
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf),
            'latex': latex
        }

    def process_image(self, 
                     input_image: Union[np.ndarray, Image.Image], 
                     session_id: str,
                     client_ip: str) -> Tuple[Image.Image, str]:
        """이미지 처리 메인 함수"""
        try:
            # 이미지 저장
            image_path = self.save_uploaded_image(input_image, session_id)
            self.logger.info(f"Image saved: {image_path}, Client IP: {client_ip}")
            
            # PIL 이미지를 CV2 형식으로 변환
            cv2_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
            
            # 이미지 전처리
            processed_image, crop_info = self.preprocess_image(cv2_image)
            original_size = cv2_image.shape[:2]
            
            # YOLO로 수식 탐지
            results = self.yolo_model.predict(processed_image)
            
            # 결과 이미지 준비
            result_image = cv2_image.copy()
            equation_data = []
            
            # 수식 검출 및 데이터 준비
            for r in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = r
                if conf > self.config.confidence_threshold:
                    coords = self.convert_coordinates([x1, y1, x2, y2], crop_info, original_size)
                    x1, y1, x2, y2 = coords
                    
                    equation_image = cv2_image[y1:y2, x1:x2]
                    equation_data.append((equation_image, coords, conf))
                    
                    # 바운딩 박스 표시
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_image, f"{conf:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 병렬 처리로 수식 인식
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                detected_equations = list(executor.map(self.process_equation, equation_data))
            
            # 결과 로깅
            self.log_processing_results(session_id, image_path, detected_equations, client_ip)
            self.logger.info(
                f"Processing completed - Session ID: {session_id}, "
                f"IP: {client_ip}, Found {len(detected_equations)} equations"
            )
            
            # 결과 이미지 변환 및 HTML 생성
            result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            html_output = self.create_html_output(detected_equations, cv2_image)
            
            return result_image, html_output
            
        except Exception as e:
            self.logger.error(f"Error in processing - Session ID: {session_id}, Error: {str(e)}")
            raise gr.Error(f"처리 중 오류가 발생했습니다: {str(e)}")

    def save_uploaded_image(self, image: Union[np.ndarray, Image.Image], session_id: str) -> str:
        """업로드된 이미지 저장"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{timestamp}.png"
        image_path = self.config.image_dir / filename
        
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(image_path)
        else:
            image.save(image_path)
        
        return str(image_path)

    def log_processing_results(self, 
                             session_id: str, 
                             image_path: str, 
                             equations: List[Dict], 
                             client_ip: str):
        """처리 결과 로깅"""
        result_log = {
            "session_id": session_id,
            "client_ip": client_ip,
            "timestamp": datetime.datetime.now().isoformat(),
            "image_path": str(image_path),
            "num_equations": len(equations),
            "equations": [
                {
                    "confidence": float(eq['confidence']),
                    "latex": eq['latex']
                }
                for eq in equations
            ]
        }
        
        log_path = self.config.log_dir / f"results_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(result_log)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving results to JSON: {e}")

    @staticmethod
    def image_to_base64(image_array: np.ndarray) -> str:
        """이미지를 base64 문자열로 변환"""
        img = Image.fromarray(image_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def create_html_output(self, equations: List[Dict], original_image: np.ndarray) -> str:
        """HTML 출력 생성"""
        html = """
        <style>
        .equation-container {
            margin: 20px 0;
            padding: 20px;
            border: 2px solid #3b82f6;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .equation-title {
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 15px;
            color: white;
            padding: 8px 15px;
            background-color: #2563eb;
            border-radius: 6px;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            letter-spacing: 0.5px;
        }
        .equation-image {
            margin: 15px 0;
            padding: 15px;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
        }
        .equation-image img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        .latex-code {
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            background-color: #1e293b;
            color: #ffffff;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            white-space: pre-wrap;
            word-break: break-all;
            border: 1px solid #64748b;
        }
        .confidence-score {
            font-weight: bold;
            color: #fbbf24;
        }
        </style>
        """
        
        for i, eq in enumerate(equations, 1):
            x1, y1, x2, y2 = eq['bbox']
            equation_image = original_image[y1:y2, x1:x2]
            equation_image_rgb = cv2.cvtColor(equation_image, cv2.COLOR_BGR2RGB)
            img_base64 = self.image_to_base64(equation_image_rgb)
            
            html += f"""
            <div class="equation-container">
                <div class="equation-title">
                    Equation {i} <span class="confidence-score">(Confidence: {eq['confidence']:.2f})</span>
                </div>
                <div class="equation-image">
                    <img src="data:image/png;base64,{img_base64}" alt="Equation {i}"/>
                </div>
                <div class="latex-code">{eq['latex']}</div>
            </div>
            """
        
        return html

def main():
    # 설정 초기화
    config = ProcessingConfig()
    
    # 절대 경로로 변환
    config.log_dir = Path.cwd() / "logs"
    config.image_dir = Path.cwd() / "uploaded_images"
    
    # 디렉토리 생성
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.image_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 프로세서 초기화
    processor = ImageProcessor(config)
    
    # ... 나머지 코드는 동일
    
    def process_wrapper(input_image, request: gr.Request):
        session_id = str(uuid.uuid4())
        client_ip = request.headers.get('X-Forwarded-For', 
                                      request.headers.get('Remote-Addr', 'unknown'))
        return processor.process_image(input_image, session_id, client_ip)
    
    # Gradio 인터페이스 생성
    iface = gr.Interface(
        fn=process_wrapper,
        inputs=gr.Image(label="Upload Paper Image", type="pil"),
        outputs=[
            gr.Image(label="Detected Equations"),
            gr.HTML(label="Equation Results")
        ],
        title="Math Equation Detector and LaTeX Converter",
        description="Upload an image containing mathematical equations. "
                   "The system will detect equations and convert them to LaTeX format."
    )
    
    iface.launch(share=True)

if __name__ == "__main__":
    main()