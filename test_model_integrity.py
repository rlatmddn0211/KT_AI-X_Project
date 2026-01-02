# test_model_integrity.py
import numpy as np
from pathlib import Path
from app.model_loader import create_model, load_plant_model_from_h5, CLASS_NAMES, IMG_SIZE, preprocess_image_from_base64, predict_species

MODEL_H5_PATH = "saved_model/mobilenet_plant_classifier.h5"  # h5 경로

def test_model():
    h5_path = Path(MODEL_H5_PATH)
    if not h5_path.exists():
        print(f"[ERROR] h5 파일이 존재하지 않습니다: {h5_path}")
        return

    # 1 모델 생성 + weights-only 로드
    try:
        model = load_plant_model_from_h5(MODEL_H5_PATH)
        print("[INFO] 모델 가중치 로드 성공")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return

    # 2 출력 레이어 확인
    output_units = model.output_shape[-1]
    if output_units != len(CLASS_NAMES):
        print(f"[WARN] 출력 유닛 수({output_units})와 CLASS_NAMES 길이({len(CLASS_NAMES)}) 불일치!")
    else:
        print(f"[INFO] 출력 유닛 수와 CLASS_NAMES 길이 일치 ✅")

    # 3 임의 테스트 이미지 (모든 값 0)로 예측
    dummy_image = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    try:
        result = predict_species(model, dummy_image)
        print("[INFO] 더미 이미지 예측 결과:", result)
    except Exception as e:
        print(f"[ERROR] 예측 테스트 실패: {e}")

if __name__ == "__main__":
    test_model()
