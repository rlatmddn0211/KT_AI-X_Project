# app/diagnose_service.py (교체할 함수)

import cv2, os, time
import numpy as np
from fastapi import HTTPException
from pathlib import Path
from app.model_loader import IMG_SIZE
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SAVE_CROPS = True
YOLO_DETECT_SAVE_DIR = Path("yolo_log_detect")

def _save_crop(species: str, img_np: np.ndarray):
    try:
        if not SAVE_CROPS:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest_dir = YOLO_DETECT_SAVE_DIR / f"{species}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        fn = dest_dir / f"crop_{ts}.jpg"
        cv2.imwrite(str(fn), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[WARN] crop save failed: {e}")

def preprocess_whole_image(img_bytes: bytes, img_size: int = IMG_SIZE):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 리사이즈 실패")
    img_batch = np.expand_dims(img_resized, axis=0).astype("float32")
    processed = preprocess_input(img_batch)
    return processed

def detect_leaf_and_crop_by_species(
    img_bgr: np.ndarray,
    species: str,
    yolo_model=None,
    conf_threshold: float = 0.18,
    iou_threshold: float = 0.7,
    fallback_yolo_model=None,
    try_thresholds: list | None = None
):
    """
    개선된 detect: 여러 threshold로 재시도하고, 모델.names(이 있으면) 출력.
    - fallback_yolo_model: species 전용 모델이 못 찾을 때 대체로 시도할 전역 YOLO (optional)
    - try_thresholds: 우선 시도할 threshold 리스트 (예: [0.3,0.15,0.1,0.05])
    반환: crop_rgb (RGB)
    """
    if yolo_model is None:
        raise HTTPException(status_code=400, detail="YOLO 모델이 전달되지 않았습니다.")

    if try_thresholds is None:
        try_thresholds = [conf_threshold, 0.15, 0.1]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape
    print(f"[DEBUG-DETECT] image shape: ({H},{W}) species={species}")

    try:
        names = None
        if hasattr(yolo_model, "model") and hasattr(yolo_model.model, "names"):
            names = yolo_model.model.names
        elif hasattr(yolo_model, "names"):
            names = yolo_model.names
        if names is not None:
            print(f"[DEBUG-DETECT] species YOLO names: {names}")
    except Exception as e:
        print(f"[DEBUG-DETECT] model.names 조회 중 예외: {e}")

    def _run_detect(model, conf,iou):
        try:
            results = model(img_rgb, conf=conf, iou=iou, verbose=False)
            return results
        except TypeError:
            try:
                results = model(img_rgb, verbose=False)
                return results
            except Exception as e:
                print(f"[DEBUG-DETECT] detect 호출 실패 (no-conf fallback) : {e}")
                raise
        except Exception as e:
            print(f"[DEBUG-DETECT] detect 호출 실패: {e}")
            raise

    results = None
    used_model = yolo_model
    used_conf = None
    for conf in try_thresholds:
        try:
            print(f"[DEBUG-DETECT] 시도: species YOLO conf={conf}")
            results = _run_detect(yolo_model, conf,iou_threshold)
            if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
                used_conf = conf
                print(f"[DEBUG-DETECT] species YOLO에서 박스 발견 conf={conf}")
                break
            else:
                print(f"[DEBUG-DETECT] species YOLO conf={conf} -> boxes 없음")
        except Exception as e:
            print(f"[DEBUG-DETECT] species YOLO detect 예외(conf={conf}): {e}")

    if (results is None or len(results) == 0 or len(getattr(results[0], "boxes", [])) == 0) and fallback_yolo_model is not None:
        print("[DEBUG-DETECT] species YOLO 실패 -> fallback_yolo_model로 재시도")
        try:
            for conf in try_thresholds:
                print(f"[DEBUG-DETECT] 시도: fallback YOLO conf={conf}")
                results = _run_detect(fallback_yolo_model, conf,iou_threshold)
                if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
                    used_model = fallback_yolo_model
                    used_conf = conf
                    print(f"[DEBUG-DETECT] fallback YOLO에서 박스 발견 conf={conf}")
                    break
                else:
                    print(f"[DEBUG-DETECT] fallback YOLO conf={conf} -> boxes 없음")
        except Exception as e:
            print(f"[DEBUG-DETECT] fallback YOLO detect 예외: {e}")

    if results is None or len(results) == 0 or len(getattr(results[0], "boxes", [])) == 0:
        print("[DEBUG-DETECT] 모든 시도에서 박스 발견 실패. 잎을 찾지 못했습니다.")
        raise HTTPException(status_code=400, detail="잎을 찾지 못했습니다.")

    r = results[0]
    boxes = r.boxes
    print(f"[DEBUG-DETECT] 사용 모델: {'fallback' if used_model is fallback_yolo_model else 'species'} conf={used_conf} boxes_count={len(boxes)}")

    best_box = None
    best_area = -1
    for b in boxes:
        try:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        except Exception:
            try:
                arr = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
            except Exception as e:
                print(f"[DEBUG-DETECT] box 좌표 읽기 실패: {e}")
                continue
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        print("[DEBUG-DETECT] best_box 없음 -> 잎 검출 실패")
        raise HTTPException(status_code=400, detail="잎을 찾지 못했습니다.")

    x1, y1, x2, y2 = best_box
    print(f"[DEBUG-DETECT] 선택된 best_box: ({x1},{y1},{x2},{y2}), area={best_area}")

    w = x2 - x1
    h = y2 - y1
    pad = 0.1
    x1p = max(0, int(x1 - w * pad))
    y1p = max(0, int(y1 - h * pad))
    x2p = min(W, int(x2 + w * pad))
    y2p = min(H, int(y2 + h * pad))
    print(f"[DEBUG-DETECT] padded coords: ({x1p},{y1p},{x2p},{y2p}), pad={pad}")

    crop_rgb = img_rgb[y1p:y2p, x1p:x2p]
    if crop_rgb.size == 0:
        print("[DEBUG-DETECT] crop_rgb.size == 0")
        raise HTTPException(status_code=400, detail="Crop된 이미지가 비어 있습니다.")

    _save_crop(species, crop_rgb)
    print(f"[DEBUG-DETECT] crop saved, shape={crop_rgb.shape}")
    return crop_rgb
