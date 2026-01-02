# app/model_loader.py
import os
from pathlib import Path
import traceback

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
from ultralytics import YOLO
# ultralytics YOLO (환경에 설치되어 있어야 함)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

CLASS_NAMES = [
    "관음죽", "금전수", "디펜바키아","몬스테라","벵갈고무나무","보스턴고사리",'부레옥잠',
    '선인장', '스투키', '스파티필럼', '오렌지쟈스민', '올리브나무', '테이블야자', '호접란', '홍콩야자'
]

IMG_SIZE = 299
CONFIDENCE_THRESHOLD = 0.05


# ---------- helper: savedmodel signature/shape 추론 ----------
def _infer_savedmodel_input_shape(model_path: str):
    try:
        loaded = tf.saved_model.load(model_path)
    except Exception as e:
        print(f"[WARN] tf.saved_model.load 실패: {e}")
        return None

    candidates = []
    signatures = getattr(loaded, "signatures", None)
    if isinstance(signatures, dict):
        for k, fn in signatures.items():
            try:
                sig = fn.structured_input_signature
                _, kw = sig
                if isinstance(kw, dict):
                    for name, spec in kw.items():
                        try:
                            shape = spec.shape.as_list()
                        except Exception:
                            try:
                                shape = list(spec.shape)
                            except Exception:
                                shape = None
                        if shape and len(shape) == 4:
                            candidates.append((k, name, shape))
            except Exception:
                continue

    try:
        for attr_name in dir(loaded):
            attr = getattr(loaded, attr_name)
            if hasattr(attr, "structured_input_signature"):
                try:
                    sig = attr.structured_input_signature
                    _, kw = sig
                    if isinstance(kw, dict):
                        for name, spec in kw.items():
                            try:
                                shape = spec.shape.as_list()
                            except Exception:
                                try:
                                    shape = list(spec.shape)
                                except Exception:
                                    shape = None
                            if shape and len(shape) == 4:
                                candidates.append((attr_name, name, shape))
                except Exception:
                    continue
    except Exception:
        pass

    for cand in candidates:
        _, _, shape = cand
        if len(shape) == 4:
            if shape[1] is not None and shape[2] is not None:
                H, W, C = shape[1], shape[2], shape[3]
                if all(isinstance(x, int) and x > 0 for x in (H, W, C)):
                    return (int(H), int(W), int(C))

    for cand in candidates:
        _, _, shape = cand
        if len(shape) == 4:
            maybe_H = shape[1] if shape[1] is not None else (shape[2] if shape[2] is not None else None)
            maybe_W = shape[2] if shape[2] is not None else (shape[1] if shape[1] is not None else None)
            C = shape[3] if shape[3] is not None else 3
            if maybe_H and maybe_W:
                return (int(maybe_H), int(maybe_W), int(C))
    return None


# ---------- load_plant_model: TFSMLayer / concrete_fn / loaded.__call__ 등 가능한 것 전부 시도 ----------
def load_plant_model(model_path: str):
    if not tf.io.gfile.isdir(model_path) and not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 경로가 폴더가 아니거나 존재하지 않음: {model_path}")

    print(f"[INFO] 분류 모델 로드 시도: {model_path}")

    # 1) 파일(.h5) 먼저 시도
    if os.path.isfile(model_path):
        try:
            m = load_model(model_path, compile=False)
            try:
                ishape = m.input_shape
                if ishape and len(ishape) == 4:
                    _, H, W, C = ishape
                    m._expected_input_shape = (int(H), int(W), int(C))
                else:
                    m._expected_input_shape = None
            except Exception:
                m._expected_input_shape = None
            # Keras model은 predict 메소드가 있으므로 _callable_fn는 None으로 두어도 됨
            m._callable_fn = None
            return m
        except Exception as e:
            print(f"[WARN] keras load_model 실패: {e}")
            # proceed to SavedModel attempts

    # 2) SavedModel 로드 시도
    try:
        loaded = tf.saved_model.load(model_path)
    except Exception as e:
        print(f"[FATAL] tf.saved_model.load 실패: {e}")
        raise

    # try signatures -> serving_default
    signatures = getattr(loaded, "signatures", None)
    concrete_fn = None
    if isinstance(signatures, dict) and "serving_default" in signatures:
        concrete_fn = signatures["serving_default"]
    elif isinstance(signatures, dict) and len(signatures) > 0:
        # pick the first signature as candidate
        concrete_fn = next(iter(signatures.values()))

    # if no signatures, try to find any attr with structured_input_signature
    if concrete_fn is None:
        for name in dir(loaded):
            attr = getattr(loaded, name)
            if hasattr(attr, "structured_input_signature"):
                concrete_fn = attr
                break

    # infer input shape
    inferred_shape = None
    if concrete_fn is not None:
        try:
            sig = concrete_fn.structured_input_signature
            _, kw = sig
            if isinstance(kw, dict) and len(kw) > 0:
                for nm, spec in kw.items():
                    try:
                        shape = spec.shape.as_list()
                    except Exception:
                        try:
                            shape = list(spec.shape)
                        except Exception:
                            shape = None
                    if shape and len(shape) == 4:
                        H = shape[1]
                        W = shape[2]
                        C = shape[3] if len(shape) > 3 else 3
                        if H is not None and W is not None:
                            inferred_shape = (int(H), int(W), int(C))
                            break
        except Exception:
            inferred_shape = None

    if inferred_shape is None:
        inferred_shape = _infer_savedmodel_input_shape(model_path)

    if inferred_shape is None:
        print("[WARN] SavedModel에서 input signature를 못 찾았습니다. _expected_input_shape은 None으로 설정됩니다.")
    else:
        print(f"[INFO] Detected SavedModel input shape -> {inferred_shape}")

    # 3) TFSMLayer 시도 (원래 방식)
    tfsm_layer = None
    try:
        tfsm_layer = TFSMLayer(model_path, call_endpoint="serving_default")
    except Exception as e:
        # TFSMLayer 생성 실패시 로그만 남김, 계속 진행
        print(f"[WARN] TFSMLayer 생성 실패: {e}")
        tfsm_layer = None

    if tfsm_layer is not None:
        # build Input according to inferred shape (fallback IMG_SIZE)
        input_h, input_w, input_c = (inferred_shape if inferred_shape is not None else (IMG_SIZE, IMG_SIZE, 3))
        inp = Input(shape=(input_h, input_w, input_c), name="input_layer_1")
        try:
            out = tfsm_layer(inp)
            model = Model(inputs=inp, outputs=out)
            model._expected_input_shape = (int(input_h), int(input_w), int(input_c))
            model._callable_fn = None  # Keras-like predict exists
            print("[INFO] SavedModel -> TFSMLayer 래핑 성공")
            return model
        except Exception as e:
            print(f"[WARN] TFSMLayer 래핑 실패: {e}")
            # 계속

    # 4) concrete_fn이 있으면 그것으로 wrapper 생성
    if concrete_fn is not None:
        # try to extract input names
        input_names = []
        try:
            _, kw = concrete_fn.structured_input_signature
            if isinstance(kw, dict):
                input_names = list(kw.keys())
        except Exception:
            input_names = []

        class SavedModelWrapper:
            def __init__(self, fn, inferred_shape, input_names):
                self._fn = fn
                self._expected_input_shape = tuple(inferred_shape) if inferred_shape is not None else None
                self._input_names = input_names

            def predict(self, x, verbose=0):
                # x: numpy array batch
                xt = tf.constant(x)
                last_err = None
                # try keyword with input names
                for name in self._input_names:
                    try:
                        res = self._fn(**{name: xt})
                        if isinstance(res, dict):
                            return list(res.values())[0].numpy()
                        else:
                            return res.numpy()
                    except Exception as e:
                        last_err = e
                        continue
                # try positional
                try:
                    res = self._fn(xt)
                    if isinstance(res, dict):
                        return list(res.values())[0].numpy()
                    else:
                        return res.numpy()
                except Exception as e:
                    raise RuntimeError("SavedModelWrapper: 모든 호출 방식 실패: " + str(e)) from last_err

        wrapper = SavedModelWrapper(concrete_fn, inferred_shape, input_names)
        wrapper._callable_fn = None
        print("[INFO] SavedModel -> WrapperModel(ConcreteFunction) 준비")
        return wrapper

    # 5) 마지막 시도: loaded 객체 자체가 callable (일부 경우 가능)
    try:
        if callable(loaded):
            class DirectWrapper:
                def __init__(self, mod, inferred_shape):
                    self._mod = mod
                    self._expected_input_shape = tuple(inferred_shape) if inferred_shape is not None else None
                def predict(self, x, verbose=0):
                    xt = tf.constant(x)
                    try:
                        res = self._mod(xt)
                        if isinstance(res, dict):
                            return list(res.values())[0].numpy()
                        else:
                            return res.numpy()
                    except Exception as e:
                        raise RuntimeError("Direct loaded(...) 호출 실패: " + str(e))
            dw = DirectWrapper(loaded, inferred_shape)
            print("[INFO] loaded 모듈 자체가 callable하여 DirectWrapper 준비")
            return dw
    except Exception:
        pass

    # 실패
    raise RuntimeError("SavedModelWrapper: 내부 호출 가능한 serving function이나 call 가능 멤버를 찾을 수 없습니다. " +
                       "모델에 'serving_default' signature가 없거나 호출 방식이 특이합니다.")


# ---------- YOLO 로드 ----------
def load_yolo_model(pt_path: str):
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"YOLO 파일이 존재하지 않음: {pt_path}")
    if YOLO is None:
        raise RuntimeError("ultralytics YOLO 라이브러리가 설치되어 있지 않습니다.")
    print(f"[INFO] YOLO 모델 로드 중: {pt_path}")
    return YOLO(str(pt_path))


# ---------- 전처리 (target_model의 expected shape 우선 사용) ----------
def preprocess_image_pipeline(img_bytes: bytes, yolo_model, img_size: int = IMG_SIZE, target_model=None):
    if target_model is not None:
        try:
            exp = getattr(target_model, "_expected_input_shape", None)
            if exp is not None and isinstance(exp, (tuple, list)) and len(exp) >= 2:
                img_size = int(exp[0])
        except Exception:
            pass

    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("이미지 디코딩 실패")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    results = yolo_model(img_rgb, verbose=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        w, h = x2 - x1, y2 - y1
        pad = 0.1
        x1p, y1p = max(0, int(x1 - w*pad)), max(0, int(y1 - h*pad))
        x2p, y2p = min(w_img, int(x2 + w*pad)), min(h_img, int(y2 + h*pad))
        plant_img = img_rgb[y1p:y2p, x1p:x2p]
        print(f"[INFO] 식물 탐지 성공! 좌표: {x1p},{y1p},{x2p},{y2p}")
    else:
        plant_img = img_rgb
        print("[WARN] 식물 탐지 실패, 전체 이미지 사용")

    img_resized = cv2.resize(plant_img, (img_size, img_size))
    img_batch = np.expand_dims(img_resized, axis=0)
    processed_image = preprocess_input(img_batch)
    return processed_image


def preprocess_image_from_bytes(img_bytes: bytes, yolo_model=None, img_size: int = IMG_SIZE, target_model=None):
    if yolo_model is None:
        raise ValueError("YOLO 모델 인스턴스 필요")

    # target_model의 expected_input_shape 확인
    if target_model is not None:
        exp = getattr(target_model, "_expected_input_shape", None)
        if exp is not None and len(exp) == 3:
            img_size = exp[0]  # H == W 가 동일하다고 가정

    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("이미지 디코딩 실패")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    # YOLO로 식물 영역 crop
    results = yolo_model(img_rgb, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        w, h = x2 - x1, y2 - y1
        pad = 0.1
        x1p, y1p = max(0, int(x1 - w*pad)), max(0, int(y1 - h*pad))
        x2p, y2p = min(w_img, int(x2 + w*pad)), min(h_img, int(y2 + h*pad))
        plant_img = img_rgb[y1p:y2p, x1p:x2p]
        print(f"[INFO] 식물 탐지 성공! 좌표: {x1p},{y1p},{x2p},{y2p}")
    else:
        plant_img = img_rgb
        print("[WARN] 식물 탐지 실패, 전체 이미지 사용")

    img_resized = cv2.resize(plant_img, (img_size, img_size))
    img_batch = np.expand_dims(img_resized, axis=0)
    processed_image = preprocess_input(img_batch)
    return processed_image


# ---------- predict_species: 다양한 model 타입 지원 ----------
def predict_species(model, processed_image) -> dict:
    if processed_image is None:
        raise ValueError("processed_image가 None입니다.")

    # Try Keras-style predict
    preds = None
    last_err = None
    try:
        if hasattr(model, "predict"):
            preds = model.predict(processed_image, verbose=0)
            # print debug
            print("[DEBUG] model.predict 사용 (hasattr predict)")
    except Exception as e:
        last_err = e
        print(f"[WARN] model.predict 호출 실패: {e}")

    # If not obtained, try wrapper _callable_fn or model as callable
    if preds is None:
        # If wrapper stored a concrete function, try it
        try:
            # model might be a wrapper with _fn or be callable
            if hasattr(model, "_fn"):
                fn = getattr(model, "_fn")
                try:
                    res = fn(processed_image)
                    preds = res
                except Exception:
                    # try with tf.constant
                    res = fn(tf.constant(processed_image))
                    preds = res
            elif hasattr(model, "_callable_fn") and model._callable_fn is not None:
                fn = model._callable_fn
                res = fn(processed_image)
                preds = res
            elif callable(model):
                # some wrappers return numpy if passed np array, some expect tf.Tensor
                try:
                    res = model(processed_image)
                    preds = res
                except Exception:
                    res = model(tf.constant(processed_image))
                    preds = res
            else:
                raise RuntimeError("모델에 대해 시도할 수 있는 호출 방식이 없습니다.")
        except Exception as e:
            print(f"[ERROR] 대체 호출 방식 실패: {e}")
            # raise the original failure to surface the root cause
            raise RuntimeError(f"model.predict 실패: {last_err or e}") from (last_err or e)

    # Normalize preds to probability vector
    if isinstance(preds, dict):
        key = list(preds.keys())[0]
        probs = preds[key]
    else:
        # preds could be tf.Tensor or numpy array; if shape is (1, n) or (batch, n) choose first row
        if isinstance(preds, tf.Tensor):
            preds = preds.numpy()
        preds = np.asarray(preds)
        if preds.ndim == 1:
            probs = preds
        else:
            probs = preds[0]

    probs = np.asarray(probs).flatten()

    top5_idx = probs.argsort()[-5:][::-1]
    top5 = [(CLASS_NAMES[int(i)], float(probs[int(i)])) for i in top5_idx]
    print(f"[DEBUG] Top-5 예측: {top5}")

    predicted_index = int(top5_idx[0])
    species_name = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index])

    # Top-2 우선 logic (optional)
    target_class = "호접란"
    top2_idx = top5_idx[:1]
    forced_selection = False
    for i in top2_idx:
        cls_name = CLASS_NAMES[int(i)]
        if cls_name == target_class:
            species_name = target_class
            confidence = float(probs[int(i)])
            predicted_index = int(i)
            forced_selection = True
            print(f"[DEBUG] Top-2 안에 '{target_class}' 발견, 우선 선택")
            break

    if not forced_selection and confidence < CONFIDENCE_THRESHOLD:
        species_name = "unknown species"
        print(f"[DEBUG] confidence {confidence:.4f} < {CONFIDENCE_THRESHOLD}, unknown 처리")

    return {"species": species_name, "confidence": round(confidence, 4), "index": predicted_index}


# ---------- 진단 모델 로드/진단 ----------
# 기존 load_plant_model에서 TFSMLayer 관련 제거 후, SavedModelWrapper만 사용
def load_diagnosis_model(model_path: str):
    if not tf.io.gfile.exists(model_path):
        raise FileNotFoundError(f"모델 경로가 존재하지 않음: {model_path}")

    print(f"[INFO] 진단 모델 로드 시도: {model_path}")

    # SavedModel 로드
    loaded = tf.saved_model.load(model_path)

    # serving_default ConcreteFunction 강제 사용
    if 'serving_default' not in loaded.signatures:
        raise RuntimeError(f"진단 모델 로드 실패: {model_path} 내부에 'serving_default'가 없음")
    concrete_fn = loaded.signatures['serving_default']

    # 입력/출력 이름 명시
    input_name = 'input_layer'   # CLI에서 확인된 입력 이름
    output_name = 'output_0'     # CLI에서 확인된 출력 이름

    # 입력 shape 추출
    input_shape = (224, 224, 3)  # CLI에서 확인됨

    # wrapper 생성
    class SavedModelWrapper:
        def __init__(self, fn, input_shape):
            self._fn = fn
            self._expected_input_shape = input_shape

        def predict(self, x, verbose=0):
            xt = tf.constant(x, dtype=tf.float32)
            try:
                res = self._fn(**{input_name: xt})
                return res[output_name].numpy()
            except Exception as e:
                raise RuntimeError(f"SavedModelWrapper 호출 실패: {e}")

    wrapper = SavedModelWrapper(concrete_fn, input_shape)
    print(f"[INFO] 진단 모델 wrapper 준비, expected_input_shape={input_shape}")
    return wrapper


def diagnose_state(diagnosis_model, processed_image) -> dict:
    if diagnosis_model is None:
        raise ValueError("diagnosis_model이 없습니다.")

    # --- 예측 실행 + 디버그 로그 + traceback 추가 ---
    try:
        if hasattr(diagnosis_model, "predict"):
            preds = diagnosis_model.predict(processed_image, verbose=0)
        else:
            serving_fn = diagnosis_model.signatures["serving_default"]
            input_name = list(serving_fn.structured_input_signature[1].keys())[0]
            preds = serving_fn(**{input_name: tf.constant(processed_image)})
    except Exception as e:
        print("[DEBUG-DIAG] diagnosis_model 호출 실패, 예외 trace:")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"diagnosis_model.predict 실패: {e}")

    # --- raw preds 디버그 출력 ---
    print(f"[DEBUG-DIAG] raw preds type={type(preds)} shape(if array)={getattr(preds, 'shape', 'N/A')}")

    # --- 예측 결과 후처리 ---
    if isinstance(preds, dict):
        preds = list(preds.values())[0]

    if isinstance(preds, tf.Tensor):
        preds = preds.numpy()

    probs = np.asarray(preds).squeeze()
    if probs.ndim != 1:
        probs = probs.flatten()

    DIAG_STATUSES = ["DRY", "APPROPRIATE", "OVERWATERED"]
    top_idx = int(np.argmax(probs))
    status = DIAG_STATUSES[top_idx] if top_idx < len(DIAG_STATUSES) else "UNKNOWN"
    confidence = float(probs[top_idx])

    return {
        "status": status,
        "confidence": round(confidence, 4)
    }



YOLO_SEARCH_ROOTS = [Path("yolo_log"), Path("saved_model") / "yolo_log"]

def find_yolo_path_for_species(species: str) -> str | None:
    """
    species 이름이 포함된 yolo 로그 디렉토리에서 weights/best.pt 를 찾아 반환.
    우선순위:
      1) 경로에 'stage2' 또는 'stage2_' 포함된 파일 우선
      2) 그 다음 'stage1'
      3) 그 외 발견된 것 중 첫 번째
      4) fallback saved_model/best1.pt
    """
    candidates = []
    for root in YOLO_SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.iterdir():
            try:
                if species in p.name:
                    # 직접 weights/best.pt 경로 체크
                    w1 = p.joinpath("weights", "best.pt")
                    if w1.exists():
                        candidates.append(w1)
                        continue
                    # 내부 rglob로 best.pt 수집
                    for cand in p.rglob("best.pt"):
                        candidates.append(cand)
            except Exception:
                continue
        # fallback: 전역적으로 best.pt 검색
        for cand in root.rglob("best.pt"):
            candidates.append(cand)

    # 중복 제거 & 절대경로 문자열 기준 정렬 안정화
    seen = set()
    filtered = []
    for c in candidates:
        try:
            s = str(c.resolve())
        except Exception:
            s = str(c)
        if s not in seen:
            seen.add(s)
            filtered.append(Path(s))

    if len(filtered) == 0:
        fallback = Path("saved_model") / "best1.pt"
        if fallback.exists():
            return str(fallback)
        return None

    # 1) species 이름 포함 경로 우선 필터 (이미 대부분 후보는 포함되어 있음)
    species_candidates = [c for c in filtered if species in str(c)]
    pool = species_candidates if species_candidates else filtered

    # 2) 우선순위로 정렬: stage2 > stage1 > others
    def priority_score(p: Path):
        s = str(p).lower()
        if "stage2" in s or "stage_2" in s or "stage2_" in s or "1280" in s:
            return 0
        if "stage1" in s or "stage_1" in s or "960" in s:
            return 1
        # 약간 더 높은 점수는 낮은 우선순위
        return 2

    pool_sorted = sorted(pool, key=lambda p: (priority_score(p), len(str(p)), str(p)))

    # (디버그) 발견된 후보 목록 로그 출력 (원하면 주석처리)
    print(f"[DEBUG] find_yolo_path_for_species candidates for '{species}':")
    for i, c in enumerate(pool_sorted):
        print(f"  {i}: {c}")

    # 최종 반환
    return str(pool_sorted[0])



# 캐시용 딕셔너리
_yolo_cache = {}

def load_yolo_model_for_species(species: str):
    """
    species에 맞는 YOLO 모델을 로드(캐시). 파일 못 찾으면 None 반환.
    """
    path = find_yolo_path_for_species(species)
    if path is None:
        print(f"[WARN] '{species}'용 YOLO 모델을 찾을 수 없습니다.")
        return None
    if path in _yolo_cache:
        return _yolo_cache[path]
    try:
        print(f"[INFO] Trying to load YOLO from: {path}")
        y = YOLO(path)
        _yolo_cache[path] = y
        print(f"[INFO] YOLO loaded for species={species}, path={path}")
        return y
    except Exception as e:
        print(f"[WARN] YOLO load failed for {path}: {e}")
        return None
