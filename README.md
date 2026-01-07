# AI-Model

![Image](https://github.com/user-attachments/assets/eca8bed0-1327-4f99-9419-197a5dca1138)

**🪴 AI Modeling & Problem Solving (My Role)**
본 프로젝트에서 저는 기능 구현을 위한 AI 모델 설계을 담당하였습니다. 단순히 기존 데이터셋을 학습시키는 것에 그치지 않고, 데이터의 한계를 극복하기 위해 다음과 같은 단계별 접근 방식을 도입했습니다.

![Image](https://github.com/user-attachments/assets/215c7d7d-8248-456d-99a2-932af6dca53e)

**🛠️ Core Contributions: AI Modeling & Optimization**

**1. Problem Definition & Analysis**

배경 노이즈 문제: AIHub 원예식물 데이터셋 내 배경(화분, 가구, 벽 등)이 복잡하여 모델이 식물 본연의 특징을 학습하지 못하고 과적합
(Overfitting)되는 문제 발생 

시각적 특징 인식 한계: 식물 종 간의 미세한 차이를 인식하는 데 초기 모델 성능의 한계를 확인 

**2. Technical Solutions**
객체 중심 파이프라인(Plant-Centric Pipeline) 설계: 배경 노이즈 제거를 위해 Detection-then-Classification 구조 도입 

YOLOv8 기반 Custom Detection:

LabelImg를 활용하여 데이터셋 내 식물 객체에 대한 Bounding Box 좌표를 직접 추출 및 라벨링 

'Plant' 클래스에 대한 전이학습(Transfer Learning)을 통해 식물 영역만 타이트하게 추출(Crop)하는 모델 구현 

데이터 정제 및 재구축: 전이학습된 YOLO 모델을 전체 데이터셋에 적용하여, 분류 모델이 식물 자체에만 집중할 수 있는 고품질 데이터셋으로 재구성 

**3. Multi-Stage Diagnosis System**

식물 종 분류: Crop된 이미지를 MobileNetV2에 입력하여 15종 다중 분류 수행 

상태 진단 및 관수 조절:

YOLOv11-Nano를 활용해 잎(Leaf) 단위를 탐지하고, MobileNetV3를 통해 과습/비과습 상태를 최종 판정하는 Voting 방식 도입 

진단 결과에 따라 식물 특성별 가중치를 부여하여 차기 관수 예정일을 유연하게 조절하는 알고리즘 구현


🔗 More Information
For a more in-depth look at the project overview, technical challenges, and the research journey behind this implementation, please visit my GitHub blog:

👉 **[My GitHub Blog: 식물을 부탁해 프로젝트 리뷰 보러가기](https://rlatmddn0211.github.io/)**
