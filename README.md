## 📌 머신러닝 기반 얼굴 및 제스처 인식 출퇴근 기록 시스템
- 웹캠을 통해 사용자의 얼굴을 등록 및 인식하고, 손 제스처를 통해 출근/퇴근/외출/복귀 기록
- 기존 출입카드 시스템의 단점(분실, 도용 등)을 해결하고, 비접촉 방식으로 출입 관리 가능


---


## 📅 개발 기간
- 24/05/27 ~ 24/06/16


---


## 👨‍💻 개발 담당


---


## 🛠️ 주요 라이브러리
- OpenCV (cv2) : 얼굴 감지, 웹캠 연동, 이미지 전처리
- Face Recognition (face_recognition) : 얼굴 특징 추출(128차원 벡터) 및 비교
- MediaPipe : 손 제스처 인식 (21개 랜드마크 검출)
- Scikit-learn (sklearn) : SVM, KNN, RandomForest, AdaBoost 모델 학습
- NumPy (numpy) : 벡터 연산 및 데이터 처리
- Pandas (pandas) : 출퇴근 로그 파일(attendance_log.txt) 관리
- Pillow (PIL) → OpenCV 이미지를 활용한 UI 버튼 및 텍스트 출력
- os → 시스템 폴더/파일 관
- time → 인증 시간 기록 및 지연 처리

---


## 🎯 구현 기능
- 얼굴 인식 기반 사용자 인증 (Face Recognition)
- 손 제스처 인식 기능 (MediaPipe)
- 가상 버튼 UI를 활용한 출퇴근 입력 시스템
- 출퇴근 로그 기록 (attendance_log.txt)
- 머신러닝 모델 학습 (SVM, AdaBoost, CNN)
- 데이터 증강 기법 적용 (얼굴 이미지 10배 증강 후 학습)


---


## 📊 기능블록도


---


## 📂 프로젝트 구조
```
📂출퇴근시스템/
│── 📂.ipynb_checkpoints/            # Jupyter 자동 백업 폴더
│── 📂김주리/                         # 사용자 얼굴 데이터 폴더
│── 📂유창민/
│── 📂조윤서/
│── 📂README.md                      # 프로젝트 설명
│── 📂attendance_log.txt              # 출퇴근 기록 로그 파일
│── 📂ensemble_model.pkl              # 머신러닝 앙상블 모델
│── 📂ensemble_model2.pkl             # 추가 학습된 모델
│── 📂앙상블 모델(최종).ipynb         # Jupyter Notebook (머신러닝 모델 학습 및 출퇴근 기록)

```


---


## 📷 시연
### 1️⃣ 웹캠을 통해 사용자 얼굴 캡처 및 저장 (100장)
![image](https://github.com/user-attachments/assets/d1db95f6-69f7-496d-bc1a-997f387074a7)


![image](https://github.com/user-attachments/assets/8fdaf392-c556-4fb1-b2a6-c8989e1d3aed)


![image](https://github.com/user-attachments/assets/c3d46231-2cec-427c-bdd7-f28141332d15)


### 2️⃣ 한 장당 10개의 데이터셋 증가 (이미지 증강 기법)
![image](https://github.com/user-attachments/assets/bfdb3510-e322-4ce1-aef3-1cb4abf9bedc)


![image](https://github.com/user-attachments/assets/97e81eee-a851-457e-a5ff-b9e7180fdfd2)


### 3️⃣ 얼굴 데이터 학습
![image](https://github.com/user-attachments/assets/32ad9400-2c85-42cf-8983-bd2409650fe4)


### 3️⃣ 사용자 인증 
![image](https://github.com/user-attachments/assets/ee472bc0-5a9d-42c4-8727-07b8e0be0833)


![image](https://github.com/user-attachments/assets/f7f1d0c5-e72e-486e-9beb-8a742bd9acb8)


### 4️⃣ 인증 성공 시 가상 버튼 인터페이스 출력
![image](https://github.com/user-attachments/assets/0c390fa8-706c-4893-9272-801a6c2f71dc)


### 5️⃣ 버튼에 3초 이상 손을 대고 있으면 선택 완료(손이 아닌 것은 인식 X)
![image](https://github.com/user-attachments/assets/3e3977ab-5baf-4236-a79b-95c32f8d2d80)


![image](https://github.com/user-attachments/assets/b6c08b27-78eb-4f09-bb33-adcf382cae21)


![image](https://github.com/user-attachments/assets/2fd09040-9e40-4d80-81c3-c8a846a9589d)


![image](https://github.com/user-attachments/assets/fc0269cd-55a8-4665-abbf-3d3d47d865dc)


### 6️⃣ 실시간 출퇴근 로그 기록
![image](https://github.com/user-attachments/assets/d0d52184-f35b-45bf-805d-f2379fab6012)


---


## 🔗 블로그: 프로젝트 관련 포스트


---

