import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 전제: xgb_model이 이미 학습되어 있어야 함
# ==========================================
# 예시 데이터가 없으므로 가정하고 넘어갑니다. 
# X_train, X_test, y_train, y_test, xgb_model은 이미 메모리에 있다고 가정합니다.

# ==========================================
# 2. 지식 증류: XGBoost를 단일 트리로 압축
# ==========================================
print("Creating Student Tree (Distillation)...")

# 선생님(XGBoost)의 예측값을 정답지로 사용
y_pred_teacher = xgb_model.predict(X_train)

# 학생(Single Tree) 학습
# 주의: XGBoost 성능을 따라잡기 위해 깊이 제한을 풉니다 (Overfitting 유도)
student_tree = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=None,      # 깊이 무제한 (복잡한 로직을 다 베끼기 위해)
    min_samples_leaf=1,  # 끝까지 분할
    random_state=42
)
student_tree.fit(X_train, y_pred_teacher)

# 압축 성능 확인
student_acc = student_tree.score(X_test, y_test)
print(f"XGBoost 성능: 96% (가정)")
print(f"단일 트리로 압축 후 성능: {student_acc * 100:.2f}%")
print("-" * 30)

# ==========================================
# 3. C언어 변환기 (If-Else + LUT 구조)
# ==========================================
def export_to_c_with_lut(tree, class_names):
    tree_ = tree.tree_
    feature_names = [f"input[{i}]" for i in range(7)] # 입력이 7개 정수 배열이라고 가정
    
    # 1. LUT (Look-Up Table) 생성
    # 트리의 리프 노드가 뱉는 클래스 인덱스를 실제 값(17~60)으로 매핑
    print("// [LUT] Class Index -> Actual Value Mapping")
    print(f"static const int CLASS_LUT[] = {{ {', '.join(map(str, class_names))} }};")
    print("\n// Prediction Function")
    print("int predict_class(int *input) {")

    # 2. 재귀적으로 트리를 순회하며 If-Else 생성
    def recurse(node, depth):
        indent = "    " * depth
        
        # 리프 노드인 경우
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # 리프 노드에 도달하면, 가장 빈도 높은 클래스의 '인덱스'를 반환
            # (여기서 인덱스는 class_names 배열의 인덱스입니다)
            value = tree_.value[node]
            class_idx = np.argmax(value)
            
            # 주석으로 어떤 값인지 표시해주고, LUT 참조 리턴
            print(f"{indent}// Prediction: {class_names[class_idx]}")
            print(f"{indent}return CLASS_LUT[{class_idx}];")
        
        # 분기 노드인 경우
        else:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # 정수 데이터이므로 threshold를 int로 변환하여 깔끔하게 출력
            # Scikit-learn은 threshold가 float(x.5) 형태이므로 int로 내림 처리해도 됨
            # 문맥상 'x <= 2.5'는 'x <= 2'와 같음 (정수일 때)
            thresh_int = int(threshold)
            
            print(f"{indent}if ({name} <= {thresh_int}) {{")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            print(f"{indent}}}")

    recurse(0, 1)
    print("}")

# ==========================================
# 4. 실제 변환 실행
# ==========================================
# xgb_model.classes_ 에는 [17, 19, 25, 60 ...] 실제 정답 값들이 들어있음
export_to_c_with_lut(student_tree, xgb_model.classes_)
