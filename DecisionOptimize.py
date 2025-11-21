import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# 1. 가지치기 경로(후보 alpha 값들) 추출
# 이미 학습된 student_tree가 아니라, 새로 만들어서 경로를 탐색합니다.
# (데이터가 작다면 금방 끝납니다)
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_pred_teacher) # 선생님의 답을 정답으로
ccp_alphas = path.ccp_alphas # 후보 alpha 값들 (약한 가지치기 -> 강한 가지치기 순)

# 2. 후보 alpha들로 모델을 여러 개 만들어서 테스트
clfs = []
# 너무 많으면 오래 걸리니 끝에서부터 10~20개 정도만 보거나, 적당한 간격으로 테스트
selected_alphas = ccp_alphas[::int(len(ccp_alphas)/20) + 1] # 약 20개만 샘플링

print(f"총 {len(selected_alphas)}개의 가지치기 강도를 테스트합니다...")

train_scores = []
test_scores = []
node_counts = []

for ccp_alpha in selected_alphas:
    # 해당 alpha로 가지치기된 트리 생성
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_pred_teacher)
    
    # 성능 및 크기 기록
    train_scores.append(clf.score(X_train, y_pred_teacher))
    test_scores.append(clf.score(X_test, y_test)) # 실제 정답(y_test)과 비교
    node_counts.append(clf.tree_.node_count)

# 3. 결과 출력 (노드 개수 vs 정확도)
print("\n[결과 분석]")
print(f"{'Nodes':<10} | {'Accuracy':<10} | {'Alpha':<10}")
print("-" * 35)

for nodes, acc, alpha in zip(node_counts, test_scores, selected_alphas):
    # 정확도가 95% 이상 유지되는 것만 출력해서 보기
    if acc >= 0.95: 
        print(f"{nodes:<10} | {acc:.4f}     | {alpha:.6f}")

# 팁: 노드 개수(Nodes)가 확 줄어드는데 정확도(Accuracy)는 거의 그대로인 지점을 찾으세요!
