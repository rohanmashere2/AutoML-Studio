"""End-to-end test for AutoML Studio pipeline."""
import requests
import json
import time

BASE = "http://localhost:5000"

def test_pipeline():
    # Step 1: Upload
    print("=== STEP 1: Upload ===")
    with open("test_loan_data.csv", "rb") as f:
        r = requests.post(f"{BASE}/api/upload", files={"file": ("test_loan_data.csv", f)})
    data = r.json()
    sid = data.get("session_id", "")
    profile = data.get("profile", data)
    print(f"Session: {sid}")
    print(f"Rows: {profile.get('n_rows')}, Cols: {profile.get('n_cols')}")
    print(f"Target: {profile.get('target_column')}, Type: {profile.get('problem_type')}")

    # Step 2: Cleaning suggestions
    print("\n=== STEP 2: Cleaning Suggestions ===")
    r = requests.get(f"{BASE}/api/cleaning-suggestions/{sid}")
    cs = r.json()
    suggestions = cs.get("suggestions", [])
    print(f"Got {len(suggestions)} suggestions")
    for s in suggestions[:3]:
        print(f"  [{s['impact']}] {s['title']}")

    # Step 3: AutoEDA
    print("\n=== STEP 3: AutoEDA ===")
    r = requests.get(f"{BASE}/api/eda/{sid}")
    eda = r.json()
    if "error" in eda:
        print(f"EDA ERROR: {eda['error']}")
    else:
        insights = eda.get("insights", [])
        corrs = eda.get("correlations", {}).get("top_correlations", [])
        print(f"EDA OK: {len(insights)} insights, {len(corrs)} correlations")

    # Step 4: Clean & Transform
    print("\n=== STEP 4: Clean & Transform ===")
    r = requests.post(f"{BASE}/api/clean-transform", json={"session_id": sid})
    ct = r.json()
    print(f"Clean result: {ct.get('status', ct.get('error', '?'))}")

    # Step 5: Train
    print("\n=== STEP 5: Train Models ===")
    r = requests.post(f"{BASE}/api/train", json={"session_id": sid})
    tr = r.json()
    print(f"Train started: {tr.get('status', tr.get('error', '?'))}")

    # Poll for results
    print("Polling for results...")
    tres = None
    for i in range(60):
        time.sleep(2)
        r = requests.get(f"{BASE}/api/status/{sid}")
        st = r.json()
        if st.get("status") == "complete" and st.get("training_results"):
            tres = st["training_results"]
            best = tres.get("best_model", "?")
            score = tres.get("best_score", 0)
            print(f"Training DONE! Best: {best} Score: {score:.4f}")
            lb = tres.get("leaderboard", [])
            print(f"Leaderboard ({len(lb)} models):")
            for m in lb[:5]:
                mname = m.get('model', m.get('name', '?'))
                mscore = m.get('primary_metric', m.get('score', 0))
                print(f"  {mname}: {float(mscore):.4f}")
            break
        if st.get("status") == "error":
            print(f"Training ERROR")
            break
    else:
        print("Training timed out")
        return

    # Step 6: Explainability
    print("\n=== STEP 6: Explainability ===")
    r = requests.get(f"{BASE}/api/explain/{sid}")
    ex = r.json()
    if "global_importance" in ex:
        top3 = ex["global_importance"][:3]
        print(f"Top features: {[f['feature'] for f in top3]}")
    else:
        print(f"Explain: {ex.get('error', 'No data')}")

    # Step 7: Diagnostics
    print("\n=== STEP 7: Diagnostics ===")
    r = requests.get(f"{BASE}/api/diagnostics/{sid}")
    diag = r.json()
    has_cm = "confusion_matrix" in diag
    has_roc = "roc_curve" in diag or "roc_auc" in diag
    print(f"Confusion Matrix: {has_cm}, ROC: {has_roc}")

    # Step 8: Executive Summary
    print("\n=== STEP 8: Executive Summary ===")
    r = requests.get(f"{BASE}/api/executive-summary/{sid}")
    es = r.json()
    print(f"Best Model: {es.get('model', {}).get('best_model', '?')}")

    # Step 9: Pipeline Status
    print("\n=== STEP 9: Pipeline Status ===")
    r = requests.get(f"{BASE}/api/pipeline-status/{sid}")
    ps = r.json()
    for s in ps.get("steps", []):
        icon = "V" if s["status"] == "complete" else ">" if s["status"] == "active" else "_"
        print(f"  [{icon}] {s['name']}")

    # Step 10: Chat
    print("\n=== STEP 10: Chat ===")
    r = requests.post(f"{BASE}/api/chat/{sid}", json={"message": "What is the best model?"})
    chat = r.json()
    resp = str(chat.get("response", chat.get("error", "?")))
    print(f"Chat: {resp[:120].encode('ascii', 'replace').decode()}...")

    # Step 11: Hyperparameter Optimization
    print("\n=== STEP 11: Hyperparameter Optimization ===")
    r = requests.post(f"{BASE}/api/optimize/{sid}", json={"method": "random", "budget": 10})
    ho = r.json()
    if "error" in ho:
        print(f"Hyperopt error: {ho['error']}")
    else:
        print(f"Optimized {ho.get('optimized_count', 0)} models, best: {ho.get('best_model', '?')}")

    # Step 12: Projects
    print("\n=== STEP 12: Save Project ===")
    r = requests.post(f"{BASE}/api/projects/save", json={"session_id": sid, "name": "test_project"})
    sp = r.json()
    print(f"Save: {sp.get('success', sp.get('error', '?'))}")

    r = requests.get(f"{BASE}/api/projects")
    pl = r.json()
    print(f"Projects: {len(pl.get('projects', []))} saved")

    print("\n=== ALL 12 TESTS COMPLETE ===")


if __name__ == "__main__":
    test_pipeline()
