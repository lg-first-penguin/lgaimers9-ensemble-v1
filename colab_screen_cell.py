# ======================================================================
#  Colab A100 — 구조적 재검증 스크린 v2  (season 제거 + D1/B 크로스체크)
#  ▶ 이 셀 하나만 실행.
#
#  준비 (Google Drive):
#    MyDrive/lgaimers9_colab/
#      ├─ repo.zip                  ← 레포 루트 통째로 zip
#      ├─ train.csv  test.csv  trackman_history.csv  sample_submission.csv
#
#  configs: baseline / mlp_drop_season / cat_drop_season / d1_career_catdrop / b_monotone
#  regimes: cutoff7, 2023   (3-seed, cat_team 1126.77 베이스라인)
#  결과: 셀 출력 + MyDrive/lgaimers9_colab/structural_result_v2.json
#
#  ★ v1 이 "regime=cutoff7" 직후 에러로 멈춘 원인 = Colab numpy 2.x / catboost 1.2.10 ABI.
#    -> 아래에서 numpy<2 먼저 설치. 그래도 죽으면 셀 맨아래에 전체 traceback 이 찍힘.
# ======================================================================

DRIVE_DIR = "/content/drive/MyDrive/lgaimers9_colab"
CONFIGS   = "baseline,mlp_drop_season,cat_drop_season,d1_career_catdrop,b_monotone"
REGIMES   = ["cutoff7", "2023"]
SEEDS     = 3

# --- 0. catboost (대회 서버 버전) ---
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "catboost==1.2.10"], check=False)
import os, glob, shutil, time, json
import numpy as np
print(">> numpy", np.__version__, "(2.x 여도 catboost 1.2.10 은 지원 — 문제시 아래 traceback 확인)")

# --- 1. Drive + 레포/데이터 (레포는 무조건 재압축해제 = 옛 코드 잔존 방지) ---
from google.colab import drive
drive.mount("/content/drive")
assert os.path.isdir(DRIVE_DIR), f"{DRIVE_DIR} 없음 — Drive 에 폴더 만들고 repo.zip + csv 4개 올리기"

REPO = "/content/lgaimers9"
z = f"{DRIVE_DIR}/repo.zip"
assert os.path.exists(z), f"{z} 없음"
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree("/content/_rx", ignore_errors=True)
os.makedirs(REPO, exist_ok=True)
shutil.unpack_archive(z, "/content/_rx")
hit = next(iter(glob.glob("/content/_rx/**/code/train.py", recursive=True)), None)
assert hit, "repo.zip 안에서 code/train.py 못 찾음 (레포 루트를 통째로 압축했는지 확인)"
root = os.path.dirname(os.path.dirname(hit))
for item in os.listdir(root):
    shutil.move(os.path.join(root, item), os.path.join(REPO, item))
shutil.rmtree("/content/_rx", ignore_errors=True)
# 새 config 존재 확인 (repo.zip 이 최신인지 즉시 판별)
_sb = open(f"{REPO}/code/experiment_structural_batch.py").read()
assert "mlp_drop_season" in _sb, \
    "repo.zip 이 옛 버전 — Drive 의 repo.zip 을 새로 받은 것으로 덮어쓰고 다시 실행"
print(">> repo.zip 최신 확인 (mlp_drop_season config 존재)")

os.chdir(REPO)
os.makedirs("open/data", exist_ok=True)
os.makedirs("scratchpad", exist_ok=True)
for n in ("train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv"):
    if not os.path.exists(f"open/data/{n}"):
        s = f"{DRIVE_DIR}/{n}"
        assert os.path.exists(s), f"{s} 없음"
        shutil.copy2(s, f"open/data/{n}")
assert os.path.exists("teammate/yudam/model_py311/best_catboost_hparams_v2.json"), "repo.zip 에 teammate/ 누락"
try:
    import torch
    print(">> GPU:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print(">> torch import warn:", e)
print(">> 준비 완료:", sorted(os.listdir("open/data")))

# --- 2. 스크린 (live 스트리밍 + 로그 캡처) ---
OUT = "scratchpad/structural_result_v2.json"
if os.path.exists(OUT):
    os.remove(OUT)


def run_regime(R):
    print(f"\n{'#' * 26} regime={R}  {time.strftime('%H:%M:%S')} {'#' * 26}", flush=True)
    t0 = time.time()
    cmd = [sys.executable, "-u", "-m", "code.experiment_structural_batch",
           "--seeds", str(SEEDS), "--regimes", R, "--configs", CONFIGS, "--out", OUT]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            if len(tail) > 400:
                tail = tail[-400:]
        rc = proc.wait(timeout=60)
    except Exception as e:
        proc.kill()
        print(f"\n!! regime={R} 예외: {e!r}")
        rc = -1
    dt = time.time() - t0
    print(f"\n  -> regime={R} rc={rc}  {dt:.0f}s")
    if rc != 0:
        print(f"\n{'!' * 60}\n!! regime={R} 실패 — 마지막 40줄:\n{'!' * 60}")
        print("".join(tail[-40:]))
    return rc


for R in REGIMES:
    run_regime(R)

# --- 3. 결과 표 + Drive 저장 ---
if os.path.exists(OUT):
    shutil.copy2(OUT, f"{DRIVE_DIR}/structural_result_v2.json")
    d = json.load(open(OUT))
    print("\n" + "=" * 64 + "\n결과 (blend Δ vs cat_team 1126.77 baseline)\n" + "=" * 64)
    for reg, cf in d.get("regimes", {}).items():
        b = cf.get("baseline", {})
        bbl, bc, bm = b.get("blend"), b.get("cat_solo"), b.get("mlp_solo")
        print(f"\n[{reg}] baseline  blend={bbl:.2f}  cat={bc:.2f}  mlp={bm:.2f}"
              if bbl is not None else f"\n[{reg}] baseline 실패")
        if bbl is None:
            continue
        for k, r in cf.items():
            if k == "baseline":
                continue
            if "error" in r:
                print(f"  {k:22s} ERROR {r['error']}")
                continue
            print(f"  {k:22s} blend {r['blend'] - bbl:+7.2f} | cat {r['cat_solo'] - bc:+7.2f} | "
                  f"mlp {r['mlp_solo'] - bm:+7.2f}  (w_cat={r['w_cat']:.3f} w_mlp={r['w_mlp']:.3f})")
    print(f"\n>> {DRIVE_DIR}/structural_result_v2.json 저장 — 로컬로 받아서 공유")
else:
    print("\n!! 결과 json 없음 — 위 traceback 확인")
