import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(script_dir, "main_Hao_Summary.py")

    env = os.environ.copy()
    env["ABLATION_SKIP_CONNECTION"] = "1"
    env["RPPG_DISABLE_TAM"] = "1"
    env["ABLATION_K_FOLD"] = "3"
    env["ABLATION_TAG"] = "ablation_residual_tsm_wo_tam_pure_36_3fold"

    cmd = [sys.executable, target]
    raise SystemExit(subprocess.call(cmd, cwd=script_dir, env=env))


if __name__ == "__main__":
    main()
