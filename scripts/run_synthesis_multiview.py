"""Two jointly trained virtual views; forward all overrides to the unified runner."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    defaults = ['--models','clip_adapter','--datasets','cifar100','--methods','fedavg',
                '--defenses','risk_synthesis','--attacks','all','--seeds','43','--rounds','100',
                '--set','defense.synthesis.views_per_record=2',
                '--set','defense.synthesis.risk_history=none',
                '--set','defense.synthesis.candidate_selection=first_semantic']
    return subprocess.call([sys.executable,str(ROOT/'scripts/run_privacy_experiments.py'),
                            *defaults,*sys.argv[1:]],cwd=ROOT)


if __name__=='__main__':
    raise SystemExit(main())
