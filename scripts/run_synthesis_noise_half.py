"""Five-round, two-view Adapter/LoRA check with explicit noise_scale=0.5."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_global_synthesis_validation import main as run_validation


def build_overrides(arguments):
    return [
        '--defenses', 'risk_synthesis',
        '--rounds', '5',
        '--attacks', 'none',
        '--set', 'eval_interval=1',
        '--set', 'defense.synthesis.noise_scale=0.5',
        *arguments,
    ]


if __name__ == '__main__':
    raise SystemExit(run_validation(build_overrides(sys.argv[1:])))
