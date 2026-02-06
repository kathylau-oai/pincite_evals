import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pincite_evals.synthetic_generation.config import load_config  # noqa: E402
from pincite_evals.synthetic_generation.pipeline import SyntheticGenerationPipeline  # noqa: E402


def test_pipeline_e2e_333_dry_run(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
packet_id: packet_1
output_root: {(tmp_path / 'results').as_posix()}
dataset_root: {(tmp_path / 'datasets').as_posix()}
dry_run: true
generate_count:
  overextension: 3
  precedence: 3
  fake_citations: 3
final_keep_count:
  overextension: 3
  precedence: 3
  fake_citations: 3
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    pipeline = SyntheticGenerationPipeline(config)
    context = pipeline.bootstrap(run_id="e2e")

    summary = pipeline.run_all(context=context, openai_client=None)

    assert summary["selected_items"] == 9

    dataset_csv = Path(summary["dataset_dir"]) / "synthetic_items.csv"
    saved = pd.read_csv(dataset_csv)
    mode_counts = saved.groupby("target_error_mode")["item_id"].count().to_dict()

    assert mode_counts["A"] == 3
    assert mode_counts["C"] == 3
    assert mode_counts["D"] == 3
