import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pincite_evals.synthetic_generation.config import load_config  # noqa: E402
from pincite_evals.synthetic_generation.pipeline import SyntheticGenerationPipeline  # noqa: E402


def test_pipeline_smoke_dry_run(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
packet_id: packet_1
output_root: {(tmp_path / 'results').as_posix()}
dataset_root: {(tmp_path / 'datasets').as_posix()}
dry_run: true
generate_count:
  overextension: 2
  precedence: 2
  fake_citations: 2
final_keep_count:
  overextension: 1
  precedence: 1
  fake_citations: 1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    pipeline = SyntheticGenerationPipeline(config)
    context = pipeline.bootstrap(run_id="smoke")

    summary = pipeline.run_all(context=context, openai_client=None)

    assert summary["packet_block_rows"] > 0
    assert summary["generated_counts"]["overextension"] == 2
    assert summary["generated_counts"]["precedence"] == 2
    assert summary["generated_counts"]["fake_citations"] == 2
    assert summary["selected_items"] == 3
    assert Path(summary["dataset_dir"]).exists()

    run_root = Path(summary["run_root"])
    llm_reviews_csv = run_root / "validation" / "llm_consensus_reviews.csv"
    validation_datapoints_csv = run_root / "validation" / "validation_datapoints.csv"

    assert llm_reviews_csv.exists()
    assert validation_datapoints_csv.exists()

    llm_reviews_df = pd.read_csv(llm_reviews_csv)
    validation_datapoints_df = pd.read_csv(validation_datapoints_csv)

    assert len(llm_reviews_df) == 6
    assert len(validation_datapoints_df) == 6
    assert "llm_verdict" in llm_reviews_df.columns
    assert "final_validation_status" in validation_datapoints_df.columns
