"""Query-shape regressions for annotation QA endpoints."""
import sqlite3

from app import db
from app.api.qa import inconsistent_samples


def test_consistency_fetches_captions_in_one_batch():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    for sample_id in range(1, 5):
        conn.execute(
            "INSERT INTO samples("
            "id, dataset, filename, split, width, height, filesize, caption_consistency"
            ") VALUES (?, 'fixture', ?, 'train', 10, 10, 100, ?)",
            (sample_id, f"{sample_id}.jpg", sample_id / 10),
        )
        for caption_idx in range(2):
            conn.execute(
                "INSERT INTO captions(sample_id, idx, text) VALUES (?, ?, ?)",
                (sample_id, caption_idx, f"sample {sample_id} caption {caption_idx}"),
            )
    conn.commit()

    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    result = inconsistent_samples(limit=4, split=None, conn=conn)
    conn.set_trace_callback(None)

    selects = [statement for statement in statements if statement.lstrip().startswith("SELECT")]
    assert len(selects) == 2
    assert [item.caption for item in result] == [
        f"sample {sample_id} caption 0" for sample_id in range(1, 5)
    ]
