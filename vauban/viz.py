import lancedb
import numpy as np
from embedding_atlas.widget import EmbeddingAtlasWidget


def create_atlas(db_path: str = ".lancedb", table_name: str = "refusals"):
    """
    Load vectors from LanceDB and create an Embedding Atlas.
    """
    db = lancedb.connect(db_path)
    if table_name not in db.table_names():
        print("No data to visualize.")
        return

    tbl = db.open_table(table_name)
    df = tbl.to_pandas()

    if df.empty:
        print("Table is empty.")
        return

    # Extract vectors and metadata
    vectors = np.stack(df["vector"].values)

    # Prepare metadata for the atlas
    # We want to color by 'type' (refusal vs attack vs failed_attack)
    metadata = df[["prompt", "response", "type", "score"]].to_dict(orient="records")

    print(f"Building Atlas with {len(vectors)} points...")

    # Build the atlas
    # EmbeddingAtlasWidget expects embeddings and metadata
    # It seems to be a widget, so it might not show up in a script without a server or notebook.
    # But we can at least instantiate it to verify it works.
    atlas = EmbeddingAtlasWidget(embeddings=vectors, metadata=metadata)

    print(
        "Atlas widget created. In a Jupyter environment, this would display the interactive map."
    )
    return atlas
