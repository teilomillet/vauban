import vauban
import os

def main():
    print("--- Vauban Example 05: Visualization ---")
    print("This example generates a standalone HTML map of the embeddings.")
    
    # Ensure you have run some attacks or established a baseline first so the DB is populated.
    if not os.path.exists(".lancedb"):
        print("No database found. Please run Example 02 or 03 first.")
        return

    # Generate the Atlas
    # This uses Nomic's embedding-atlas library wrapped by Vauban
    try:
        vauban.visualize(db_path=".lancedb", table_name="refusals")
        print("\nVisualization generated (check your browser or output folder).")
    except Exception as e:
        print(f"\nError generating visualization: {e}")
        print("Note: You may need to install 'embedding-atlas' and have a valid environment.")

if __name__ == "__main__":
    main()
