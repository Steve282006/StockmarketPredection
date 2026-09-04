import json
import traceback

def run_notebook_cells(nb_path="stock_analysis_prediction.ipynb", out_path="stock_analysis_prediction.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Local execution context
    exec_globals = {}
    
    execution_count = 1
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source_code = "".join(cell.get("source", []))
            cell["execution_count"] = execution_count
            execution_count += 1
            try:
                # Execute Python code snippet
                exec(source_code, exec_globals)
                cell["outputs"] = [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["Cell executed successfully.\n"]
                }]
            except Exception as e:
                err_msg = traceback.format_exc()
                cell["outputs"] = [{
                    "output_type": "stream",
                    "name": "stderr",
                    "text": [err_msg]
                }]
                print(f"Error executing cell {cell['execution_count']}: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Notebook executed and saved successfully!")

if __name__ == "__main__":
    run_notebook_cells()
