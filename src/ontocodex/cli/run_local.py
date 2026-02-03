import argparse
import sys
import yaml
from pathlib import Path

# Ensure src is in python path if running directly
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ontocodex.engine.state import OntoCodexState
from ontocodex.agents.graph import create_graph


def load_config(path: str) -> dict:
    """Loads YAML configuration file."""
    if not Path(path).exists():
        print(f"Error: Config file not found at {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run OntoCodex Local Workflow")
    parser.add_argument("--ontology", required=True, help="Path to the input OWL ontology file")
    parser.add_argument("--config", default="configs/ontocodex.yaml", help="Path to configuration YAML")
    parser.add_argument("--task", default="enrich", choices=["map", "enrich"], help="Task type (map or enrich)")
    
    args = parser.parse_args()

    # 1. Load Configuration
    options = load_config(args.config)
    
    # 2. Initialize State
    print(f"Initializing OntoCodex for task '{args.task}' on ontology: {args.ontology}")
    state = OntoCodexState(
        run_id="local_run",
        task_type=args.task,
        ontology_path=args.ontology,
        options=options
    )

    # 3. Compile and Run Graph
    app = create_graph()
    print("Workflow started...")
    
    final_state = app.invoke(state)
    
    # 4. Report Results
    print("\nWorkflow finished.")
    if final_state.get("errors"):
        print(f"Errors encountered: {final_state['errors']}")
    else:
        print(f"Success! Output saved to: {final_state.get('artifacts', {}).get('enriched_owl', 'unknown location')}")

if __name__ == "__main__":
    main()