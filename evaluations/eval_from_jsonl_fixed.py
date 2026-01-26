def extract_task_name_from_results(output_dir: Path) -> str:
    """Extract task name from results JSON file"""
    # Look for results JSON file in any __data__* subdirectory
    for subdir in output_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("__data__"):
            # Find the most recent results JSON file
            results_files = list(subdir.glob("results_*.json"))
            if results_files:
                # Sort by modification time, get the most recent
                results_file = max(results_files, key=lambda p: p.stat().st_mtime)
                with open(results_file) as f:
                    results = json.load(f)
                    # Extract task name from configs
                    if "configs" in results and results["configs"]:
                        task_name = list(results["configs"].keys())[0]
                        config = results["configs"][task_name]
                        if "task" in config:
                            return config["task"]
                        return task_name
    return None

