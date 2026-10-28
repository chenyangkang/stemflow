"""Execute example notebooks with Jupyter or an IPython process without sockets."""

import argparse
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]


def execute_in_process(notebook):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    import matplotlib
    from matplotlib_inline.backend_inline import configure_inline_support
    import nbformat
    shell = InteractiveShell.instance()
    matplotlib.use("module://matplotlib_inline.backend_inline")
    configure_inline_support(shell, "inline")
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        cell.outputs = []
        for stream in ["stdout", "stderr"]:
            value = getattr(captured, stream)
            if value:
                cell.outputs.append(nbformat.v4.new_output("stream", name=stream, text=value))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data,
                                                       metadata=output.metadata))
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise RuntimeError(f"Cell {count} failed: {error}") from error
        print(f"Executed cell {count}", flush=True)
    notebook.metadata["execution_engine"] = "IPython in process; all code cells executed"
    notebook.metadata.setdefault("language_info", {}).update(
        name="python", version=platform.python_version())
    import stemflow
    notebook.metadata["stemflow_execution"] = dict(
        version=stemflow.__version__, python=platform.python_version(),
        executed_at_utc=datetime.now(timezone.utc).isoformat(),
        engine="IPython in process")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--extra-site-packages", type=Path,
                        help="Optional notebook tools from a matching Python ABI; runtime packages take precedence.")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    if args.extra_site_packages:
        sys.path.append(str(args.extra_site_packages.resolve()))
    import nbformat
    notebook = nbformat.read(args.notebook, as_version=4)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if args.in_process:
        execute_in_process(notebook)
    else:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=300, kernel_name=args.kernel,
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Saved executed notebook: {args.output}")


if __name__ == "__main__":
    main()
