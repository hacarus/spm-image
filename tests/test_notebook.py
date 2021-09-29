import json
import os
import pathlib
import subprocess
import sys
import time
import unittest

class TestJupyterNotebook(unittest.TestCase):
    def run_notebook(self, notebook) -> None:
        """
        execute notebook and write the result into `out_notebook`
        Parameter:
            notebook: a path of .ipynb
        Return:
            None
        """
        out_notebook = os.path.join(notebook.parent, self.out_notebook_name)
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                notebook,
                "--output",
                out_notebook,
                "--ExecutePreprocessor.timeout=86400",
                "--allow-errors",
                "--debug",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return result.returncode

    def parse_notebook(self, notebook) -> bool:
        """
        parse notebook and report errors
        Parameter:
            notebook: a path of .ipynb
        Return:
              True: an error exists
              False: no error
        """
        is_error_exists = False
        out_notebook = os.path.join(notebook.parent, self.out_notebook_name)
        with open(out_notebook, "r") as f:
            json_dict = json.load(f)
            for cell in json_dict["cells"]:
                if (
                    "outputs" in cell
                    and len(cell["outputs"]) > 0
                    and "output_type" in cell["outputs"][0]
                    and cell["outputs"][0]["output_type"] == "error"
                ):
                    msg_notebook_info = (
                        "notebook name: {name}\n"
                        "execution_count: {cnt}\n".format(
                            name=notebook, cnt=cell["execution_count"]
                        )
                    )
                    msg_code = (
                        "-----code-----\n"
                        "{code}\n"
                        "--------------\n".format(code=("\n").join(cell["source"]))
                    )
                    msg_traceback = (
                        "---traceback--\n"
                        "{traceback}"
                        "\n--------------\n".format(
                            traceback=("\n").join(cell["outputs"][0]["traceback"])
                        )
                    )
                    error_msg = msg_notebook_info + msg_code + msg_traceback
                    logger.error(error_msg)

                    is_error_exists = True

        if os.path.exists(out_notebook):
            os.remove(out_notebook)

        return is_error_exists

    def setUp(self):
        self.out_notebook_name = "tmp.ipynb"
        self.DIR_NOTEBOOK = "../examples"
        sys.path = sys.path + [self.DIR_NOTEBOOK]

        # the notebook in this directory is pass in unittest
        self.EXCLUDE_DIR = [
            ".ipynb_checkpoints",
        ]

    # @unittest.skip("due to high computation")
    def test_all_notebooks(self):
        # get directory of notebooks
        path_notebook = pathlib.Path(
            os.path.join(os.path.dirname(__file__), self.DIR_NOTEBOOK)
        )

        is_error = []
        times = {}
        for notebook in path_notebook.glob("**/*.ipynb"):
            skip_flag = False
            for exclude_dir in self.EXCLUDE_DIR:
                if exclude_dir in str(notebook.parent):
                    skip_flag = True
            if skip_flag:
                continue
            print("Run: ", notebook.name)
            start = time.time()
            status = self.run_notebook(notebook)
            if status != 0:
                print("error:", notebook)
            else:
                elapsed_time = time.time() - start
                times[notebook.name] = elapsed_time
                result = self.parse_notebook(notebook)
                is_error.append(result)

        print("---computation time for each notebook---")
        for notebook, t in times.items():
            print(f"{notebook:<50}: {t:<8.2f} [sec]")

        self.assertNotIn(True, is_error)
