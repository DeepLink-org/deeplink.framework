import io
import os
import pathlib
import queue
import sys
from multiprocessing import Process, Queue, set_start_method
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, List, Tuple, TypedDict, Union
from .stdout_redirector import stdout_redirector


class Args(TypedDict, total=False):
    args: tuple
    kwargs: dict


def _run_individual_test_cases_sequential(
    entry_points: Iterable[Tuple[Callable, Args]]
) -> None:
    all_tests_pass = True
    for entry_point, args in entry_points:
        p = Process(
            target=entry_point, args=args.get("args", ()), kwargs=args.get("kwargs", {})
        )
        p.start()
        p.join()
        all_tests_pass = all_tests_pass and p.exitcode == 0
    assert all_tests_pass


def _entry_point_wrapper(
    entry_point: Callable, future_output: Queue, log_dir: str, *args, **kwargs
) -> None:
    sys.stderr = open(f"{log_dir}/stderr_{os.getpid()}", "w")
    captured = io.BytesIO()
    try:
        with stdout_redirector(captured):
            entry_point(*args, **kwargs)
    finally:
        future_output.put(captured.getvalue().decode("utf-8"))


def _run_individual_test_cases_parallel(
    entry_points: Iterable[Tuple[Callable, Args]]
) -> None:
    with TemporaryDirectory() as tmpdir:
        future_outputs: List[Queue] = []
        ps: List[Process] = []
        for entry_point, args in entry_points:
            future_output = Queue()
            p = Process(
                target=_entry_point_wrapper,
                args=(entry_point, future_output, tmpdir) + args.get("args", ()),
                kwargs=args.get("kwargs", {}),
            )
            p.start()
            future_outputs.append(future_output)
            ps.append(p)

        all_tests_pass = True
        for p, future_output in zip(ps, future_outputs):
            p.join()
            try:
                print(future_output.get_nowait(), end="")
            except queue.Empty:
                all_tests_pass = False
            print(
                pathlib.Path(f"{tmpdir}/stderr_{p.pid}").read_text(),
                end="",
                file=sys.stderr,
            )
            all_tests_pass = all_tests_pass and p.exitcode == 0
        assert all_tests_pass


def run_individual_test_cases(
    entry_points: Iterable[Union[Callable, Tuple[Callable, Args]]],
    in_parallel: bool = False,
) -> None:
    """
    Run test cases in individual processes in parallel or sequential.
    WARN: This function must be called within an `if __name__ == "__main__"` region.
    ---
    Args:
        `entry_points`: A sequence of test cases. Each test case is either a function
            or a tuple of a function and its arguments
            `(func, {"args": [...], "kwargs": {...}})`.
        `in_parallel`: Whether to run test cases in parallel.
    """
    set_start_method("spawn", force=True)  # this is required for gcov to work
    uniform_entry_points: Iterable[Tuple[Callable, Args]] = map(
        lambda x: x if isinstance(x, tuple) else (x, {}), entry_points
    )
    if in_parallel:
        _run_individual_test_cases_parallel(uniform_entry_points)
    else:
        _run_individual_test_cases_sequential(uniform_entry_points)
