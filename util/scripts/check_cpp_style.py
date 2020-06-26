import subprocess
import re
import os
import shutil
import argparse
import multiprocessing
from functools import partial
import time
import sys
import fnmatch

CPP_FORMAT_DIRS = [
    "cpp",
    "examples",
    "docs/_static",
]


def _glob_files_py2(open3d_root_dir, directories, extensions):
    """
    Find files with certain extensions in directories recursively.
    To be compatible with Python 2, pathlib is avoided.

    Args:
        open3d_root_dir: Open3D root directory
        directories: list of directories, relative to open3d_root_dir.
        extensions: list of extensions, e.g. ["cpp", "h"].

    Return:
        List of file paths.
    """

    def _glob_recursive(directory, extension_regex):
        # https://stackoverflow.com/a/2186565/1255535
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, extension_regex):
                matches.append(os.path.join(root, filename))
        return matches

    file_paths = []
    for directory in directories:
        directory = os.path.join(open3d_root_dir, directory)
        for extension in extensions:
            extension_regex = "*." + extension
            file_paths.extend(_glob_recursive(directory, extension_regex))
    file_paths = [file_path for file_path in file_paths]
    file_paths = sorted(list(set(file_paths)))
    return file_paths


def _glob_files(open3d_root_dir, directories, extensions):
    """
    Find files with certain extensions in directories recursively.

    Args:
        open3d_root_dir: Open3D root directory
        directories: list of directories, relative to open3d_root_dir.
        extensions: list of extensions, e.g. ["cpp", "h"].

    Return:
        List of file paths.
    """
    file_path_new = _glob_files_py2(open3d_root_dir, directories, extensions)

    from pathlib import Path
    file_paths = []
    for directory in directories:
        directory = Path(open3d_root_dir) / directory
        for extension in extensions:
            extension_regex = "*." + extension
            file_paths.extend(directory.rglob(extension_regex))
    file_paths = [str(file_path) for file_path in file_paths]
    file_paths = sorted(list(set(file_paths)))

    assert (file_paths, file_path_new)
    return file_paths


glob_files = _glob_files


def _find_clang_format():
    # Find clang-format
    # > not found: throw exception
    # > version mismatch: print warning
    clang_format_bin = shutil.which("clang-format-5.0")
    if clang_format_bin is None:
        clang_format_bin = shutil.which("clang-format")
    if clang_format_bin is None:
        raise RuntimeError(
            "clang-format not found. "
            "See http://www.open3d.org/docs/release/contribute.html#automated-style-checker "
            "for help on clang-format installation.")
    version_str = subprocess.check_output([clang_format_bin, "--version"
                                          ]).decode("utf-8").strip()
    try:
        m = re.match("^clang-format version ([0-9.-]*) .*$", version_str)
        if m:
            version_str = m.group(1)
            version_str_token = version_str.split(".")
            major = int(version_str_token[0])
            minor = int(version_str_token[1])
            if major != 5 or minor != 0:
                print("Warning: clang-format 5.0 required, but got {}.".format(
                    version_str))
        else:
            raise
    except:
        print("Warning: failed to parse clang-format version {}".format(
            version_str))
        print("Please ensure clang-format 5.0 is used.")
    print("Using clang-format version {}.".format(version_str))

    return clang_format_bin


class CppFormatter:

    def __init__(self, file_paths, clang_format_bin):
        self.file_paths = file_paths
        self.clang_format_bin = clang_format_bin

    @staticmethod
    def _check_style(file_path, clang_format_bin):
        """
        Returns true if style is valid.
        """
        cmd = [
            clang_format_bin,
            "-style=file",
            "-output-replacements-xml",
            file_path,
        ]
        result = subprocess.check_output(cmd).decode("utf-8")
        if "<replacement " in result:
            return False
        else:
            return True

    @staticmethod
    def _apply_style(file_path, clang_format_bin):
        cmd = [
            clang_format_bin,
            "-style=file",
            "-i",
            file_path,
        ]
        subprocess.check_output(cmd)

    def run(self, do_apply_style, no_parallel, verbose):
        if do_apply_style:
            print("Applying C++/CUDA style...")
        else:
            print("Checking C++/CUDA style...")

        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print("> {}".format(file_path))

        start_time = time.time()
        if no_parallel:
            is_valid_files = map(
                partial(self._check_style,
                        clang_format_bin=self.clang_format_bin),
                self.file_paths)
        else:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                is_valid_files = pool.map(
                    partial(self._check_style,
                            clang_format_bin=self.clang_format_bin),
                    self.file_paths)

        changed_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            if not is_valid:
                changed_files.append(file_path)
                if do_apply_style:
                    self._apply_style(file_path, self.clang_format_bin)
        print("Formatting takes {:.2f}s".format(time.time() - start_time))

        return changed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_apply_style",
        dest="do_apply_style",
        action="store_true",
        default=False,
        help="Apply style to files in-place.",
    )
    parser.add_argument(
        "--no_parallel",
        dest="no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel execution.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="If true, prints file names while formatting.",
    )
    args = parser.parse_args()

    # Check formatting libs
    clang_format_bin = _find_clang_format()
    pwd = os.path.dirname(os.path.abspath(__file__))
    open3d_root_dir = os.path.join(pwd, "..", "..")

    # Check or apply style
    cpp_formatter = CppFormatter(
        _glob_files(open3d_root_dir=open3d_root_dir,
                    directories=CPP_FORMAT_DIRS,
                    extensions=["cpp", "h", "cu", "cuh"]),
        clang_format_bin=clang_format_bin,
    )

    changed_files = []
    changed_files.extend(
        cpp_formatter.run(do_apply_style=args.do_apply_style,
                          no_parallel=args.no_parallel,
                          verbose=args.verbose))

    if len(changed_files) != 0:
        if args.do_apply_style:
            print("Style applied to the following files:")
            print("\n".join(changed_files))
        else:
            print("Style error found in the following files:")
            print("\n".join(changed_files))
            exit(1)
    else:
        print("All files passed style check.")
