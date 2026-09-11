#!/usr/bin/env python3
"""Offline C++/CUDA environment in an ordinary user's workspace, without TE.

Input: a directory containing manifest.json {basename: sha256} and the pinned
CMake/Ninja wheels, CUTLASS tarball and MPICH wheel. No network installation,
driver changes, global Python changes, or shell startup-file edits.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex
import socket
import subprocess
import tarfile
import tempfile
import venv
import zipfile


def run(argv, **kwargs):
    subprocess.run([str(x) for x in argv], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--offline-dir', required=True, type=Path)
    args = parser.parse_args()
    user = pwd.getpwuid(os.geteuid())
    workspace = args.workspace
    expected = Path(user.pw_dir) / 'workspace_wct'
    if os.geteuid() == 0 or workspace != expected or workspace.resolve() != workspace:
        raise ValueError('Use a real workspace_wct immediately inside the current non-root home')
    if socket.gethostname() not in ('l20d-xerkjfcp-0001', 'l20d-ebed3kz6-0000'):
        raise ValueError('Not a configured L20D node')
    workspace.mkdir(exist_ok=True)
    if workspace.stat().st_uid != os.geteuid():
        raise ValueError('Workspace is not owned by the current user')
    control = workspace / '.l20d'
    control.mkdir(exist_ok=True)
    with (control / 'workspace.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        offline = args.offline_dir.resolve()
        manifest = json.loads((offline / 'manifest.json').read_text())
        for name, digest in manifest.items():
            if Path(name).name != name or not name:
                raise ValueError('Unsafe package basename')
            with (offline / name).open('rb') as source:
                actual = hashlib.file_digest(source, 'sha256').hexdigest()
            if actual != digest:
                raise ValueError('Package hash mismatch: ' + name)
        print('BOOTSTRAP verified offline package hashes', flush=True)
        for path in ('bench-env', 'toolchain', 'deps'):
            target = workspace / path
            if target.is_symlink() or (target.exists() and target.stat().st_uid != os.geteuid()):
                raise ValueError('Unsafe existing environment path: ' + str(target))
        python = workspace / 'bench-env/bin/python'
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(workspace / 'bench-env')
        tools = workspace / 'toolchain'
        tools.mkdir(exist_ok=True)
        tools_bin = tools / 'bin'
        tools_bin.mkdir(exist_ok=True)
        for package in ('cmake', 'ninja'):
            wheels = [name for name in manifest if name.startswith(package + '-') and name.endswith('.whl')]
            if len(wheels) != 1:
                raise ValueError('Expected exactly one ' + package + ' wheel')
            binary = tools / 'python' / ('cmake/data/bin/cmake' if package == 'cmake' else 'bin/ninja')
            if not binary.exists():
                if package == 'ninja':
                    # This wheel stores the ELF in .data/scripts, not a Python
                    # package. Avoid pip --target silently skipping an existing bin/.
                    with zipfile.ZipFile(offline / wheels[0]) as wheel:
                        names = [n for n in wheel.namelist() if n.endswith('.data/scripts/ninja')]
                        if len(names) != 1:
                            raise ValueError('Unexpected Ninja wheel layout')
                        binary.parent.mkdir(parents=True, exist_ok=True)
                        binary.write_bytes(wheel.read(names[0]))
                        binary.chmod(0o755)
                else:
                    run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
                         '--disable-pip-version-check', '--target', tools / 'python', offline / wheels[0]])
            link = tools_bin / package
            if not os.path.lexists(link):
                link.symlink_to(binary.relative_to(tools_bin, walk_up=True))
            if link.resolve() != binary.resolve():
                raise ValueError('Conflicting tool: ' + str(link))
            run([link, '--version'])
        deps = workspace / 'deps'
        deps.mkdir(exist_ok=True)
        cutlass_name = 'cutlass-57e3cfb47a2d9e0d46eb6335c3dc411498efa198'
        if not (deps / cutlass_name).exists():
            with tempfile.TemporaryDirectory(prefix='.cutlass-', dir=deps) as staging:
                with tarfile.open(offline / 'cutlass-57e3cfb.tar.gz') as archive:
                    archive.extractall(staging, filter='data')
                (Path(staging) / cutlass_name).rename(deps / cutlass_name)
        if not (deps / cutlass_name / 'include/cutlass/cutlass.h').is_file():
            raise ValueError('Incomplete CUTLASS checkout')
        env_file = workspace / 'env.sh'
        mpi = tools / 'mpich-5.0.1.post1'
        content = ('# Source only in a fuse build/test subshell. No global environment edits.\n'
            f'export FUSE_WORKSPACE={shlex.quote(str(workspace))}\n'
            f'export CUTLASS_ROOT={shlex.quote(str(deps / cutlass_name))}\n'
            'export CUDACXX=/usr/local/cuda/bin/nvcc\n'
            f'export PATH="{tools_bin}:{mpi}/bin:{python.parent}:'
            '/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda/bin:${PATH}"\n'
            f'export LD_LIBRARY_PATH="{mpi}/lib:/usr/local/cuda/lib64:${{LD_LIBRARY_PATH:-}}"\n'
            'export MPICH_CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++\n'
            'export UCX_TLS=sm,self\nexport BUILD_JOBS=4\n')
        if env_file.exists() and env_file.read_text() != content:
            raise ValueError('Existing env.sh differs; inspect instead of overwriting')
        env_file.write_text(content)
        print('BOOTSTRAP CMake/Ninja/CUTLASS ready; Python is private and has no TE', flush=True)
    # The MPI setup shares this same lock, so call only after releasing it.
    mpi_wheels = [name for name in manifest if name.startswith('mpich-') and name.endswith('.whl')]
    if len(mpi_wheels) != 1:
        raise ValueError('Expected one pinned MPICH wheel')
    run(['bash', Path(__file__).with_name('setup_sm103_mpi.sh'), offline / mpi_wheels[0]],
        env=os.environ | {'FUSE_WORKSPACE': str(workspace)})
    print('BOOTSTRAP PASS: private tools + CPU MPI; GPU execution not yet validated', flush=True)


if __name__ == '__main__':
    main()
