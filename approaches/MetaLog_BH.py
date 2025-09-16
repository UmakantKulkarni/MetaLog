import os
import subprocess
import sys


if __name__ == '__main__':
    script_path = os.path.join(os.path.dirname(__file__), 'MetaLog.py')
    cmd = [sys.executable, script_path]
    args = sys.argv[1:]
    if '--target_dataset' not in args and '--target-dataset' not in args:
        cmd.extend(['--target_dataset', 'HDFS'])
    if '--source_dataset' not in args and '--source-dataset' not in args:
        cmd.extend(['--source_dataset', 'BGL'])
    cmd.extend(args)
    sys.exit(subprocess.call(cmd))
