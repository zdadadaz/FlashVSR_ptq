import sys, os, subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, REPO_ROOT)

def test_cli_has_w8a8_ptq_flags():
    result = subprocess.run(
        [sys.executable, 'cli_main.py', '--help'],
        capture_output=True, text=True,
        cwd=REPO_ROOT
    )
    assert 'W8A8_PTQ' in result.stdout, "W8A8_PTQ not in --help output"
    assert '--trt_engine' in result.stdout, "--trt_engine not in --help output"
    print("test_cli_has_w8a8_ptq_flags PASS")