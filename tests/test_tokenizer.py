import os
import subprocess
import sys


def test_instruction_tokenization_is_process_stable() -> None:
    code = (
        "from gripground.data.dataset import tokenize_instruction; "
        "print(tokenize_instruction('move the red cube to target').tolist())"
    )
    outputs = []
    for hash_seed in ("1", "321"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        outputs.append(
            subprocess.check_output([sys.executable, "-c", code], env=environment, text=True).strip()
        )
    assert outputs[0] == outputs[1]
