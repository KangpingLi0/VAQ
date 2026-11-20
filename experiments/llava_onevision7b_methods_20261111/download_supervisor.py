"""Keep verified resumable model acquisition alive across transient failures."""
import bootstrap
import json
import subprocess
import sys
import time


if __name__=='__main__':
    attempts=0
    while not (bootstrap.OUT/'model_integrity.json').exists():
        attempts+=1
        print('Resumable download attempt',attempts,flush=True)
        result=subprocess.run([sys.executable,'-u',str(bootstrap.BASE/'complete_download_ranges.py')])
        if result.returncode==0:
            integrity=json.loads((bootstrap.OUT/'model_integrity.json').read_text())
            assert integrity['status']=='passed'
            break
        print('Download connection failure; all completed ranges retained; retrying',flush=True)
        time.sleep(15)
