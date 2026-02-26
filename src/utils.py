import subprocess
import logging
import sys
import os

def run_external_script(script_name, args=None):
    """
    Wrapper to run external python scripts and pipe logs to Airflow.
    """
    logger = logging.getLogger("airflow.task")
    script_path = os.path.join("/opt/airflow/src", script_name)
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, "-u", script_path]
    if args:
        cmd.extend(args)

    logger.info(f"Executing: {' '.join(cmd)}")
    
    with subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1 
    ) as proc:
        for line in proc.stdout:
            logger.info(f"[{script_name}] {line.strip()}")
            
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    
    return True
