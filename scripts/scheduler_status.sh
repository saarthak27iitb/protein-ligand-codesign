#!/bin/bash
set -euo pipefail
squeue -u "$(id -u)" -o '%.18i %.24j %.10T %.10M %.6D %R'
