#!/bin/bash
#SBATCH --job-name=sd-usage   # Kurzname des Jobs
#SBATCH --nodes=1                 # Anzahl benötigter Knoten
#SBATCH --ntasks=1                # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --partition=p0            # Verwendete Partition (z.B. p0, p1, p2 oder all)
#SBATCH --time=08:00:00           # Gesamtlimit für Laufzeit des Jobs (Format: HH:MM:SS)
#SBATCH --cpus-per-task=8         # Rechenkerne pro Task
#SBATCH --mem=32G                 # Gesamter Hauptspeicher pro Knoten
#SBATCH --gres=gpu:1              # Gesamtzahl GPUs pro Knoten
#SBATCH --qos=basic               # Quality-of-Service
#SBATCH --mail-type=FAIL           # Art des Mailversands (gültige Werte z.B. ALL, BEGIN, END, FAIL oder REQUEUE)
#SBATCH --mail-user=kremlingph95027@th-nuernberg.de # Emailadresse für Statusmails

echo "=================================================================="
echo "Starting Batch Job at $(date)"
echo "Job submitted to partition ${SLURM_JOB_PARTITION} on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "Requested ${SLURM_CPUS_ON_NODE} CPUs on compute node $(hostname)"
echo "Working directory: $(pwd)"
echo "=================================================================="

###################### Optional for Pythonnutzer*innen #######################
# Die folgenden Umgebungsvariablen stellen sicher, dass
# Modellgewichte von Huggingface und PIP Packages nicht unter 
# /home/$USER/.cache landen. 
CACHE_DIR=/nfs/scratch/students/$USER/.cache
export PIP_CACHE_DIR=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
mkdir -p "$CACHE_DIR"

export LD_LIBRARY_PATH=/nfs/tools/spack/v0.21.0/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/cuda-11.8.0-y3u5n3kohg7mgnff4loe6t2pz6awb7ck/lib64/:$LD_LIBRARY_PATH
export PATH=/nfs/tools/spack/v0.21.0/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/cuda-11.8.0-y3u5n3kohg7mgnff4loe6t2pz6awb7ck/bin/:$PATH
########################################################

module purge
module load cuda/cuda-11.8.0
source /nfs/scratch/students/$(whoami)/venv39/bin/activate
cd /home/$(whoami)/api

############### Starte eigenen Job hier ################
srun python /home/$(whoami)/api/Controller.py