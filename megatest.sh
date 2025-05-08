sbatch --err=slurm_chip/cycle_test/voxel.err --out=slurm_chip/cycle_test/voxel.out runpygpu_chip.sh cycle_training.py
sbatch --err=slurm_chip/cycle_test/voxelfp16.err --out=slurm_chip/cycle_test/voxelfp16.out runpygpu_chip.sh cycle_training.py --mixed_precision fp16
sbatch --err=slurm_chip/cycle_test/voxel_disc.err --out=slurm_chip/cycle_test/voxel_disc.out runpygpu_chip.sh cycle_training.py --use_discriminator

sbatch --err=slurm_chip/cycle_test/array.err --out=slurm_chip/cycle_test/array.out runpygpu_chip.sh cycle_training.py --fmri_type array --kernel_size 8 --n_layers 3
sbatch --err=slurm_chip/cycle_test/arrayfp16.err --out=slurm_chip/cycle_test/arrayfp16.out runpygpu_chip.sh cycle_training.py --mixed_precision fp16 --fmri_type array --kernel_size 8 --n_layers 3
sbatch --err=slurm_chip/cycle_test/array_disc.err --out=slurm_chip/cycle_test/array_disc.out runpygpu_chip.sh cycle_training.py --use_discriminator --fmri_type array --kernel_size 8 --n_layers 3