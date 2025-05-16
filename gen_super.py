for sub in [1,2,5,7]:
    for res in [0,1,2]:
        name=f"sub_{sub}_res_{res}"
        command=f"sbatch -J super --out=slurm/super/{name}.out --err=slurm/super/{name}.err --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010],gpuk[001-012] runpygpu.sh "
        command+=" super_res_training.py --epochs 250 --gradient_accumulation_steps 8 --project_name super "
        command+=f" --sublist {sub} --residual_blocks {res} --mixed_precision bf16 "
        print(command)