for sub in [1,2]:
    for res in [3,4]:
        for epochs in [2000]:
            name=f"sub_{sub}_{epochs}_res_{res}"
            command=f"sbatch -J super --out=slurm_chip/super/{name}.out --err=slurm_chip/super/{name}.err runpygpu_chip_L40S.sh "
            command+=f" super_res_training.py --epochs {epochs} --gradient_accumulation_steps 8 --project_name super "
            command+=f" --sublist {sub} --residual_blocks {res} --mixed_precision bf16 --save_path {name} "
            print(command)