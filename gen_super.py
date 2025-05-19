for sub in [1,2]:
    for res in [3,4]:
        for epochs in [4000]:
            name=f"sub_{sub}_{epochs}_res_{res}"
            command=f"sbatch -J super --out=slurm/super/{name}.out --err=slurm/super/{name}.err runpygpu.sh "
            command+=f" super_res_training.py --epochs {epochs} --gradient_accumulation_steps 8 --project_name super "
            command+=f" --sublist {sub} --residual_blocks {res} --mixed_precision bf16 --name {name} "
            print(command)