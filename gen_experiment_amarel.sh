for pairing in ["unpaired","paired"]:
    for translate in ["trans","untrans"]:
        for recons in ["recons","unrecons"]:
            for disc in ["disc","no_disc"]:
                name=f"{pairing}_{translate}_{recons}_{disc}"
                command=f" sbatch -J cycle --err=slurm_chip/cycle/{name}.err --out=slurm_chip/cycle/{name}.out --exclude=gpu[005,006,010-014,017,018],cuda[001-008],pascal[001-010] runpygpu.sh "
                command+=" cycle_training.py --epochs 1000 --validation_interval 50 --project_name cycle_sub1 --sublist 1 --fmri_type array "
                command+=" --gradient_accumulation_steps 16 "
                if pairing=="paired":
                    command+=" --unpaired_image_dataset nouman-10/wikiart_testing "
                if translate=="trans":
                    command+=" --translation_loss "
                if recons=="recons":
                    command+=" --reconstruction_loss "
                if disc=="disc":
                    command+=" --use_discriminator "
                if pairing=="paired" and translate=="untrans" and disc=="no_disc":
                    pass
                else:
                    print(command)
        print("")