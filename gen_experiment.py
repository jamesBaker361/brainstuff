for pairing in ["unpaired","paired"]:
    for translate in ["trans","untrans"]:
        for recons in ["recons","unrecons"]:
            for disc in ["disc","no_disc"]:
                name=f"{pairing}_{translate}_{recons}_{disc}"
                command=f" sbatch -J cycle --err=slurm_chip/cycle/{name}.err --out=slurm_chip/script_test/{name}.out runpygpu_chip_L40S.sh "
                command+=" --epochs 250 --validation_interval 10 --project_name cycle_sub1 --sublist 1 "
                if pairing=="paired":
                    command+=" --unpaired_image_dataset nouman-10/wikiart_testing "
                if translate=="trans":
                    command+=" --translation_loss "
                if recons=="recons":
                    command+=" --reconstruction_loss "
                if disc=="disc":
                    command+=" --use_discriminator "
                if pairing=="unpaired" and translate=="untrans" and disc=="no_disc":
                    pass
                else:
                    print(command)
        print("")