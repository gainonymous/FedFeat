import sys
import logging
import copy
import torch
from utils import factory
from utils.fedavg import fed_avg_experiment, fed_avg_fedfeat_experiment
from utils.fedprox import fedprox_experiment, fedprox_fedfeat_experiment
from utils.moon import moon_experiment, moon_fedfeat_experiment
from utils.fedma import fedma_experiment, fedma_fedfeat_experiment
from utils.scaffold import scaffold_experiment, scaffold_fedfeat_experiment
from utils.fedopt import fedopt_experiment, fedopt_fedfeat_experiment
from utils.fednova import fednova_experiment, fednova_fedfeat_experiment
from utils.feddyn import feddyn_experiment, feddyn_fedfeat_experiment

import os
from utils.util import num_params
from utils.data_loader import get_dataset_by_name, inspect_client_distribution
import numpy as np

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    #  Get total GPUs available
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
   # Automatically use GPU if available, else fallback to CPU
    if args['device'] == "0":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args['device'] == "1":
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif args['device'] == "2":
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    for seed in seed_list:
        args["seed"] = seed
        _train(args, device)


def _train(args, device):

    logs_name = "logs/{}/{}/{}/{}".format(args["method_name"], args["dataset"], args["seed"], args["convnet_type"])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}".format(
        args["method_name"],
        args["dataset"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    #_set_random()
    # _set_device(args)
    print_args(args)
    if args["method_name"] == "MOON":
        model = factory.get_model(args["convnet_type"], fedfeat=True, args=args)
    else:
        model = factory.get_model(args["convnet_type"], args["FedFeat"], args)
    
    logging.info("All params: {}".format(num_params(model)))
    logging.info(
        "Trainable params: {}".format(num_params(model))
    )
    
    output_dir = "./output/"  # or any path you want
    iid_client_train_loader, noniid_client_train_loader, test_loader = get_dataset_by_name(args['dataset'])
    inspect_client_distribution(args['dataset'], iid_client_train_loader, noniid_client_train_loader)
    
    # FedAvg-----------------------------------
    if args['method_name'] == "FedAvg":
        if not args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedAvg_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fed_avg_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedAvg_noniid_{args['num_clients_per_round']}")

                        # Run experiment
            acc = fed_avg_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

        elif args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedAvg_FedFeat_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fed_avg_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedAvg_FedFeat_noniid_{args['num_clients_per_round']}")

            
            # Run experiment
            acc = fed_avg_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

    elif args['method_name'] == "FedProx":
        if not args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedProx_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fedprox_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedProx_noniid_{args['num_clients_per_round']}")

                        # Run experiment
            acc = fedprox_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

        elif args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedProx_FedFeat_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fedprox_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedProx_FedFeat_noniid_{args['num_clients_per_round']}")

            
            # Run experiment
            acc = fedprox_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

    elif args['method_name'] == "MOON":
        if not args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_MOON_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = moon_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_MOON_noniid_{args['num_clients_per_round']}")

                        # Run experiment
            acc = moon_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

        elif args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_MOON_FedFeat_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = moon_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_MOON_FedFeat_noniid_{args['num_clients_per_round']}")

            
            # Run experiment
            acc = moon_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

    elif args['method_name'] == "FedMa":
        if not args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedMa_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fedma_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedMa_noniid_{args['num_clients_per_round']}")

                        # Run experiment
            acc = fedma_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

        elif args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedMa_FedFeat_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fedma_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedMa_FedFeat_noniid_{args['num_clients_per_round']}")

            
            # Run experiment
            acc = fedma_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

    elif args['method_name'] == "FedOpt":
        if not args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedOpt_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fedopt_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedOpt_noniid_{args['num_clients_per_round']}")

                        # Run experiment
            acc = fedopt_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

        elif args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedOpt_FedFeat_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fedopt_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedOpt_FedFeat_noniid_{args['num_clients_per_round']}")

            
            # Run experiment
            acc = fedopt_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

    elif args['method_name'] == "FedNova":
        if not args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedNova_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fednova_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedNova_noniid_{args['num_clients_per_round']}")

                        # Run experiment
            acc = fednova_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

        elif args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedNova_FedFeat_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = fednova_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedNova_FedFeat_noniid_{args['num_clients_per_round']}")

            
            # Run experiment
            acc = fednova_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

    elif args['method_name'] == "FedDyn":
        if not args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedDyn_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = feddyn_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedDyn_noniid_{args['num_clients_per_round']}")

                        # Run experiment
            acc = feddyn_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

        elif args['FedFeat']:
            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedDyn_FedFeat_iid_{args['num_clients_per_round']}")
            
            # Run experiment
            acc = feddyn_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = iid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

            # Clone model
            model_copy = copy.deepcopy(model)
            # Build output path
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{args['dataset']}_acc_{args['convnet_type']}_FedDyn_FedFeat_noniid_{args['num_clients_per_round']}")

            
            # Run experiment
            acc = feddyn_fedfeat_experiment(
                model_copy,
                num_clients_per_round=args["num_clients_per_round"],
                num_local_epochs=args["num_local_epochs"],
                lr=args["lr"],
                client_train_loader = noniid_client_train_loader,
                max_rounds=args["max_rounds"],
                filename=filename,
                test_loader = test_loader,
                device=device,
                num_global_epochs = args["num_global_epochs"],
                global_lr = args["global_lr"]
            )
            print(f"[{filename}] Accuracy:\n{acc}")
            np.save(filename + ".npy", acc)

    else:
        print("Unknown Method")


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
