# uv run python main.py -cfg configs/resnet18.yaml --data ./data/dataset/ -cat screw --validate -ckpt _fastflow_experiment_checkpoints/w-overkill/499.pt 
# uv run python main.py -cfg configs/resnet18.yaml --data ./data/dataset/ -cat screw --validate -ckpt _fastflow_experiment_checkpoints/no-overkill/499.pt 
uv run python main.py -cfg configs/resnet18.yaml --data ./data/dataset/ -cat inspection --validate -ckpt _fastflow_experiment_checkpoints/pass-only/499.pt 
