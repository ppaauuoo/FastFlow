# uv run python main.py -cfg configs/resnet18.yaml --data ./data/dataset/ -cat screw --eval -ckpt _fastflow_experiment_checkpoints/w-overkill/499.pt 
uv run python main.py -cfg configs/resnet18.yaml --data ./data/dataset/ -cat inspection --eval -ckpt _fastflow_experiment_checkpoints/pass-only/499.pt 
