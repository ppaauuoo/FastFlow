import argparse
import os

import torch
import yaml
from ignite.contrib import metrics
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constants as const
import dataset
import fastflow
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

def build_validate_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
        return_filename=True,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.to(device)
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))

def validate_once(dataloader, model):
    model.eval()
    outputs_list = []
    filenames_list = []
    for data, filenames in dataloader:
        data = data.to(device)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        for i in range(outputs.shape[0]):
            outputs_list.append(outputs[i, 0])
            filenames_list.append(filenames[i])
    return outputs_list, filenames_list


def plot_outputs(outputs_list, output_dir, prefix="validate", filenames_list=None):
    os.makedirs(output_dir, exist_ok=True)

    for i, outputs in enumerate(outputs_list):
        img = outputs.numpy()
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="hot", interpolation="nearest")
        plt.colorbar()
        if filenames_list:
            filename = filenames_list[i]
            plt.title(f"{prefix} - {filename}")
            save_name = os.path.splitext(filename)[0] + ".png"
        else:
            plt.title(f"{prefix} image {i}")
            save_name = f"{prefix}_img_{i:03d}.png"
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, save_name), dpi=150)
        plt.close(fig)

    n = len(outputs_list)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, outputs in enumerate(outputs_list):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        img = outputs.numpy()
        im = ax.imshow(img, cmap="hot", interpolation="nearest")
        if filenames_list:
            ax.set_title(os.path.splitext(filenames_list[idx])[0], fontsize=8)
        else:
            ax.set_title(f"img {idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_combined.png"), dpi=150)
    plt.close(fig)

def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.to(device)
    print(f"Using device: {device}")

    for epoch in tqdm(range(const.NUM_EPOCHS)):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.to(device)
    print(f"Using device: {device}")
    eval_once(test_dataloader, model)

def validate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # print(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    validate_dataloader = build_validate_data_loader(args, config)
    model.to(device)
    print(f"Using device: {device}")
    outputs_list, filenames_list = validate_once(validate_dataloader, model)
    plot_dir = os.path.join(os.path.dirname(args.checkpoint) or ".", "validate_plots")
    plot_outputs(outputs_list, plot_dir, prefix=f"validate_{args.category}", filenames_list=filenames_list)

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    parser.add_argument("--validate", action="store_true", help="run validate only")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    elif args.validate:
        validate(args)
    else:
        train(args)
