import os.path
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers.schedulers import DDPMScheduler
from diffusers.models import UNet2DModel
import matplotlib.pyplot as plt
import wandb
import argparse
from distutils.util import strtobool
import torch.nn.functional as F
from pathlib import Path


def show(imgs):
    fig, axs = plt.subplots(ncols=4, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        # img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,)
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'))
    parser.add_argument('--wandb-project', type=str, default='hands_on_diffusion')
    parser.add_argument('--wandb-entity', type=str, default='Slientea98')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--cache-path', type=str, default='/home/yibo/.cache/huggingface/datasets')
    parser.add_argument('--huggan-dataset', type=str, default='huggan/smithsonian_butterflies_subset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,)
    parser.add_argument('--num-samples', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # # prepare the u-net
    model = UNet2DModel(
        in_channels=3,
        sample_size=64,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              beta_start=0.001,
                              beta_end=0.02, )

    device = torch.device(args.device)
    model.to(device)

    if not args.sample_mode:
        dataset = load_dataset(args.huggan_dataset, split='train', cache_dir=args.cache_path)
        image_size = args.image_size
        # # preprocess data
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        def transform(samples):
            images = [preprocess(image.convert('RGB')) for image in samples['image']]
            return {'image': images}

        dataset.set_transform(transform)

        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        # # visualize some samples for the dataset
        # batch = next(iter(train_dataloader))
        # print(batch['image'][:8].shape, )
        # show(batch['image'][:8]*0.5 + 0.5)
        # plt.show()

        # # visualize the corrupted images
        # batch = next(iter(train_dataloader))
        # time_steps = torch.linspace(0, 999, 8).long()   # should specify time steps for each sample
        # noise = torch.randn_like(batch['image'][:8], dtype=torch.float32)
        # noised_x = scheduler.add_noise(batch['image'][:8], noise, time_steps)
        # show((noised_x*0.5 + 0.5).clip(0., 1.))
        # plt.show()

        # log with wandb
        if args.track:
            wandb.init(project=args.wandb_project,
                       entity=args.wandb_entity,
                       name=args.exp_name + str(args.seed),
                       config=vars(args))

        # # step into the training iteration
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.num_epochs):
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                x = batch['image'].to(device)
                noise = torch.randn_like(x, dtype=torch.float32).to(device)
                # sample a random time step for each sample in the mini-batch
                time_steps = torch.randint(low=0, high=scheduler.num_train_timesteps,
                                           size=(x.shape[0],), device=device).long()
                noised_x = scheduler.add_noise(x, noise, time_steps)
                # take in the corrupted image but predict the noise
                noise_pred = model(noised_x, time_steps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                if args.track:
                    wandb.log({'train/loss': loss.item()})
                print(f'epoch: {epoch}, loss: {loss.item()}')

        # save the model for later use or evaluation
        work_dir = Path().cwd() / 'logs' / 'valina_diffusion' / str(args.seed)
        if not work_dir.exists():
            work_dir.mkdir(parents=True)
        torch.save(model.state_dict(), work_dir / 'model.pt')

    # # sample from the optimized model
    if args.sample_mode:
        model_dir = Path().cwd() / 'logs' / 'valina_diffusion' / str(args.seed) / 'model.pt'
        assert model_dir.exists()
        model_state_dict = torch.load(model_dir)
        model.load_state_dict(model_state_dict)

    sample = torch.randn((args.num_samples, 3, args.image_size, args.image_size)).to(device)
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = model(sample, t).sample

        sample = scheduler.step(noise_pred, t, sample).prev_sample
    show(sample.clip(-1, 1) * 0.5 + 0.5)
    plt.show()
