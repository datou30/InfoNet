import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
from data import gen_train_data
from model.decoder import Decoder
from model.encoder import Encoder
from model.infonet import infonet
from model.query import Query_Gen_transformer
from scipy.stats import rankdata
from tensorboardX import SummaryWriter
from torch import nn, optim
from sklearn.mixture import GaussianMixture
import pickle

#import tensorrt as trt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from data import gen_train_data, gen_gauss_xyz
from model.decoder import Decoder 
from model.encoder import Encoder 
from model.infonet import infonet
from model.query import Query_Gen_transformer
from scipy.stats import rankdata
from torch.optim import Adam
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from send2trash import send2trash
import hydra
from hydra.core.hydra_config import HydraConfig
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.backends.cuda.enable_flash_sdp(False)


learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InfoNetDataset(torch.utils.data.Dataset):
    def __init__(self, total_epoch):
        super().__init__()
        self.total_epoch = total_epoch

    def __len__(self):
        return self.total_epoch

    def __getitem__(self, idx):
        #res = gen_train_data(batchsize=1, seq_len=seq_len, dim=1, max_num_components=20).squeeze(0)
        res = gen_train_data(batchsize=1, seq_len=seq_len, dim=1, com_num=20).squeeze(0)
        return res


class LightningWrapper(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        encoder = Encoder(
            input_dim=input_dim,
            latent_num=latent_num,
            latent_dim=latent_dim,
            cross_attn_heads=8,
            self_attn_heads=16,
            num_self_attn_per_block=8,
            num_self_attn_blocks=1,
        )

        decoder = Decoder(
            q_dim=decoder_query_dim,
            latent_dim=latent_dim,
        )

        query_gen = Query_Gen_transformer(
            input_dim = input_dim,
            dim = decoder_query_dim,
        )
        self.model = infonet(encoder=encoder, decoder=decoder, query_gen=query_gen,
                                 decoder_query_dim=decoder_query_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mi_lb = self(batch)
        loss = -torch.mean(mi_lb)
        self.logger.experiment.add_scalars('train_loss', 
                                           {f'dim{input_dim//2}': loss,
                                            },
                                           global_step=self.global_step)
        self.log('loss', loss, on_step=True, prog_bar=True, sync_dist=True, batch_size=batchsize)
        return loss

    def validation_step(self, batch, batch_idx):
        
        dim_list = [1, 2, 5, 10]
        for dim in dim_list:
            eval_acc = gmm_order_eval(self.model, dim, seq_len=2000, number_test=500, iter=self.current_epoch + 1)
            self.logger.experiment.add_scalars('val_acc', 
                                {f'dim{dim}': eval_acc}, 
                                global_step=self.global_step)
        self.log(f'val_acc', eval_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return eval_acc

    def test_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=8000, gamma=0.9)
        #scheduler = CosineAnnealingLR(optimizer, T_max=40000, eta_min=0)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    def on_load_checkpoint(self, checkpoint):
        if "optimizer_states" in checkpoint:
            for state in checkpoint["optimizer_states"]:
                for param_group in state['param_groups']:
                    print("Loaded learning rate:", param_group['lr'])
                    param_group['lr'] = learning_rate

    def train_dataloader(self):
        dataset = InfoNetDataset(total_epoch=6400)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        return dataloader

    def test_dataloader(self):
        return torch.utils.data.DataLoader(InfoNetDataset(total_epoch=1), batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(InfoNetDataset(total_epoch=1), batch_size=1)
    
def compare(list_a, list_b):
    return [1 if a >= b else 0 for a, b in zip(list_a, list_b)]
    
def gmm_order_eval(module, dim, iter, seq_len=2000, number_test=500, reg=0.1):
    
    module.eval()
    path = "/home/hzy/work/kNN/InfoNet-main-V2/num_compare"

    full_path = os.path.join(path, f'num_{dim}')
    MI_XY = []
    MI_XZ = []
    results_xy = []
    results_xz = []

    with open(os.path.join(full_path, "weight.pkl"), 'rb') as f:
        weights = pickle.load(f)

    with open(os.path.join(full_path, "mean.pkl"), 'rb') as f:
        means = pickle.load(f)

    with open(os.path.join(full_path, "cov.pkl"), 'rb') as f:
        covs = pickle.load(f)

    with open(os.path.join(full_path, "XY_mi_gauss.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = float(line.strip())
            MI_XY.append(value)

    with open(os.path.join(full_path, "XZ_mi_gauss.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = float(line.strip())
            MI_XZ.append(value)

    for idx in range(number_test):
            
        if idx>=len(weights):
            continue

        gm = GaussianMixture(n_components=5)
        gm.weights_ = np.array(weights[idx])
        gm.means_ = np.array(means[idx])
        gm.covariances_ = np.array(covs[idx])
            
        data, labels = gm.sample(n_samples=seq_len)

        data = np.array(data)
        for row in range(3):
            data[:, row] = rankdata(data[:, row])/seq_len

        data_xy = torch.from_numpy(data[:, [0, 1]]).unsqueeze(0).type(torch.float32).to(device)
        data_xz = torch.from_numpy(data[:, [0, 2]]).unsqueeze(0).type(torch.float32).to(device)
        with torch.no_grad():
            infer_1 = module(data_xy).cpu().numpy()
            infer_2 = module(data_xz).cpu().numpy()
            
        results_xy.append(infer_1)
        results_xz.append(infer_2)
        
    comp = compare(MI_XY[:number_test], MI_XZ[:number_test])
    comp_c = compare(results_xy, results_xz)

    accuracy_c = 0
    for j in range(len(comp_c)):
        if comp[j] == comp_c[j]:
            accuracy_c+=1
    acc_rate = accuracy_c / len(comp) * 100

    save_text_root_path = os.path.join(logger.log_dir, "gauss_order")
    text_path = os.path.join(save_text_root_path, f"dim-{dim}")
    if not os.path.exists(text_path):
        os.makedirs(text_path, exist_ok=True)
    with open(os.path.join(text_path, f"model-firsttry-{seq_len}.txt"), "a") as f:
        output_string = f"order test acc on {dim}-dimension gauss of model_{iter} is {acc_rate}\n"
        f.write(output_string)
        print(output_string)

    fig_root_path = os.path.join(logger.log_dir, 'figure')
    os.makedirs(fig_root_path, exist_ok=True)
    fig_save_cat_path = os.path.join(fig_root_path, f"{dim}")
    os.makedirs(fig_save_cat_path, exist_ok=True)

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(121)
    d_draw = np.arange(number_test)

    zipped_lists = sorted(zip(MI_XY[:number_test], results_xy[:number_test]), reverse=False)
    sorted_list1, sorted_list2 = zip(*zipped_lists)
    ax1.plot(d_draw[:50], MI_XY[:50], color="red", lw=2, ls="-", label="real mutual information XY", markersize=10)
    ax1.plot(d_draw[:50], results_xy[:50], color="blue", lw=2, ls="-", label="estimate MI XY", markersize=10)
    ax1.set_xlabel("# experiment times", fontweight="bold", fontsize=20)
    ax1.set_ylabel(" mutual information ", fontweight="bold", fontsize=20)
    ax1.legend(fontsize=20, loc="upper left")

    ax2 = fig.add_subplot(122)
    ax2.plot(d_draw, sorted_list1, color="red", lw=2, ls="-", label="real mutual information XY", markersize=10)
    ax2.plot(d_draw, sorted_list2, color="blue", lw=2, ls="-", label="estimate MI XY", markersize=10)
    ax2.set_xlabel("# experiment times", fontweight="bold", fontsize=20)
    ax2.set_ylabel(" mutual information ", fontweight="bold", fontsize=20)
    ax2.legend(fontsize=20, loc="upper left")

    image_save_path = os.path.join(fig_save_cat_path, f"gauss-dim{dim}-iter{iter}.png")
    plt.savefig(image_save_path)

    return acc_rate

@hydra.main(config_path='configs', config_name='cfg_train_noflash', version_base='1.1')
def main(cfg):
    ma_rate = 1.0
    global_step = 0

    global latent_dim, latent_num, input_dim, batchsize, seq_len, decoder_query_dim, max_input_dim
    batchsize = cfg.batchsize
    latent_dim = cfg.latent_dim
    latent_num = cfg.latent_num
    input_dim = cfg.input_dim
    seq_len = cfg.seq_len
    decoder_query_dim = cfg.decoder_query_dim

    torch.set_float32_matmul_precision('medium')
    module = LightningWrapper()

    global logger
    logger = TensorBoardLogger('logs', name=cfg.name, version=cfg.version)
    #if os.path.isdir(logger.log_dir):
        #send2trash(logger.log_dir)
        
    trainer = lightning.Trainer(max_epochs=1500,
                                accelerator='auto',
                                devices=[2,3],
                                logger=logger,
                                num_sanity_val_steps=0,
                                strategy='ddp_find_unused_parameters_true',
                                gradient_clip_val=1.0,
                                check_val_every_n_epoch=10,
                                # limit_val_batches=0,  # Disable validation for now as the script is not correct.
                                callbacks=[
                                    ModelCheckpoint(
                                        monitor='val_acc',
                                        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
                                        filename='basic_hypernet-{epoch:02d}-{val_acc:.2f}',
                                        save_top_k=100,
                                        mode='max',
                                        every_n_epochs=10,
                                        save_last=True
                                    )
                                ]
                                )
    trainer.fit(module)

if __name__ == '__main__':
    main()
