import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def kernel_clip(y: torch.tensor, i: int, j: int, size_clip=3, inverse=True):
    """
    Clip image patch.
    """
    if size_clip % 2 == 0:
        raise Exception('Size of clip should be odd.')
    if not inverse:
        H, W = y.shape[0], y.shape[1]
    else:
        H, W = y.shape[-1], y.shape[1]
    x_end, y_end = i+size_clip//2+1, j+size_clip//2+1
    if x_end > H or y_end > W or x_end < size_clip or y_end < size_clip:
        print('Warning: index exceed the margin! x_end={}, y_end={}'.format(x_end, y_end))
    x_end = np.clip(x_end, size_clip, H)
    y_end = np.clip(y_end, size_clip, W)
    x_start, y_start = x_end-size_clip, y_end-size_clip

    if not inverse:
        return y[x_start:x_end, y_start:y_end]
    else:
        return y[:, y_start:y_end, x_start:x_end]


class MyDataset(Dataset):
    def __init__(self, data, label=None):
        self.train = torch.tensor(data).float()
        self.label = torch.tensor(label).type(torch.int64)

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, index):
        return self.train[index].cuda(), self.label[index]


class SscatAE(nn.Module):
    def __init__(self, L, P):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
        )
        self.SPATIAL_LEN = 40
        self.fc_conv = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, self.SPATIAL_LEN),
            nn.Tanh(),
        )
        self.fc_band = nn.Sequential(
            nn.Linear(L, 128),
            nn.Tanh(),
            nn.Linear(128, 100-self.SPATIAL_LEN),
            nn.Tanh(),
        )
        self.fc_cat = nn.Sequential(
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, P),
            nn.Softmax(dim=1),
        )
        self.de_main = nn.Linear(P, L, bias=False)

    def forward(self, x):
        line1 = self.conv(x)
        line1 = line1.view(-1, 128)
        line1 = self.fc_conv(line1)
        line2 = self.fc_band(x[:, :, 1, 1])

        cat = torch.cat([line1, line2], dim=1)
        code = self.fc_cat(cat)
        output = self.de_main(code)
        return code, output


class SSCAT:
    def __init__(self, height, width, sparsity=1e-3):
        self.height = height
        self.width = width
        self.sparsity = sparsity
        self.cnt = 0

    def _load_weight(self, model, weight_name, edm):
        if type(edm) == np.ndarray:
            edm = torch.from_numpy(edm)
        model_dict = model.state_dict()
        model_dict[weight_name] = edm
        model.load_state_dict(model_dict)

    def _decay_lr(self, optim, decay_percent=0.95, decay_rate=100):
        if self.cnt % decay_rate == 0:
            for para_grp in optim.param_groups:
                para_grp['lr'] = para_grp['lr'] * decay_percent

    @staticmethod
    def _set_non_negative(model, weight_name, smaller_one=False):
        assert weight_name in model.state_dict().keys()
        weight = model.state_dict()[weight_name]
        weight[weight < 0.] = 1e-6
        if smaller_one:
            weight[weight > 1.] = 1.
        model.state_dict()[weight_name].copy_(weight)

    def _convert_input(self, y):
        L, N = y.shape
        label = np.arange(y.shape[-1])
        data = y.T.copy()  # shape (N, L)
        data_3d = data.reshape(self.height, self.width, -1)
        data_3d = torch.tensor(data_3d.T).float()
        data_3d = F.pad(data_3d, [1, 1, 1, 1])
        data = torch.zeros(N, L, 3, 3)
        for i in range(N):
            row, col = i // self.width + 1, i % self.width + 1
            data[i] = kernel_clip(data_3d, row, col)
        dataset = MyDataset(data=data, label=label)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def _switch_freeze(self, model, layer):
        for name, para in model.named_parameters():
            if name in layer:
                para.requires_grad = not para.requires_grad

    def _archive_abd(self, code):
        code = code.cpu().data.numpy()
        code = np.squeeze(code)
        if len(code.shape) == 3:
            code = code.reshape(code.shape[0], self.height*self.width)
        else:
            code = code.T
        return code

    def fit(self, m: np.array, y: np.array, max_iter: int = 100):
        """Initialization Stage"""
        P, L, N = m.shape[1], y.shape[0], y.shape[1]
        model = SscatAE(L, P).cuda()
        name_edm = 'de_main.weight'

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self._load_weight(model, name_edm, m)  # initialize endmember

        x = self._convert_input(y)  # create

        """Training Process"""
        self._switch_freeze(model, [name_edm])
        for i in tqdm(range(max_iter)):
            if i == int(max_iter * 0.9):
                self._switch_freeze(model, [name_edm])  # freeze for fine-tune

            for j, (in_x, _) in enumerate(x):
                code, re_x = model(in_x)

                cos_theta = torch.sum(re_x * in_x[:, :, 1, 1], dim=1) / \
                            (torch.norm(re_x, dim=1, p=2) * torch.norm(in_x[:, :, 1, 1], dim=1, p=2))
                loss = torch.mean(1 - cos_theta**2)  # squared sine distance

                loss += self.sparsity *\
                        torch.mean(torch.sum(torch.sqrt(torch.abs(code)), dim=-1))  # sparsity regularization

                opt.zero_grad()
                loss.backward()
                opt.step()

                self.cnt += 1
                self._set_non_negative(model, name_edm, smaller_one=True)
                self._decay_lr(opt, 0.95)
  
        """Estimation Process"""
        rec = torch.zeros(N, L).cuda()
        abd = torch.zeros(N, P).cuda()
        model.eval()
        for j, (in_x, num_x) in enumerate(x):
            output = model(in_x)
            abd[num_x] = output[0]
            rec[num_x] = output[1]

        """Return Results"""
        return self._archive_abd(abd)


if __name__ == '__main__':

    # fit the real data
    P = 4
    im = np.load('data/jasperRidge_R198.npy').reshape(100*100, 198).T
    edm = np.load('data/edm_extracted.npy').T
    model = SSCAT(100, 100, sparsity=0.004)
    abd = model.fit(edm, im, max_iter=50).reshape((P, 100, 100))
    for i in range(P):
        plt.subplot(1, P, i+1)
        plt.imshow(abd[i], cmap=plt.cm.jet)
        plt.axis('off')
    plt.savefig(f'abundance-real')
    plt.show()

    # fit the synthetic data
    P = 5
    im = np.load('data/syn.npy').reshape(70*70, 200).T
    edm = np.load('data/syn_edm.npy').T
    model = SSCAT(70, 70, sparsity=1e-5)
    abd = model.fit(edm, im, max_iter=50).reshape((P, 70, 70))
    for i in range(P):
        plt.subplot(1, P, i+1)
        plt.imshow(abd[i], cmap=plt.cm.jet)
        plt.axis('off')
    plt.savefig(f'abundance-syn')
    plt.show()



