"""
Sei architecture
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.interpolate import splev
from typing import Tuple

from .sei_globals import SEI_WINDOW, SEI_STEP, SEI_TARGETS, SEI_CLASSES
from .seq_utils import one_hot_encode, rev_compl
from .dataset import SubSequenceDataset

def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.

    """

    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d"
                  % (order - (1 - intercept)))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])

    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))

    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * ( 2*(dist>0)-1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist),np.max(dist),n_knots+2)[1:-1]
        return torch.from_numpy(bs(
            dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(
            dist, df=df, intercept=True)).float()



class BSplineTransformation(nn.Module):

    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.to(input.device)
        
        return  torch.matmul(input, self._spline_tr)



class BSplineConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, degrees_of_freedom, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, log=False, scaled = True):
        super(BSplineConv1D, self).__init__()
        self._df = degrees_of_freedom
        self._log = log
        self._scaled = scaled

        self.spline = nn.Conv1d(1, degrees_of_freedom, kernel_size, stride, padding, dilation,
            bias=False)
        self.spline.weight = spline_factory(kernel_size, self._df, log=log).view(self._df, 1, kernel_size)
        if scaled:
            self.spline.weight = self.spline.weight / kernel_size            
        self.spline.weight = nn.Parameter(self.spline.weight)
        self.spline.weight.requires_grad = False
        self.conv1d = nn.Conv1d(in_channels * degrees_of_freedom, out_channels, 1, 
            groups = groups, bias=bias)

    def forward(self, input):
        batch_size, n_channels, length = input.size()
        spline_out = self.spline(input.view(batch_size * n_channels,1,length))
        conv1d_out = self.conv1d(spline_out.view(batch_size, n_channels * self._df,  length))
        return conv1d_out


class Sei(nn.Module):
    def __init__(self, sequence_length=SEI_WINDOW, n_genomic_features=SEI_TARGETS):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(Sei, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))

        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))

        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True))

        self._spline_df = int(128/8)        
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5),
            BSplineTransformation(self._spline_df, scaled=False))

        self.classifier = nn.Sequential(
            nn.Linear(960 * self._spline_df, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features))
            #nn.Sigmoid()) # removed sigmoid to allow model to predict logits

    def forward(self, x, return_logits=False):
        """Forward propagation of a batch.
        """
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        
        spline_out = self.spline_tr(out)
        reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)
        logits = self.classifier(reshape_out)

        if return_logits:
            return logits
        return torch.sigmoid(logits)
    
    def seq_predict(self, seq: str, reverse_aug: bool = True) -> np.ndarray:
        sequence = one_hot_encode(seq, dtype=torch.float32)
        device = get_device(self)
        
        with torch.inference_mode():
            preds = self(sequence.unsqueeze(0).to(device))
        
        if reverse_aug:
            rev_seq = rev_compl(seq)
            rev_sequence =  one_hot_encode(rev_seq, dtype=torch.float32)
            with torch.inference_mode():
                preds_rev = self(rev_sequence.unsqueeze(0).to(device))
            preds = (preds + preds_rev) / 2
        
        return preds.cpu().numpy()

    def seq_sliding_predict(self,
                            seq: str, 
                            reverse_aug: bool = True,
                            window_size: int = SEI_WINDOW,
                            step: int = SEI_STEP,
                            batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:

        dataset = SubSequenceDataset(seq, window_size=window_size, step=step, reverse_aug=reverse_aug)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        offsets = []
        device = get_device(self)

        with torch.inference_mode():
            if reverse_aug:
                for X, rev_X, offs in dataloader:
                    X = X.to(device)
                    rev_X = rev_X.to(device)
                    pr = self(X).cpu().numpy()
                    pr_rev = self(rev_X).cpu().numpy()
                    pr = (pr + pr_rev) / 2
                    preds.append(pr)
                    offsets.append(offs.numpy())
            else:
                for X, offs in dataloader:
                    X = X.to(device)
                    pr = self(X).cpu().numpy()
                    preds.append(pr)
                    offsets.append(offs.numpy())
        
        preds = np.concatenate(preds)
        offsets = np.concatenate(offsets)
        return preds, offsets 

def get_device(m: torch.nn.Module) -> torch.device:
    return next(m.parameters()).device

class SeiProjector:
    """
    Project model predictions to get 40 high-order classes derived by Sei authors 
    """
    def __init__(self, weights: str, n_classes: int = SEI_CLASSES):
        self.n_classes = n_classes
        self.weights =  np.load(weights)
        self.norm = np.linalg.norm(self.weights, axis=1)
    
    def project(self, preds):
        return (np.dot(preds, self.weights.T) / self.norm)[:, :self.n_classes]
    
    def __call__(self, preds):
        return self.project(preds)
        
class SeiNormalizer:
    """
    Normalize model predictions by nucleosome occupance as recommended
    by Sei authors 
    """
    def __init__(self, histone_inds: str):
        self.histone_inds = np.load(histone_inds)
    
    def normalize(self, preds_ref, preds_alt):
        preds_ref_adjust = preds_ref.copy()
        preds_alt_adjust = preds_alt.copy()

        sum_ref = np.sum(preds_ref_adjust[:, self.histone_inds], axis=1)
        sum_alt = np.sum(preds_ref_adjust[:, self.histone_inds], axis=1)

        pred_avg = (sum_ref + sum_alt) / 2
         
        norm_ref = (pred_avg / sum_ref)[:, None]
        norm_alt = (pred_avg / sum_alt)[:, None]
        preds_ref_adjust[:, self.histone_inds] = preds_ref_adjust[:, self.histone_inds] * norm_ref
        preds_alt_adjust[:, self.histone_inds] = preds_alt_adjust[:, self.histone_inds] * norm_alt

        return preds_ref_adjust, preds_alt_adjust
    
    def __call__(self, preds_ref, preds_alt):
        return self.normalize(preds_ref, preds_alt)