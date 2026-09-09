from .istftnet import _w1 as _x1
from torch.nn.utils.parametrizations import weight_norm
import transformers.utils.import_utils as _x2

for _x3 in ('_torchvision_available', '_librosa_available', '_cv2_available'):
    if hasattr(_x2, _x3):
        setattr(_x2, _x3, False)
if hasattr(_x2, '_torchvision_version'):
    _x2._torchvision_version = 'N/A'

from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _x4(nn.Module):
    def __init__(self, _x5, _x6, _x7=True, _x8='linear'):
        super(_x4, self).__init__()
        self._x9 = nn.Linear(_x5, _x6, bias=_x7)
        nn.init.xavier_uniform_(self._x9.weight, gain=nn.init.calculate_gain(_x8))

    def forward(self, _xa):
        return self._x9(_xa)


class _xb(nn.Module):
    def __init__(self, _xc, _xd=1e-5):
        super().__init__()
        self._xc = _xc
        self._xd = _xd
        self._xe = nn.Parameter(torch.ones(_xc))
        self._xf = nn.Parameter(torch.zeros(_xc))

    def forward(self, _xa):
        _xa = _xa.transpose(1, -1)
        _xa = F.layer_norm(_xa, (self._xc,), self._xe, self._xf, self._xd)
        return _xa.transpose(1, -1)


class _xg(nn.Module):
    def __init__(self, _xh, _xi, _xj, _xk, _xl=nn.LeakyReLU(0.2)):
        super().__init__()
        self._xm = nn.Embedding(_xk, _xh)
        _xn = (_xi - 1) // 2
        self._xo = nn.ModuleList()
        for _ in range(_xj):
            self._xo.append(nn.Sequential(
                weight_norm(nn.Conv1d(_xh, _xh, kernel_size=_xi, padding=_xn)),
                _xb(_xh),
                _xl,
                nn.Dropout(0.2),
            ))
        self._xp = nn.LSTM(_xh, _xh//2, 1, batch_first=True, bidirectional=True)

    def forward(self, _xa, _xq, _xr):
        _xa = self._xm(_xa)
        _xa = _xa.transpose(1, 2)
        _xr = _xr.unsqueeze(1)
        _xa.masked_fill_(_xr, 0.0)
        for _xs in self._xo:
            _xa = _xs(_xa)
            _xa.masked_fill_(_xr, 0.0)
        _xa = _xa.transpose(1, 2)
        _xt = _xq if _xq.device == torch.device('cpu') else _xq.to('cpu')
        _xa = nn.utils.rnn.pack_padded_sequence(_xa, _xt, batch_first=True, enforce_sorted=False)
        self._xp.flatten_parameters()
        _xa, _ = self._xp(_xa)
        _xa, _ = nn.utils.rnn.pad_packed_sequence(_xa, batch_first=True)
        _xa = _xa.transpose(-1, -2)
        _xu = torch.zeros([_xa.shape[0], _xa.shape[1], _xr.shape[-1]], device=_xa.device)
        _xu[:, :, :_xa.shape[-1]] = _xa
        _xa = _xu
        _xa.masked_fill_(_xr, 0.0)
        return _xa


class _xv(nn.Module):
    def __init__(self, _xw, _xx, _xd=1e-5):
        super().__init__()
        self._xx = _xx
        self._xd = _xd
        self._xy = nn.Linear(_xw, _xx*2)

    def forward(self, _xa, _xz):
        _xa = _xa.transpose(-1, -2)
        _xa = _xa.transpose(1, -1)
        _y0 = self._xy(_xz)
        _y0 = _y0.view(_y0.size(0), _y0.size(1), 1)
        _y1, _y2 = torch.chunk(_y0, chunks=2, dim=1)
        _y1, _y2 = _y1.transpose(1, -1), _y2.transpose(1, -1)
        _xa = F.layer_norm(_xa, (self._xx,), eps=self._xd)
        _xa = (1 + _y1) * _xa + _y2
        return _xa.transpose(1, -1).transpose(-1, -2)


class _y3(nn.Module):
    def __init__(self, _y4, _y5, _y6, _y7=50, _y8=0.1):
        super().__init__()
        self._y9 = _ya(_ys=_y4, _yt=_y5, _yu=_y6, _y8=_y8)
        self._yb = nn.LSTM(_y5 + _y4, _y5 // 2, 1, batch_first=True, bidirectional=True)
        self._yc = _x4(_y5, _y7)
        self._yd = nn.LSTM(_y5 + _y4, _y5 // 2, 1, batch_first=True, bidirectional=True)
        self._ye = nn.ModuleList()
        self._ye.append(_x1(_y5, _y5, _y4, _y8=_y8))
        self._ye.append(_x1(_y5, _y5 // 2, _y4, upsample=True, _y8=_y8))
        self._ye.append(_x1(_y5 // 2, _y5 // 2, _y4, _y8=_y8))
        self._yf = nn.ModuleList()
        self._yf.append(_x1(_y5, _y5, _y4, _y8=_y8))
        self._yf.append(_x1(_y5, _y5 // 2, _y4, upsample=True, _y8=_y8))
        self._yf.append(_x1(_y5 // 2, _y5 // 2, _y4, _y8=_y8))
        self._yg = nn.Conv1d(_y5 // 2, 1, 1, 1, 0)
        self._yh = nn.Conv1d(_y5 // 2, 1, 1, 1, 0)

    def forward(self, _yi, _yj, _yk, _yl, _xr):
        _d = self._y9(_yi, _yj, _yk, _xr)
        _xr = _xr.unsqueeze(1)
        _xt = _yk if _yk.device == torch.device('cpu') else _yk.to('cpu')
        _xa = nn.utils.rnn.pack_padded_sequence(_d, _xt, batch_first=True, enforce_sorted=False)
        self._yb.flatten_parameters()
        _xa, _ = self._yb(_xa)
        _xa, _ = nn.utils.rnn.pad_packed_sequence(_xa, batch_first=True)
        _xu = torch.zeros([_xa.shape[0], _xr.shape[-1], _xa.shape[-1]], device=_xa.device)
        _xu[:, :_xa.shape[1], :] = _xa
        _xa = _xu
        _ym = self._yc(nn.functional.dropout(_xa, 0.5, training=False))
        _yn = (_d.transpose(-1, -2) @ _yl)
        return _ym.squeeze(-1), _yn

    def _yo(self, _xa, _se):
        _xa, _ = self._yd(_xa.transpose(-1, -2))
        _yp = _xa.transpose(-1, -2)
        for _yq in self._ye:
            _yp = _yq(_yp, _se)
        _yp = self._yg(_yp)
        _yr = _xa.transpose(-1, -2)
        for _yq in self._yf:
            _yr = _yq(_yr, _se)
        _yr = self._yh(_yr)
        return _yp.squeeze(1), _yr.squeeze(1)


class _ya(nn.Module):
    def __init__(self, _ys, _yt, _yu, _y8=0.1):
        super().__init__()
        self._yv = nn.ModuleList()
        for _ in range(_yu):
            self._yv.append(nn.LSTM(_yt + _ys, _yt // 2, num_layers=1, batch_first=True, bidirectional=True))
            self._yv.append(_xv(_ys, _yt))
        self._y8 = _y8
        self._yt = _yt
        self._ys = _ys

    def forward(self, _xa, _yj, _yk, _xr):
        _yw = _xr
        _xa = _xa.permute(2, 0, 1)
        _se = _yj.expand(_xa.shape[0], _xa.shape[1], -1)
        _xa = torch.cat([_xa, _se], axis=-1)
        _xa.masked_fill_(_yw.unsqueeze(-1).transpose(0, 1), 0.0)
        _xa = _xa.transpose(0, 1)
        _xa = _xa.transpose(-1, -2)
        for _yx in self._yv:
            if isinstance(_yx, _xv):
                _xa = _yx(_xa.transpose(-1, -2), _yj).transpose(-1, -2)
                _xa = torch.cat([_xa, _se.permute(1, 2, 0)], axis=1)
                _xa.masked_fill_(_yw.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                _xt = _yk if _yk.device == torch.device('cpu') else _yk.to('cpu')
                _xa = _xa.transpose(-1, -2)
                _xa = nn.utils.rnn.pack_padded_sequence(
                    _xa, _xt, batch_first=True, enforce_sorted=False)
                _yx.flatten_parameters()
                _xa, _ = _yx(_xa)
                _xa, _ = nn.utils.rnn.pad_packed_sequence(
                    _xa, batch_first=True)
                _xa = F.dropout(_xa, p=self._y8, training=False)
                _xa = _xa.transpose(-1, -2)
                _xu = torch.zeros([_xa.shape[0], _xa.shape[1], _xr.shape[-1]], device=_xa.device)
                _xu[:, :, :_xa.shape[-1]] = _xa
                _xa = _xu

        return _xa.transpose(-1, -2)


class _yy(AlbertModel):
    def forward(self, *_yz, **_z0):
        _z1 = super().forward(*_yz, **_z0)
        return _z1.last_hidden_state
