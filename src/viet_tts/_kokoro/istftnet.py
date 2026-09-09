from .custom_stft import _p0 as _s0
from torch.nn.utils.parametrizations import weight_norm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _s1(m, _s2=0.0, _s3=0.01):
    _s4 = m.__class__.__name__
    if _s4.find("Conv") != -1:
        m.weight.data.normal_(_s2, _s3)

def _s5(_s6, _s7=1):
    return int((_s6*_s7 - _s7)/2)


class _s8(nn.Module):
    def __init__(self, _s9, _sa):
        super().__init__()
        self._sb = nn.InstanceNorm1d(_sa, affine=True)
        self._sc = nn.Linear(_s9, _sa*2)

    def forward(self, _sd, _se):
        _sf = self._sc(_se)
        _sf = _sf.view(_sf.size(0), _sf.size(1), 1)
        _sg, _sh = torch.chunk(_sf, chunks=2, dim=1)
        return (1 + _sg) * self._sb(_sd) + _sh


class _si(nn.Module):
    def __init__(self, _sj, _sk=3, _sl=(1, 3, 5), _sm=64):
        super(_si, self).__init__()
        self._sn = nn.ModuleList([
            weight_norm(nn.Conv1d(_sj, _sj, _sk, 1, dilation=_sl[0],
                                  padding=_s5(_sk, _sl[0]))),
            weight_norm(nn.Conv1d(_sj, _sj, _sk, 1, dilation=_sl[1],
                                  padding=_s5(_sk, _sl[1]))),
            weight_norm(nn.Conv1d(_sj, _sj, _sk, 1, dilation=_sl[2],
                                  padding=_s5(_sk, _sl[2])))
        ])
        self._sn.apply(_s1)
        self._so = nn.ModuleList([
            weight_norm(nn.Conv1d(_sj, _sj, _sk, 1, dilation=1,
                                  padding=_s5(_sk, 1))),
            weight_norm(nn.Conv1d(_sj, _sj, _sk, 1, dilation=1,
                                  padding=_s5(_sk, 1))),
            weight_norm(nn.Conv1d(_sj, _sj, _sk, 1, dilation=1,
                                  padding=_s5(_sk, 1)))
        ])
        self._so.apply(_s1)
        self._sp = nn.ModuleList([
            _s8(_sm, _sj),
            _s8(_sm, _sj),
            _s8(_sm, _sj),
        ])
        self._sq = nn.ModuleList([
            _s8(_sm, _sj),
            _s8(_sm, _sj),
            _s8(_sm, _sj),
        ])
        self._sr = nn.ParameterList([nn.Parameter(torch.ones(1, _sj, 1)) for _ in range(len(self._sn))])
        self._ss = nn.ParameterList([nn.Parameter(torch.ones(1, _sj, 1)) for _ in range(len(self._so))])

    def forward(self, _sd, _se):
        for _st, _su, _sv, _sw, _sx, _sy in zip(self._sn, self._so, self._sp, self._sq, self._sr, self._ss):
            _sz = _sv(_sd, _se)
            _sz = _sz + (1 / _sx) * (torch.sin(_sx * _sz) ** 2)
            _sz = _st(_sz)
            _sz = _sw(_sz, _se)
            _sz = _sz + (1 / _sy) * (torch.sin(_sy * _sz) ** 2)
            _sz = _su(_sz)
            _sd = _sz + _sd
        return _sd


class _t0(nn.Module):
    def __init__(self, _t1=800, _t2=200, _t3=800, _t4='hann'):
        super().__init__()
        self._t5 = _t1
        self._t6 = _t2
        self._t7 = _t3
        assert _t4 == 'hann', _t4
        self._t8 = torch.hann_window(_t3, periodic=True, dtype=torch.float32)

    def _t9(self, _ta):
        _tb = torch.stft(
            _ta,
            self._t5, self._t6, self._t7, window=self._t8.to(_ta.device),
            return_complex=True)
        return torch.abs(_tb), torch.angle(_tb)

    def _tc(self, _td, _te):
        _tf = torch.istft(
            _td * torch.exp(_te * 1j),
            self._t5, self._t6, self._t7, window=self._t8.to(_td.device))
        return _tf.unsqueeze(-2)

    def _qa(self, _td, _te, _qb=None):
        return self._tc(_td, _te)

    def forward(self, _ta):
        self._tg, self._th = self._t9(_ta)
        _ti = self._tc(self._tg, self._th)
        return _ti


class _tj(nn.Module):
    def __init__(self, _tk, _tl, _tm=0,
                 _tn=0.1, _to=0.003,
                 _tp=0,
                 _tq=False):
        super(_tj, self).__init__()
        self._tn = _tn
        self._to = _to
        self._tm = _tm
        self._tr = self._tm + 1
        self._ts = _tk
        self._tp = _tp
        self._tq = _tq
        self._tl = _tl

    def _tt(self, _tu):
        _tv = (_tu > self._tp).type(torch.float32)
        return _tv

    def _tw(self, _tu):
        _tx = (_tu / self._ts) % 1
        _ty = torch.rand(_tu.shape[0], _tu.shape[2], device=_tu.device)
        _ty[:, 0] = 0
        _tx[:, 0, :] = _tx[:, 0, :] + _ty
        if not self._tq:
            _tx = F.interpolate(_tx.transpose(1, 2), scale_factor=1/self._tl, mode="linear").transpose(1, 2)
            _tz = torch.cumsum(_tx, dim=1) * 2 * torch.pi
            _tz = F.interpolate(_tz.transpose(1, 2) * self._tl, scale_factor=self._tl, mode="linear").transpose(1, 2)
            _u0 = torch.sin(_tz)
        else:
            _u1 = self._tt(_tu)
            _u2 = torch.roll(_u1, shifts=-1, dims=1)
            _u2[:, -1, :] = 1
            _u3 = (_u1 < 1) * (_u2 > 0)
            _u4 = torch.cumsum(_tx, dim=1)
            for _u5 in range(_tu.shape[0]):
                _u6 = _u4[_u5, _u3[_u5, :, 0], :]
                _u6[1:, :] = _u6[1:, :] - _u6[0:-1, :]
                _u4[_u5, :, :] = 0
                _u4[_u5, _u3[_u5, :, 0], :] = _u6
            _u7 = torch.cumsum(_tx - _u4, dim=1)
            _u0 = torch.cos(_u7 * 2 * torch.pi)
        return _u0

    def forward(self, _u8):
        _u9 = torch.zeros(_u8.shape[0], _u8.shape[1], self._tr, device=_u8.device)
        _ua = torch.multiply(_u8, torch.FloatTensor([[range(1, self._tm + 2)]]).to(_u8.device))
        _ub = self._tw(_ua) * self._tn
        _uc = self._tt(_u8)
        _ud = _uc * self._to + (1 - _uc) * self._tn / 3
        _ue = _ud * torch.randn_like(_ub)
        _ub = _ub * _uc + _ue
        return _ub, _uc, _ue


class _uf(nn.Module):
    def __init__(self, _ug, _uh, _ui=0, _uj=0.1,
                 _uk=0.003, _ul=0):
        super(_uf, self).__init__()
        self._uj = _uj
        self._uk = _uk
        self._um = _tj(_ug, _uh, _ui,
                                 _uj, _uk, _ul)
        self._un = nn.Linear(_ui + 1, 1)
        self._uo = nn.Tanh()

    def forward(self, _up):
        with torch.no_grad():
            _uq, _ur, _ = self._um(_up)
        _us = self._uo(self._un(_uq))
        _ut = torch.randn_like(_ur) * self._uj / 3
        return _us, _ut, _ur


class _uu(nn.Module):
    def __init__(self, _uv, _uw, _ux, _uy, _uz, _v0, _v1, _v2, _v3=False):
        super(_uu, self).__init__()
        self._v4 = len(_uw)
        self._v5 = len(_ux)
        self._v6 = _uf(
                    _ug=24000,
                    _uh=math.prod(_ux) * _v2,
                    _ui=8, _ul=10)
        self._v7 = nn.Upsample(scale_factor=math.prod(_ux) * _v2)
        self._v8 = nn.ModuleList()
        self._v9 = nn.ModuleList()
        self._va = nn.ModuleList()
        for _vb, (_vc, _vd) in enumerate(zip(_ux, _v0)):
            self._va.append(weight_norm(
                nn.ConvTranspose1d(_uy//(2**_vb), _uy//(2**(_vb+1)),
                                   _vd, _vc, padding=(_vd-_vc)//2)))
        self._vb = nn.ModuleList()
        for _vb in range(len(self._va)):
            _ve = _uy//(2**(_vb+1))
            for _vf, (_vg, _vh) in enumerate(zip(_uw, _uz)):
                self._vb.append(_si(_ve, _vg, _vh, _uv))
            _vi = _uy // (2 ** (_vb + 1))
            if _vb + 1 < len(_ux):
                _vj = math.prod(_ux[_vb + 1:])
                self._v8.append(nn.Conv1d(
                    _v1 + 2, _vi, kernel_size=_vj * 2, stride=_vj, padding=(_vj+1) // 2))
                self._v9.append(_si(_vi, 7, [1,3,5], _uv))
            else:
                self._v8.append(nn.Conv1d(_v1 + 2, _vi, kernel_size=1))
                self._v9.append(_si(_vi, 11, [1,3,5], _uv))
        self._vk = _v1
        self._vl = weight_norm(nn.Conv1d(_ve, self._vk + 2, 7, 1, padding=3))
        self._va.apply(_s1)
        self._vl.apply(_s1)
        self._vm = nn.ReflectionPad1d((1, 0))
        self._vn = (
            _s0(_p1=_v1, _p2=_v2, _p3=_v1)
            if _v3
            else _t0(_t1=_v1, _t2=_v2, _t3=_v1)
        )

    def forward(self, _sd, _se, _vo):
        with torch.no_grad():
            _vo = self._v7(_vo[:, None]).transpose(1, 2)
            _vp, _vq, _vr = self._v6(_vo)
            _vp = _vp.transpose(1, 2).squeeze(1)
            _vs, _vt = self._vn._t9(_vp)
            _vu = torch.cat([_vs, _vt], dim=1)
        for _vb in range(self._v5):
            _sd = F.leaky_relu(_sd, negative_slope=0.1)
            _vv = self._v8[_vb](_vu)
            _vv = self._v9[_vb](_vv, _se)
            _sd = self._va[_vb](_sd)
            if _vb == self._v5 - 1:
                _sd = self._vm(_sd)
            _sd = _sd + _vv
            _vw = None
            for _vf in range(self._v4):
                if _vw is None:
                    _vw = self._vb[_vb*self._v4+_vf](_sd, _se)
                else:
                    _vw += self._vb[_vb*self._v4+_vf](_sd, _se)
            _sd = _vw / self._v4
        _sd = F.leaky_relu(_sd)
        _sd = self._vl(_sd)
        _vx = torch.exp(_sd[:,:self._vk // 2 + 1, :])
        _vy = torch.sin(_sd[:, self._vk // 2 + 1:, :])
        return self._vn._qa(_vx, _vy)


class _vz(nn.Module):
    def __init__(self, _w0):
        super().__init__()
        self._w0 = _w0

    def forward(self, _sd):
        if self._w0 == 'none':
            return _sd
        else:
            return F.interpolate(_sd, scale_factor=2, mode='nearest')


class _w1(nn.Module):
    def __init__(self, _w2, _w3, _w4=64, _w5=nn.LeakyReLU(0.2), _w6='none', _w7=0.0, upsample=None):
        super().__init__()
        if upsample is True:
            _w6 = 't'
        self._w5 = _w5
        self._w8 = _w6
        self._w9 = _vz(_w6)
        self._wa = _w2 != _w3
        self._wb(_w2, _w3, _w4)
        self._wc = nn.Dropout(_w7)
        if _w6 == 'none':
            self._wd = nn.Identity()
        else:
            self._wd = weight_norm(nn.ConvTranspose1d(_w2, _w2, kernel_size=3, stride=2, groups=_w2, padding=1, output_padding=1))

    def _wb(self, _w2, _w3, _w4):
        self._we = weight_norm(nn.Conv1d(_w2, _w3, 3, 1, 1))
        self._wf = weight_norm(nn.Conv1d(_w3, _w3, 3, 1, 1))
        self._wg = _s8(_w4, _w2)
        self._wh = _s8(_w4, _w3)
        if self._wa:
            self._wi = weight_norm(nn.Conv1d(_w2, _w3, 1, 1, 0, bias=False))

    def _wj(self, _sd):
        _sd = self._w9(_sd)
        if self._wa:
            _sd = self._wi(_sd)
        return _sd

    def _wk(self, _sd, _se):
        _sd = self._wg(_sd, _se)
        _sd = self._w5(_sd)
        _sd = self._wd(_sd)
        _sd = self._we(self._wc(_sd))
        _sd = self._wh(_sd, _se)
        _sd = self._w5(_sd)
        _sd = self._wf(self._wc(_sd))
        return _sd

    def forward(self, _sd, _se):
        _wl = self._wk(_sd, _se)
        _wl = (_wl + self._wj(_sd)) * torch.rsqrt(torch.tensor(2))
        return _wl


class _wm(nn.Module):
    def __init__(self, _wn, _wo, _wp, _wx=False,
                 resblock_kernel_sizes=None,
                 upsample_rates=None,
                 upsample_initial_channel=None,
                 resblock_dilation_sizes=None,
                 upsample_kernel_sizes=None,
                 gen_istft_n_fft=None,
                 gen_istft_hop_size=None,
                 _wq=None, _wr=None, _ws=None, _wt=None, _wu=None, _wv=None, _ww=None):
        super().__init__()
        _wq = _wq or resblock_kernel_sizes
        _wr = _wr or upsample_rates
        _ws = _ws or upsample_initial_channel
        _wt = _wt or resblock_dilation_sizes
        _wu = _wu or upsample_kernel_sizes
        _wv = _wv or gen_istft_n_fft
        _ww = _ww or gen_istft_hop_size
        self._wy = _w1(_wn + 2, 1024, _wo)
        self._wz = nn.ModuleList()
        self._wz.append(_w1(1024 + 2 + 64, 1024, _wo))
        self._wz.append(_w1(1024 + 2 + 64, 1024, _wo))
        self._wz.append(_w1(1024 + 2 + 64, 1024, _wo))
        self._wz.append(_w1(1024 + 2 + 64, 512, _wo, upsample=True))
        self._x0 = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self._x1 = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self._x2 = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self._x3 = _uu(_wo, _wq, _wr,
                                   _ws, _wt,
                                   _wu, _wv, _ww, _v3=_wx)

    def forward(self, _x4, _x5, _x6, _se):
        _x7 = self._x0(_x5.unsqueeze(1))
        _x8 = self._x1(_x6.unsqueeze(1))
        _sd = torch.cat([_x4, _x7, _x8], axis=1)
        _sd = self._wy(_sd, _se)
        _x9 = self._x2(_x4)
        _xa = True
        for _xb in self._wz:
            if _xa:
                _sd = torch.cat([_sd, _x9, _x7, _x8], axis=1)
            _sd = _xb(_sd, _se)
            if _xb._w8 != "none":
                _xa = False
        _sd = self._x3(_sd, _se, _x5)
        return _sd
