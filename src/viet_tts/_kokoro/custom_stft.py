from attr import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _p0(nn.Module):
    def __init__(
        self,
        _p1=800,
        _p2=200,
        _p3=800,
        _p4="hann",
        _p5=True,
        _p6="replicate",
    ):
        super().__init__()
        self._p7 = _p1
        self._p8 = _p2
        self._p9 = _p3
        self._pa = _p1
        self._pb = _p5
        self._pc = _p6

        self._pd = self._pa // 2 + 1

        assert _p4 == 'hann', _p4
        _pe = torch.hann_window(_p3, periodic=True, dtype=torch.float32)
        if self._p9 < self._pa:
            _pf = self._pa - self._p9
            _pe = F.pad(_pe, (0, _pf))
        elif self._p9 > self._pa:
            _pe = _pe[: self._pa]
        self.register_buffer("window", _pe)

        _pg = np.arange(self._pa)
        _ph = np.arange(self._pd)
        _pi = 2 * np.pi * np.outer(_ph, _pg) / self._pa
        _pj = np.cos(_pi)
        _pk = -np.sin(_pi)

        _pl = _pe.numpy()
        _pm = _pj * _pl
        _pn = _pk * _pl

        _po = torch.from_numpy(_pm).float()
        _pp = torch.from_numpy(_pn).float()

        self.register_buffer(
            "_pq", _po.unsqueeze(1)
        )
        self.register_buffer(
            "_pr", _pp.unsqueeze(1)
        )

        _ps = 1.0 / self._pa
        _pg = np.arange(self._pa)
        _pt = 2 * np.pi * np.outer(_pg, _ph) / self._pa
        _pu = np.cos(_pt).T
        _pv = np.sin(_pt).T

        _pw = _pe.numpy() * _ps
        _px = _pu * _pw
        _py = _pv * _pw

        self.register_buffer(
            "_pz", torch.from_numpy(_px).float().unsqueeze(1)
        )
        self.register_buffer(
            "_q0", torch.from_numpy(_py).float().unsqueeze(1)
        )


    def _q1(self, _q2: torch.Tensor):
        if self._pb:
            _q3 = self._pa // 2
            _q2 = F.pad(_q2, (_q3, _q3), mode=self._pc)

        _q4 = _q2.unsqueeze(1)
        _q5 = F.conv1d(
            _q4,
            self._pq,
            bias=None,
            stride=self._p8,
            padding=0,
        )
        _q6 = F.conv1d(
            _q4,
            self._pr,
            bias=None,
            stride=self._p8,
            padding=0,
        )

        _q7 = torch.sqrt(_q5**2 + _q6**2 + 1e-14)
        _q8 = torch.atan2(_q6, _q5)
        _q9 = (_q6 == 0) & (_q5 < 0)
        _q8[_q9] = torch.pi
        return _q7, _q8


    def _qa(self, _q7: torch.Tensor, _q8: torch.Tensor, _qb=None):
        _qc = _q7 * torch.cos(_q8)
        _qd = _q7 * torch.sin(_q8)

        _qc = _qc
        _qd = _qd

        _qe = F.conv_transpose1d(
            _qc,
            self._pz,
            bias=None,
            stride=self._p8,
            padding=0,
        )
        _qf = F.conv_transpose1d(
            _qd,
            self._q0,
            bias=None,
            stride=self._p8,
            padding=0,
        )
        _qg = _qe - _qf

        if self._pb:
            _qh = self._pa // 2
            _qg = _qg[..., _qh:-_qh]

        if _qb is not None:
            _qg = _qg[..., :_qb]

        return _qg

    def forward(self, _qi: torch.Tensor):
        _qj, _qk = self._q1(_qi)
        return self._qa(_qj, _qk, length=_qi.shape[-1])
