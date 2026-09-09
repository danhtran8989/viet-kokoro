from .istftnet import _wm as _x3
from .modules import _yy as _yy, _y3 as _y3, _xg as _xg
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class _q0(torch.nn.Module):
    def __init__(
        self,
        _q1: Optional[str] = None,
        _q2: Union[Dict, str, None] = None,
        _q3: Optional[str] = None,
        _q4: bool = False
    ):
        super().__init__()
        if _q1 is None:
            _q1 = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting _q1 to {_q1}. Pass _q1='{_q1}' to suppress this warning.")
        self._q5 = _q1
        if not isinstance(_q2, dict):
            if not _q2:
                logger.debug("No _q2 provided, downloading from HF")
                _q2 = hf_hub_download(repo_id=_q1, filename='config.json')
            with open(_q2, 'r', encoding='utf-8') as _q6:
                _q2 = json.load(_q6)
                logger.debug(f"Loaded _q2: {_q2}")
        self._q7 = _q2['vocab']
        self.bert = _yy(AlbertConfig(vocab_size=_q2['n_token'], **_q2['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, _q2['hidden_dim'])
        self._qa = self.bert.config.max_position_embeddings
        self.predictor = _y3(
            _y4=_q2['style_dim'], _y5=_q2['hidden_dim'],
            _y6=_q2['n_layer'], _y7=_q2['max_dur'], _y8=_q2['dropout']
        )
        self.text_encoder = _xg(
            _xh=_q2['hidden_dim'], _xi=_q2['text_encoder_kernel_size'],
            _xj=_q2['n_layer'], _xk=_q2['n_token']
        )
        self.decoder = _x3(
            _wn=_q2['hidden_dim'], _wo=_q2['style_dim'],
            _wp=_q2['n_mels'], _wx=_q4, **_q2['istftnet']
        )
        if not _q3:
            _q3 = hf_hub_download(repo_id=_q1, filename=_q0._qe[_q1])
        for _qf, _qg in torch.load(_q3, map_location='cpu', weights_only=True).items():
            assert hasattr(self, _qf), _qf
            try:
                getattr(self, _qf).load_state_dict(_qg)
            except:
                logger.debug(f"Did not load {_qf} from state_dict")
                _qg = {k[7:]: v for k, v in _qg.items()}
                getattr(self, _qf).load_state_dict(_qg, strict=False)

    _qe = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class _qh:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def _qi(
        self,
        _qj: torch.LongTensor,
        _qk: torch.FloatTensor,
        _ql: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        _qm = torch.full(
            (_qj.shape[0],), 
            _qj.shape[-1], 
            device=_qj.device,
            dtype=torch.long
        )

        _qn = torch.arange(_qm.max()).unsqueeze(0).expand(_qm.shape[0], -1).type_as(_qm)
        _qn = torch.gt(_qn+1, _qm.unsqueeze(1)).to(self.device)
        _qo = self.bert(_qj, attention_mask=(~_qn).int())
        _qp = self.bert_encoder(_qo).transpose(-1, -2)
        _qq = _qk[:, 128:]
        _qr = self.predictor._y9(_qp, _qq, _qm, _qn)
        _qs, _ = self.predictor._yb(_qr)
        _qt = self.predictor._yc(_qs)
        _qt = torch.sigmoid(_qt).sum(axis=-1) / _ql
        _qu = torch.round(_qt).clamp(min=1).long().squeeze()
        _qv = torch.repeat_interleave(torch.arange(_qj.shape[1], device=self.device), _qu)
        _qw = torch.zeros((_qj.shape[1], _qv.shape[0]), device=self.device)
        _qw[_qv, torch.arange(_qv.shape[0])] = 1
        _qw = _qw.unsqueeze(0).to(self.device)
        _qx = _qr.transpose(-1, -2) @ _qw
        _qy, _qz = self.predictor._yo(_qx, _qq)
        _r0 = self.text_encoder(_qj, _qm, _qn)
        _r1 = _r0 @ _qw
        _r2 = self.decoder(_r1, _qy, _qz, _qk[:, :128]).squeeze()
        return _r2, _qu

    def forward(
        self,
        _r3: str,
        _qk: torch.FloatTensor,
        _ql: float = 1,
        _r4: bool = False
    ) -> Union['_q0._qh', torch.FloatTensor]:
        _qj = list(filter(lambda _r5: _r5 is not None, map(lambda _r6: self._q7.get(_r6), _r3)))
        logger.debug(f"_r3: {_r3} -> _qj: {_qj}")
        assert len(_qj)+2 <= self._qa, (len(_qj)+2, self._qa)
        _qj = torch.LongTensor([[0, *_qj, 0]]).to(self.device)
        _qk = _qk.to(self.device)
        _r2, _qu = self._qi(_qj, _qk, _ql)
        _r2 = _r2.squeeze().cpu()
        _qu = _qu.cpu() if _qu is not None else None
        logger.debug(f"_qu: {_qu}")
        return self._qh(audio=_r2, pred_dur=_qu) if _r4 else _r2

class _r7(torch.nn.Module):
    def __init__(self, _r8: _q0):
        super().__init__()
        self._r8 = _r8

    def forward(
        self,
        _qj: torch.LongTensor,
        _qk: torch.FloatTensor,
        _ql: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        _r9, _qu = self._r8._qi(_qj, _qk, _ql)
        return _r9, _qu
