import torch
import torch.nn as nn

from typing import Dict, Any, Tuple
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig

from models.whisper.audio import pad_or_trim, log_mel_spectrogram


# RMIS adapter for local Qwen2-Audio encoder weights. It reuses the shared
# Whisper audio preprocessing utilities instead of duplicating them here.
class Qwen2_Audio(nn.Module):
    def __init__(
        self,
        ckpt: str,
        feat_aggre: str = 'mean_pool',
    ) -> None:
        super().__init__()
        if feat_aggre != 'mean_pool':
            raise ValueError(f'Unsupported feat_aggre: {feat_aggre}')

        audio_config = Qwen2AudioEncoderConfig()
        self.model = Qwen2AudioEncoder(audio_config)
        encoder_dict = torch.load(ckpt, map_location='cpu', weights_only=True)
        self.model.load_state_dict(encoder_dict)

    def process_audio(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_audio = pad_or_trim(x)
        input_features = log_mel_spectrogram(input_audio, n_mels=128)
        feature_lens = torch.full(
            (input_features.shape[0],),
            input_features.shape[-1],
            dtype=torch.long,
            device=input_features.device,
        )

        audio_feat_lengths, _ = self.model._get_feat_extract_output_lengths(feature_lens)
        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        seq_range = torch.arange(
            0,
            max_seq_len,
            dtype=audio_feat_lengths.dtype,
            device=audio_feat_lengths.device,
        ).unsqueeze(0).expand(batch_size, max_seq_len)
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        padding_mask = seq_range >= lengths_expand
        audio_attention_mask_ = padding_mask.view(
            batch_size,
            1,
            1,
            max_seq_len,
        ).expand(batch_size, 1, max_seq_len, max_seq_len)
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.model.conv1.weight.dtype,
            device=self.model.conv1.weight.device,
        )
        audio_attention_mask[audio_attention_mask_] = float('-inf')

        return input_features, audio_attention_mask

    def embedding(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_outputs = self.model(input_features, attention_mask=audio_attention_mask)
        x = audio_outputs.last_hidden_state
        return torch.mean(x, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        input_features, audio_attention_mask = self.process_audio(x)
        x = self.embedding(input_features, audio_attention_mask)
        output_dict = {'embedding': x}
        return output_dict
