import torch
import torch.nn as nn

from typing import Dict, Any, Tuple
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig

from models.whisper.audio import pad_or_trim, log_mel_spectrogram


# RMIS adapter for local Qwen2.5-Omni audio encoder weights. It reuses the
# shared Whisper audio preprocessing utilities instead of duplicating them here.
class Qwen2_5_Audio(nn.Module):
    def __init__(
        self,
        ckpt: str,
        feat_aggre: str = 'mean_pool',
    ) -> None:
        super().__init__()
        if feat_aggre != 'mean_pool':
            raise ValueError(f'Unsupported feat_aggre: {feat_aggre}')

        audio_config = Qwen2_5OmniAudioEncoderConfig()
        self.model = Qwen2_5OmniAudioEncoder(audio_config)
        encoder_dict = torch.load(ckpt, map_location='cpu', weights_only=True)
        self.model.load_state_dict(encoder_dict)

    def process_audio(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_mel_bins = getattr(self.model.config, 'num_mel_bins', 128)
        input_audio = pad_or_trim(x)
        input_features = log_mel_spectrogram(input_audio, n_mels=num_mel_bins)
        input_features = torch.where(torch.isfinite(input_features), input_features, torch.zeros_like(input_features))
        feature_lens = torch.full(
            (input_features.shape[0],),
            input_features.shape[-1],
            dtype=torch.long,
            device=input_features.device,
        )
        return input_features, feature_lens

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_features.device
        batch_size, _, max_len = input_features.shape

        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        feature_attention_mask = positions < feature_lens.unsqueeze(1)
        valid_feat = input_features.permute(0, 2, 1)[feature_attention_mask]
        concat_feat = valid_feat.permute(1, 0).contiguous()

        audio_feat_lengths, audio_output_lengths = self.model._get_feat_extract_output_lengths(
            feature_lens
        )
        audio_outputs = self.model(
            input_features=concat_feat,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state
        expected_len = int(audio_output_lengths.sum().item())
        if audio_features.shape[0] != expected_len:
            raise ValueError(
                f'audio feature length {audio_features.shape[0]} does not match {expected_len}'
            )
        return audio_features, audio_output_lengths

    def embedding(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        audio_features, output_lens = self.get_audio_features(input_features, feature_lens)
        feature_list = torch.split(audio_features, output_lens.tolist(), dim=0)
        pooled_features = [feat.mean(dim=0) for feat in feature_list]
        return torch.stack(pooled_features, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        input_features, feature_lens = self.process_audio(x)
        x = self.embedding(input_features, feature_lens)
        output_dict = {'embedding': x}
        return output_dict
