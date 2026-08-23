# -*- coding: utf-8 -*-

from .__info__ import __version__, __description__
from .utils import *
from .parser import *
from .arena.panda_battle import build_panda_battle
from .token_level_supervision_utils import (
    compute_token_level_supervision,
    build_tokenizer,
    unicode_tokenizer,
    utf8_tokenizer,
)
from .correcting_model.far_correction_utils import (
    CorrectionAdapter,
    FindAndReplaceCorrectionAdapter,
    NextTokenPredictionAsCorrectingBuilder,
)
from .correcting_model.system_prompts import (
    far_tokenizer_aware_system_prompt_en,
    far_tokenizer_agnostic_system_prompt_en,
    far_tokenizer_aware_system_prompt_cn,
    far_tokenizer_agnostic_system_prompt_cn,
)
from .correcting_model.verifier import (
    CorrectionVerifier,
    FindAndReplaceCodecMixin,
    ON_PANDA_PLACEHOLDER,
    ON_PANDA_TRUNCATE_PREFIX,
)
from .correcting_model.correcting_model import CorrectingModel
from .response_templates import build_response_template
