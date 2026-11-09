"""Score every pool row with the rule engine and the ProtectAI guard model (inputs to the fusion experiments)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT, load_pool  # noqa: E402

from sentinelai.audit import NullAuditLogger  # noqa: E402
from sentinelai.config import SecurityPolicy, get_settings  # noqa: E402
from sentinelai.detectors.secrets import SecretsConfig  # noqa: E402
from sentinelai.evaluation.external import guard_scorer, sentinel_scorer  # noqa: E402
from sentinelai.gateway.pipeline import SecurityGateway  # noqa: E402
from sentinelai.models import MockModel  # noqa: E402

df = load_pool()
texts = df.text.tolist()
s = get_settings()
gw = SecurityGateway(SecurityPolicy(), MockModel(), NullAuditLogger(), SecretsConfig.load(s.protected_secrets_path), max_input_chars=1_000_000)
rule = sentinel_scorer(gw, True, False)(texts).scores
run, _ = guard_scorer(ROOT / "models/guards/protectai_v2", "cuda")
guard = run(texts).scores
np.savez(OUT / "components_pool.npz", rule=np.array(rule), protectai=np.array(guard))
print("scored", len(texts), "rule>0:", float(np.mean(np.array(rule) > 0)), "guard>=.5:", float(np.mean(np.array(guard) >= 0.5)))
