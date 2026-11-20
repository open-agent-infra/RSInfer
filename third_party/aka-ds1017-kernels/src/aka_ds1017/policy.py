from dataclasses import dataclass

LARGE = (32768, 65536)
COMPACT = (1120, 2240)
SMALL = (4096, 8192, 16384)
SHORT = (140, 280, 560)
OPERATORS = frozenset(("projection", "index", "mla", "mhc", "megamoe"))


@dataclass(frozen=True)
class Policy:
    profile: str
    budget: int
    disabled: frozenset[str]

    @classmethod
    def create(cls, profile, budget, disabled=""):
        if profile not in ("official", "validated", "expanded"):
            raise ValueError(f"Unknown DS1017 kernel profile {profile!r}")
        exclusions = frozenset(s.strip() for s in disabled.split(",") if s.strip())
        if exclusions - OPERATORS:
            raise ValueError(f"Unknown disabled operators {sorted(exclusions - OPERATORS)}")
        return cls(profile, budget, exclusions)

    def select(self, op, rows, path):
        if self.profile == "official" or op in self.disabled:
            return None
        large = self.budget in LARGE and rows in LARGE + COMPACT
        small = self.profile == "expanded" and self.budget in SMALL and rows <= self.budget
        producer = rows >= 4096
        if op == "megamoe":
            if self.budget == 32768 and path == "uniform-ep4" and rows in (1120, 32768):
                return "megamoe_producer" if producer else "megamoe_consumer"
        elif op == "index":
            if large or small and ((rows in SMALL and path == "dense") or (rows in SHORT and path in ("source","user"))):
                if producer:
                    return {"user": "index_producer", "dense": "speed_i"}.get(path, "r1720_ip")
                return "r1720_ic" if path == "user" else "index_consumer"
        elif op == "mhc":
            if large or small and rows in (8192,16384) and path in ("attn","moe"):
                if not producer:
                    return "r1720_gc"
                return "r1720_gb" if path == "bf16" else "r1720_gp"
        elif op == "mla":
            if path == "paged":
                return "paged_bridge" if large and rows in COMPACT else None
            if large or small and rows in (4096,8192) and path == "sparse640":
                return "mla_producer" if producer else "mla_consumer"
        elif op == "projection":
            if large or small and ((rows in SMALL and path == "qb") or (rows == 280 and path == "qkv") or (rows in (280,560) and path == "wo_a")):
                return "projection_producer" if producer else "projection_consumer"
        return None
