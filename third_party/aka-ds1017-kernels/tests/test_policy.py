import unittest
from aka_ds1017 import Policy


class TestProfile(unittest.TestCase):
    def test_large_policy_uses_original_small_kernels(self):
        for b in (4096,8192,16384):
            p = Policy.create("validated",b)
            self.assertIsNone(p.select("projection",b,"qb"))

    def test_projection_and_mhc_layout(self):
        for b in (4096,8192,16384):
            p=Policy.create("expanded",b)
            self.assertEqual(p.select("projection",b,"qb"),"projection_producer")
            self.assertIsNone(p.select("projection",b,"qkv"))
            self.assertIsNone(p.select("mhc",b//4096*140,"attn"))
            self.assertIsNone(p.select("mhc",b,"bf16"))
            self.assertEqual(p.select("mhc",b,"moe"),None if b==4096 else "r1720_gp")
            self.assertEqual(p.select("mla",b,"sparse640"),None if b==16384 else "mla_producer")
            self.assertIsNone(p.select("mla",b,"prefill"))

    def test_collective_requires_agreement_and_budget(self):
        p=Policy.create("validated",32768)
        self.assertEqual(p.select("megamoe",32768,"uniform-ep4"),"megamoe_producer")
        self.assertEqual(p.select("megamoe",1120,"uniform-ep4"),"megamoe_consumer")
        self.assertIsNone(p.select("megamoe",32768,"mixed"))
        self.assertIsNone(Policy.create("validated",65536).select("megamoe",32768,"uniform-ep4"))

    def test_disabled_and_unsupported_routes(self):
        for profile in ("validated","expanded","official"):
            p=Policy.create(profile,32768,"index, mla")
            self.assertIsNone(p.select("index",1120,"user"))
            self.assertIsNone(p.select("mla",1120,"paged"))
        with self.assertRaises(ValueError):
            Policy.create("expanded",8192,"index,unknown")
        with self.assertRaises(ValueError):
            Policy.create("unknown",8192)
        self.assertIsNone(Policy.create("expanded",32767).select("mhc",8192,"attn"))
