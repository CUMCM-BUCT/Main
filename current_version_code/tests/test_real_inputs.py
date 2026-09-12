"""依赖当前工作区题目附件的集成检查；纯逻辑单元测试见 test_inputs.py。"""

import unittest

from a_model.inputs import load_environment, load_radius


class RealAttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = load_environment()
        cls.radius = load_radius()

    def test_environment_attachment(self):
        self.assertEqual(len(self.environment.times_s), 241)
        self.assertEqual(self.environment.at(0.0), (28.0, 0.01963))
        self.assertEqual(self.environment.at(14_401.0), (50.0, 0.05))
        temperature, moisture = self.environment.last_hour_mean()
        self.assertAlmostEqual(temperature, 49.99893442622951)
        self.assertAlmostEqual(moisture, 0.04998754098360657)
        self.assertEqual(len(self.environment.trace.sha256), 64)

    def test_radius_attachment(self):
        self.assertEqual(len(self.radius.times_s), 145)
        self.assertAlmostEqual(self.radius.at(0.0), 0.02)
        self.assertAlmostEqual(self.radius.at(259_201.0), 0.01198)
        self.assertEqual(len(self.radius.trace.sha256), 64)


if __name__ == "__main__":
    unittest.main()
