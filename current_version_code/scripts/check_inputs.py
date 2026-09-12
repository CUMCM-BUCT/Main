"""读取题目附件并打印可审阅的输入摘要，不修改源文件。"""

from __future__ import annotations

import json

from a_model.inputs import load_environment, load_radius


def main() -> None:
    environment = load_environment()
    radius = load_radius()
    last_hour_temperature, last_hour_moisture = environment.last_hour_mean()
    summary = {
        "environment": {
            "source": environment.trace.__dict__,
            "time_range_s": [environment.times_s[0], environment.times_s[-1]],
            "last_point": [environment.temperatures_c[-1], environment.equilibrium_moistures[-1]],
            "last_hour_mean": [last_hour_temperature, last_hour_moisture],
            "post_measurement_policy": [
                environment.nominal_temperature_c,
                environment.nominal_equilibrium_moisture,
            ],
        },
        "radius": {
            "source": radius.trace.__dict__,
            "time_range_s": [radius.times_s[0], radius.times_s[-1]],
            "radius_range_m": [radius.radii_m[0], radius.radii_m[-1]],
            "post_measurement_policy_m": radius.radii_m[-1],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
