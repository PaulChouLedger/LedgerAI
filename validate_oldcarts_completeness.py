#!/usr/bin/env python3
import json
from pathlib import Path

REQUIRED_ELEMENTS = [
    "onset", "location", "duration", "character", "aggravating", "relieving", "timing", "severity"
]


def check_file(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)
    keyf = data.get('key_features', {})
    struct = keyf.get('structured_oldcarts', {})
    problems = []
    if not struct:
        return ["missing structured_oldcarts"]
    for el in REQUIRED_ELEMENTS:
        section = struct.get(el)
        if section is None:
            problems.append(f"missing {el}")
            continue
        if not isinstance(section, dict):
            problems.append(f"{el} not a dict")
            continue
        inc = section.get('includes', [])
        if inc is None or len(inc) == 0:
            problems.append(f"{el} includes empty")
        else:
            bad = [x for x in inc if not isinstance(x, dict) or 'medical' not in x or 'patient_friendly' not in x]
            if bad:
                problems.append(f"{el} includes not converted ({len(bad)})")
        exc = section.get('excludes', [])
        if exc:
            bad2 = [x for x in exc if not isinstance(x, dict) or 'medical' not in x or 'patient_friendly' not in x]
            if bad2:
                problems.append(f"{el} excludes not converted ({len(bad2)})")
    return problems


def main():
    base = Path('llm-medical-container/medical/guidelines')
    systems = [p for p in base.iterdir() if p.is_dir()]
    total = 0
    issues = 0
    for sysdir in systems:
        for jf in sysdir.glob('*.json'):
            total += 1
            probs = check_file(jf)
            if probs:
                issues += 1
                print(f"{jf}:")
                for p in probs:
                    print(f"  - {p}")
    print(f"\nScanned {total} files. Files with issues: {issues}")


if __name__ == '__main__':
    main()


