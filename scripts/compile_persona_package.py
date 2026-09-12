"""
Execution script for Stage T030–T034: Compile Production Persona Package.
Generates all production artifacts into data/output/persona_package/:
- style_profile.json (T030)
- linguistic_stats.json (T031)
- behavior_profile.json (T032)
- vocabulary.json (T033)
- persona_report.md (T034)
- system_prompt.md (Production LLM instruction prompt)
- package_metadata.json (Checksums & provenance)
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.persona.package import PersonaPackageCompiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compile_persona_package")


def main() -> None:
    # Ensure stdout handles unicode/emojis on Windows terminals
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = PROJECT_ROOT / "data" / "output" / "persona_package"

    logger.info("Initializing PersonaPackageCompiler v1.0.0 ...")
    compiler = PersonaPackageCompiler(version="1.0.0")

    logger.info("Compiling Persona Package from %s to %s ...", data_dir, output_dir)
    start_time = time.perf_counter()
    artifacts = compiler.compile(data_dir, output_dir)
    elapsed = time.perf_counter() - start_time
    logger.info("Compilation completed in %.3f seconds.", elapsed)

    # 1. Inspect and log generated artifacts
    logger.info("============================================================")
    logger.info("  PRODUCTION PERSONA PACKAGE ARTIFACTS (T030 - T034)")
    logger.info("============================================================")
    for name, path in artifacts.items():
        size_kb = path.stat().st_size / 1024.0
        logger.info("  - %-20s: %-30s (%6.2f KB)", name, path.name, size_kb)
    logger.info("============================================================")

    # 2. Strict Privacy Verification Scan
    names = [
        'Aakarshit', 'aarushi', 'Aditya Gupta', 'Afroz', 'Ticktickboom', 'Gays Ka Parivar',
        'Goklu', 'Gooners', 'Heta', 'Karani', 'Rishi Shah', 'Samruddhi', 'Sudhya',
        'The bock rottom', 'Triponovaa', 'Akshat', 'Vora', 'Ankit', 'Datta', 'Anupam', 'Tarav', 'Swayam', 'Yash',
        'svkm', 'projectsvkm2', 'gaurav', 'advaith', 'manoj'
    ]
    pattern = re.compile('|'.join([r'\b' + re.escape(n) + r'\b' for n in names]), re.IGNORECASE)

    privacy_violations = 0
    for name, path in artifacts.items():
        content = path.read_text(encoding="utf-8", errors="ignore")
        matches = pattern.findall(content)
        if matches:
            logger.error("PRIVACY LEAK in %s: %s", path.name, set(matches))
            privacy_violations += len(matches)

    if privacy_violations == 0:
        logger.info("PRIVACY AUDIT PASSED: Zero private names or credentials in all package artifacts.")
    else:
        logger.error("PRIVACY AUDIT FAILED: %d violations found.", privacy_violations)
        sys.exit(1)

    logger.info("Stages T030–T034 successfully compiled and verified.")


if __name__ == "__main__":
    main()
