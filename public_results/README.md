# Public verification packages

Release-specific subdirectories in this folder contain only aggregate evidence
created by `python -m src.publish_verification`. Publication requires a verified
canonical prediction release and complete analysis manifest.

Packages intentionally exclude OASIS data, subject identifiers, subject-level
predictions, split membership, synthetic rows, and fitted models. A package
supports computational inspection of reported aggregate results; it does not
make the previously inspected test partition independent again.
