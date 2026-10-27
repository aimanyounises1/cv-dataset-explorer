# Version-resolved API evidence

Verification uses Python 3.13.9 with NumPy 2.4.6, Pydantic 2.13.4,
FastAPI 0.139.2, pytest 9.1.1 and Ruff 0.16.2. These versions were read from
the installed verification environment before editing. Dependency constraints
remain in requirements files; this is evidence for this verification run, not a lockfile.

| symbol | package@version | source | quote |
| --- | --- | --- | --- |
| `numpy.lexsort` | numpy@2.4.6 | Installed `numpy.lexsort.__doc__`; also https://numpy.org/doc/2.4/reference/generated/numpy.lexsort.html | "The last key in the sequence is used for the primary sort order, ties are broken by the second-to-last key, and so on." |
| `BaseModel` | pydantic@2.13.4 | Installed `pydantic.BaseModel.__doc__` | "A base class for creating Pydantic models." |
| `BaseModel.model_dump` | pydantic@2.13.4 | Installed `pydantic.BaseModel.model_dump.__doc__` | "Generate a dictionary representation of the model, optionally specifying which fields to include or exclude." |
| `MonkeyPatch.setattr` | pytest@9.1.1 | Installed `pytest.MonkeyPatch.setattr.__doc__` | "Set attribute value on target, memorizing the old value." |
| `hashlib.sha256` | Python@3.13.9 | Installed `hashlib.sha256.__doc__`; also https://docs.python.org/3.13/library/hashlib.html | "Returns a sha256 hash object; optionally initialized with a string" |
| `ArgumentParser.add_argument` | Python@3.13.9 | Installed `argparse.ArgumentParser.add_argument.__doc__`; also https://docs.python.org/3.13/library/argparse.html | "add_argument(option_string, option_string, ..., name=value, ...)" |
| `Path.write_text` | Python@3.13.9 | Installed `pathlib.Path.write_text.__doc__` | "Open the file in text mode, write to it, and close the file." |
| `subprocess.run` | Python@3.13.9 | Installed `subprocess.run.__doc__` | "Run command with arguments and return a CompletedProcess instance." |
| `numpy.flatnonzero` | numpy@2.4.6 | Installed `numpy.flatnonzero.__doc__` | "Return indices that are non-zero in the flattened version of a." |
| `numpy.empty` | numpy@2.4.6 | Installed `numpy.empty.__doc__` | "empty(shape, dtype=None, order='C', *, device=None, like=None)" |
| `json.dumps` | Python@3.13.9 | Installed `json.dumps.__doc__` | "Serialize ``obj`` to a JSON formatted ``str``." |
| `datetime.now` | Python@3.13.9 | Installed `datetime.datetime.now.__doc__` | "Returns new datetime object representing current time local to tz." |
| `datetime.isoformat` | Python@3.13.9 | Installed `datetime.datetime.isoformat.__doc__` | "[sep] -> string in ISO 8601 format, YYYY-MM-DDT[HH[:MM[:SS[.mmm[uuu]]]]][+HH:MM]." |
| `hash.hexdigest` | Python@3.13.9 | Installed `hashlib.sha256().hexdigest.__doc__` | "Return the digest value as a string of hexadecimal digits." |
| `statistics.quantiles` | Python@3.13.9 | Installed `statistics.quantiles.__doc__` | "Divide *data* into *n* continuous intervals with equal probability." |
| `importlib.metadata.version` | Python@3.13.9 | Installed `importlib.metadata.version.__doc__` | "Get the version string for the named package." |
| `resource.getrusage().ru_maxrss` | macOS installed system manual | `/usr/bin/man 2 getrusage` | "the maximum resident set size utilized (in bytes)." |
| `resource.getrusage().ru_maxrss` | Linux since 2.6.32 | https://man7.org/linux/man-pages/man2/getrusage.2.html | "This is the maximum resident set size used (in KiB)." |

The archived Apple web manual says kilobytes, while this machine's current
system manual says bytes. The installed system documentation is authoritative
for the macOS verification environment; Linux's documented KiB conversion is
separate. Other platforms report an unavailable reason rather than guessing.
