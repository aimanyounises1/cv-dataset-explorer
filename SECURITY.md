# Security

## What this is

A single-user local tool. `uvicorn app.main:app --port 8000` binds to localhost,
there is no authentication, no multi-tenancy, no telemetry, and no outbound
network call on the request path beyond an optional local Ollama on `:11434`.
Preparation makes two one-time downloads -- the dataset and the SigLIP 2 weights
-- and after they are cached the system runs offline. No secrets are stored in
this repository, and [.env.example](.env.example) contains none.

Consequences worth stating rather than implying:

- **Do not expose the API to a network.** Every endpoint is unauthenticated, and
  `POST /api/admin/reload`, the tag and saved-view writes and the QA sweep all
  act without a caller identity.
- **The QA sweep drives a real browser** at whatever `CVDE_QA_BASE_URL` points
  to. Pointing it anywhere other than your own dev server is not a supported use.
- **Dataset and model artifacts are downloaded from Hugging Face** at first use
  and are not pinned by hash. They carry the trust you extend to that registry.
- **An exported slice inherits the dataset licence**, not this repository's.
  Flickr8k is non-commercial research and education only; see the README section
  on data provenance.
- **Ollama models run locally and are untrusted text generators.** The assistant
  can call tools that write tags; it cannot reach anything the REST API does not
  already expose.

## Reporting something

Open an issue with the request URL and the smallest reproduction you have. If a
finding would be harmful to publish, open an issue that names only the affected
file and says so, and it can be moved somewhere private.

## Not in scope

Hardening this for multi-user or hosted deployment. That is a different system
with different requirements -- see the production scale path in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
