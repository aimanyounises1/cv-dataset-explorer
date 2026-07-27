---
name: Feature request
about: Propose a capability, and the measurement that would show it worked
labels: enhancement
---

**The question a user cannot answer today**

Phrase it as a query over the dataset if you can: "which validation samples are
both dark and rare", "which captions disagree with their siblings".

**Which surface it belongs to**

A view, a search parameter, an agent tool, a batch pass in `app.analyze`, or a
document.

**How we would know it worked**

Prefer something measurable: a recall delta under one of the protocols in the
README section on retrieval and evaluation, a query that returns the right slice,
or a workflow the QA sweep can assert. "It would look better" is a legitimate
answer only for the interface.

**What it must not break**

This project deliberately keeps: local-only operation with no cloud dependency,
graceful degradation when an optional layer is missing, filters applied inside
the ranking rather than after it, paging that stops at the ranking horizon, and
the existing API and export shapes. See
[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).
