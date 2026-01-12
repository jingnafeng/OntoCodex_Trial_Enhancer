pip install -e .

python - << 'PY'
from ontocodex.kb.terminology_store import TerminologyStore
store = TerminologyStore.from_dir("data")

hits = store.lookup("Somatic hallucination", system="SNOMEDCT", k=3)
print(hits[0]["system"], hits[0]["code"], hits[0]["term"], hits[0]["extra"])

rev = store.lookup_code("762620006", system="SNOMEDCT")
print("REV:", rev["system"], rev["code"], rev["term"])
PY
