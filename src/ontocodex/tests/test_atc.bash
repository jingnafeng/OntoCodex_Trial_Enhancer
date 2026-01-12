pip install -e .

python - << 'PY'
from ontocodex.kb.terminology_store import TerminologyStore
store = TerminologyStore.from_dir("data")

hits = store.lookup("STOMATOLOGICAL PREPARATIONS", system="ATC", k=3)
print(hits[0]["system"], hits[0]["code"], hits[0]["term"], hits[0]["extra"])

rev = store.lookup_code("A01", system="ATC")
print("REV:", rev["system"], rev["code"], rev["term"])
PY
