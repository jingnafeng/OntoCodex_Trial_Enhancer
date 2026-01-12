pip install -e .

python - << 'PY'
from ontocodex.kb.terminology_store import TerminologyStore

store = TerminologyStore.from_dir("data")
print(store.stats())

# From loinc_omop.csv
hits = store.lookup("Oxygen content in Right atrium", system="LOINC", k=1)
print("OMOP:", hits[0]["code"], hits[0]["extra"].get("cui"), hits[0]["extra"].get("preferred_label"))

# From LOINC_CUI synonym fallback (only works if that label exists in LOINC_CUI.csv)
hits2 = store.lookup("Lymphocytes.kappa/100 lymphocytes:NFr:Pt:Bld:Qn", system="LOINC", k=1)
print("CUI-FALLBACK:", hits2[0]["code"], hits2[0]["extra"].get("cui"))
PY
