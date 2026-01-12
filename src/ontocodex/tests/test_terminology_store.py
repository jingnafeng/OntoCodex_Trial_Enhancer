# This is a test script for the TerminologyStore in the OntoCodex project.


# expected output:
# STATS: {'loaded_sources': ["SKIP (unrecognized OMOP CSV schema): snomed_omop.csv (code_col=None, term_col=concept_name, cols=['concept_id', 'concept_name', 'domain_id', 'concept_class_id', 'standard_concept', 'snomed_ct_us'])", 'LOADED: rxnorm_omop.csv (161698 rows)', 'LOADED: loinc_omop.csv (235217 rows)', 'LOADED: atc_omop.csv (6897 rows)', 'LOADED: measurement_omop.csv (145060 rows)', 'SKIP (missing/empty): LOINC_CUI.xlsx', 'SKIP (no recognizable sheets): medDRA.xlsx', 'SKIP (missing/empty): MedDRA_CTCAE_mapping_v5.xlsx'], 'term_index_keys': 422376, 'code_index_keys': 435190, 'total_records': 548857}

# LOOKUP hits:
# RXNORM 1042838 benzocaine 0.18 MG/MG Oral Gel score= 1.0

# REVERSE lookup: RXNORM 1042838 benzocaine 0.18 MG/MG Oral Gel


from ontocodex.kb.terminology_store import TerminologyStore

store = TerminologyStore.from_dir("data")
print("STATS:", store.stats())

hits = store.lookup("benzocaine 0.18 MG/MG Oral Gel", system="RXNORM", k=5)
print("\nLOOKUP hits:")
for h in hits:
    print(h["system"], h["code"], h["term"], "score=", h["score"])

rev = store.lookup_code("1042838", system="RXNORM")
print("\nREVERSE lookup:", rev["system"], rev["code"], rev["term"] if rev else None)

