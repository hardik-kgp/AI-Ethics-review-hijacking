# python run_hf_bert.py -tep data/balanced_review_hijack_test.tsv -d "cuda:0" -e ReviewBERT_fix
# python run_hf_bert.py -tep data/emanual_balanced_review_hijack_test.tsv -d "cuda:0" -e EManualBERT_fix
# python run_hf_bert.py -tep data/product_augmented/back/emanual_balanced_review_hijack_test.tsv -d "cuda:0" -e EManualBERT_pb -bp bert-base-uncased -tkp "~/bert-base-uncased-tok"
python run_hf_bert.py -tep data/product+review_augmented/back/emanual_balanced_review_hijack_test.tsv -d "cuda:0" -e EManual_BERT_prb -bp bert-base-uncased -tkp "~/bert-base-uncased-tok"