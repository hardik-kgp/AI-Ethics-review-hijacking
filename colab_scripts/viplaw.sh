python run_hf_bert.py -t -tp /content/drive/MyDrive/new_data/review_augmented/front/emanual_balanced_review_hijack_train.tsv -vp /content/drive/MyDrive/new_data/review_augmented/front/emanual_balanced_review_hijack_val.tsv -d "cuda:0" -e /content/drive/MyDrive/EManualBERT_RoBERTa_rf -bp "roberta-base" -tkp "roberta-base" -bs 32 -ep 20