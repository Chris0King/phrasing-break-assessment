python3 create_json.py --wav_folder ../dataset/LJSpeech-1.1/wavs --wav_script_path ../dataset/LJSpeech-1.1/metadata.csv --json_folder ../dataset/LJSpeech-1.1/json --upload_folder ../dataset/LJSpeech-1.1 && python3 create_token.py --json_folder ../dataset/LJSpeech-1.1/json --upload_folder ../dataset/LJSpeech-1.1 --res_path ../dataset/LJSpeech-1.1/l1_dataset.csv && python3 corrupt_dataset.py --token_path ../dataset/LJSpeech-1.1/l1_dataset.csv --upload_folder ../dataset/LJSpeech-1.1 --res_root_path ../dataset/LJSpeech-1.1


python3 corrupt_dataset.py --token_path ../dataset_for_Break-BERT/l1_dataset.csv --upload_folder ../dataset_for_Break-BERT --res_root_path ../dataset_for_Break-BERT


