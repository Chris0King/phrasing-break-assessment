# fine-tune based on Break-BERT-1
# id=0
# echo $id
# for id in $(seq 0 4)  
# do   
# echo $id
# python fine_grain_finetune.py --idx $id && python inference.py --idx $id
# done

# # ? based on Break-Bert*
# for id in $(seq 0 4)  
# do
# echo $id
# python fine_grain_finetune.py --model_path ../../train/pre-training_res/Break-BERT-2 --res_path ../../train/fine-tuning2_res --idx $id && python inference.py --model_path ../../train/fine-tuning2_res/fine-grain-scoring_ --idx $id
# done

for id in $(seq 0 4)  
do
echo $id
python fine_grain_finetune.py --model_path distilbert-base-uncased --res_path ../../train/fine-tuning0_res --idx $id && python inference.py --model_path ../../train/fine-tuning0_res/fine-grain-scoring_ --idx $id
done

for id in $(seq 0 4)  
do
echo $id
python fine_grain_finetune.py --idx $id && python inference.py --idx $id
done

# for id in $(seq 0 4)  
# do
# echo $id
# python fine_grain_finetune.py --model_path ../../train/Break-Bert-2 --res_path ../../train/fine-tuning2_res --idx $id && python inference.py --model_path ../../train/fine-tuning2_res/fine-grain-scoring_ --idx $id --report True
# done