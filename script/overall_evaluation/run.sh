# fine-tune based on Break-BERT-1
# id=4
# echo $id
# python coarse_grain_finetune.py --idx $id && python inference.py --idx $id
# fine-tune based on Break-BERT-2
# ?1. train
# python coarse_grain_finetune.py --model_path ../../train/pre-training_res/Break-BERT-2 --res_path ../../train/fine-tuning2_res --idx 0
# ?2. test
# python inference.py --model_path ../../train/fine-tuning2_res/coarse-grain-scoring_ --idx 0

# python coarse_grain_finetune.py --model_path ../../train/pre-training_res/Break-BERT-2 --res_path ../../train/fine-tuning2_res --idx $id && python inference.py --model_path ../../train/fine-tuning2_res/coarse-grain-scoring_ --idx $id



# python coarse_grain_finetune.py --model_path bert-base-uncased --res_path ../../train/fine-tuning0_res --idx $id && python inference.py --model_path ../../train/fine-tuning0_res/coarse-grain-scoring_ --idx $id

for id in $(seq 0 4)  
do
echo $id
python coarse_grain_finetune.py --model_path bert-base-uncased --res_path ../../train/fine-tuning0_res --idx $id && python inference.py --model_path ../../train/fine-tuning0_res/coarse-grain-scoring_ --idx $id --report True
done

# for id in $(seq 0 4)  
# do
# echo $id
# python coarse_grain_finetune.py --idx $id && python inference.py --model_path ../../train/fine-tuning_res/coarse-grain-scoring_ --idx $id --report True
# done

# for id in $(seq 0 4)  
# do
# echo $id
# python coarse_grain_finetune.py --model_path ../../train/Break-Bert-2 --res_path ../../train/fine-tuning2_res --idx $id && python inference.py --model_path ../../train/fine-tuning2_res/coarse-grain-scoring_ --idx $id --report True
# done