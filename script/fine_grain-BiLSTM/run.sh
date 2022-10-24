for id in $(seq 0 4)  
do
echo $id
# python main.py --vocab token2idx.json tag2idx.json --idx $id && 
python inference.py --vocab token2idx.json tag2idx.json --idx $id
done


# for id in $(seq 0 0)  
# do
# echo $id
# python save_res.py --vocab token2idx.json tag2idx.json --idx $id
# done