for id in $(seq 0 0)  
do
echo $id
# python main.py  --idx $id && 
python inference.py --idx $id
done