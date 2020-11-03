model_name=$1
cd ..
export PYTHONPATH=.
python train.py conf/${model_name}_train.json
python predict.py conf/${model_name}_train.json my_data/test_data.json ${model_name}_predict.txt
python metric_acc.py ${model_name}_predict.txt
