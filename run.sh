python node_classification.py --model_type GraphSage --source_name Planetoid --dataset_name Cora --aug --lamda 30 --weight_decay 1e-3 --lr 0.01 --split 0.6 0.2 0.2 
python node_classification.py --model_type GraphSage --source_name Planetoid --dataset_name CiteSeer --aug --lamda 30 --weight_decay 1e-3 --lr 0.01 --split 0.6 0.2 0.2  
python node_classification.py --model_type GraphSage --source_name WebKB --dataset_name Texas --aug --lamda 30 --weight_decay 1e-3 --lr 0.01 --split 0.6 0.2 0.2    
python node_classification.py --model_type GraphSage --source_name WebKB --dataset_name Cornell --aug --lamda 30 --weight_decay 1e-3 --lr 0.01 --split 0.6 0.2 0.2 
