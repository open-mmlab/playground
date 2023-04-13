# export PYTHONPATH='pwd':$PYTHONPATH
# conda activate mmocr1.0

result_path=$1

for epoch in 55 50 45 40 35 30 25 20 15 10
do  
    echo "Eval the epoch $epoch..."
    python tools/analysis_tools/offline_eval.py \
        /media/jiangqing/jqssd/ICDAR-2023/results/hiertext/test_evaluator.py \
        $result_path/mmocr_result_epoch_$epoch.pkl \
        $result_path/final_epoch_$epoch.json
done