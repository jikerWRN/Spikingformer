-----------------------------------------------------------------------
20260723

离线 1 —— 离线 1

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1  

-------------------------------
在线 —— 在线 SJ反传

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_sj_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1  


在线 —— 在线 

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_UPB_10_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1  


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_UPB_15_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1  







-----------------------------------------------------------------------
-----------------------------------------------------------------------

-------------------------------
在线 —— 固定截断范围 6

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_six_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 



-------------------------------
在线 —— 加载之前的UPB作为截断范围

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fix_checkpint_UPB_20_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint 


-------------------------------
离线 固定之前的UPB —— 离线 固定之前的UPB


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fixUPB_10 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint 

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fixUPB_15 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint 


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fixUPB_20 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/E_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint 



















# -----------------------------------


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_four_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fixUPB_10 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal-20260720-061303/model_best.pth.tar


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fixUPB_15 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal-20260720-061303/model_best.pth.tar


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fix_checkpint_UPB_20_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal-20260720-061303/model_best.pth.tar


# -----------------------------------


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_six_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 



CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_eight_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 