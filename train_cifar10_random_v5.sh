-----------------------------------------------------------------------
20260708


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_count -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1
a

----------------------------------------

Dynamic backward 

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_UPB_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_UPB_15_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9



----------------------------------------
fixed backward


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_one_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9
b

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_two_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9
d

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_three_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_four_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9



CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_five_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9

----------------------------------------

adaptive fixed backward

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_fixUPB_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/C_model_4layers_baseline_ILIF01_count_to_adaptive_back_count-20260716-053605/model_best.pth.tar


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_fixUPB_13_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/C_model_4layers_baseline_ILIF01_count_to_adaptive_back_count-20260716-053605/model_best.pth.tar



CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_fixUPB_17_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/C_model_4layers_baseline_ILIF01_count_to_adaptive_back_count-20260716-053605/model_best.pth.tar




CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_back_fixUPB_20_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/C_model_4layers_baseline_ILIF01_count_to_adaptive_back_count-20260716-053605/model_best.pth.tar



-----------------------------------------------------------------------
20260718


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file C_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3_count_mem_del_max -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train/A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3_count_mem_del_max-20260707-020417/model_best.pth.tar




CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file D_model_4layers_baseline_ILIF01_count_to_adaptive_back_UPB_15_online_noatten_del_max -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9