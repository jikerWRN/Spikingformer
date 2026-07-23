-----------------------------------------------------------------------
20260722


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_six_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 


# -----------------------------------


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_four_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fixUPB_10 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal-20260720-061303/model_best.pth.tar


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fixUPB_15 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal-20260720-061303/model_best.pth.tar


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_fix_checkpint_UPB_20_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal-20260720-061303/model_best.pth.tar


# -----------------------------------


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_six_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 



CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file E_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_eight_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 