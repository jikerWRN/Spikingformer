-----------------------------------------------------------------------
20260708


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_count -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


e


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.7
d


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.8
e


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9




-----------------------------------------------------------------------
20260709


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fixUPB_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/Test_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.8 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_count-20260708-022831/model_best.pth.tar

d


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fixUPB_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_count-20260708-022831/model_best.pth.tar

e


-----------------------------------------------------------------------
20260709

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_noatten_crossT -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 4 --spike-selected-percentile 0.9

d

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_noatten_trivalues -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9


e


-----------------------------------------------------------------------
20260710

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fixUPB_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 4 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train/B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_count-20260708-022831/model_best.pth.tar

d



-----------------------------------------------------------------------
20260713


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_count -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 4


-----------------------------------------------------------------------
20260714

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_noatten_differT -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 4 --spike-selected-percentile 0.9


-----------------------------------------------------------------------
20260715

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_two_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file B_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_three_online_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9