CUDA_VISIBLE_DEVICES=1 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar
d

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar
e

--------------------------------------------------------
20260701

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_only_conv_encoder -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar
e

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB_only_conv_encoder -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar

d

--------------------------------

CUDA_VISIBLE_DEVICES=0 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1

d


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB_add_scale -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar

e


CUDA_VISIBLE_DEVICES=0 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_only_conv_mlp_fix_debug -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1






CUDA_VISIBLE_DEVICES=1 python cifar10/train.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


--------------------------------------------------------
20260702


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB_sj_back -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar

d

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB_sj_back -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.99 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar

e



----------------------------

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_sj_back -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 

d

CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB_sj_back -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.99

e