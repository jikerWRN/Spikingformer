CUDA_VISIBLE_DEVICES=1 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1

