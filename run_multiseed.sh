#!/bin/bash

# Danh sách các seed cần chạy
SEEDS=(0 1 2 3 42)

for SEED in "${SEEDS[@]}"
do
  echo "=================================================="
  echo "Bắt đầu chạy phase với seed: $SEED"
  echo "=================================================="

  # Tạo thư mục output trước để tránh lỗi
  mkdir -p outputs/multiseed/seed_$SEED

  # Chạy lệnh Python tương ứng
  python3 run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios \
      data/scenarios/generated/compiled/tiny_T1_000.yml \
      data/scenarios/generated/compiled/tiny_T2_000.yml \
      data/scenarios/generated/compiled/tiny_T3_000.yml \
      data/scenarios/generated/compiled/tiny_T4_000.yml \
      data/scenarios/generated/compiled/small-linear_T1_000.yml \
      data/scenarios/generated/compiled/small-linear_T2_000.yml \
      data/scenarios/generated/compiled/small-linear_T3_000.yml \
      data/scenarios/generated/compiled/small-linear_T4_000.yml \
    --episode-config data/config/curriculum_episodes.json \
    --training-mode intra_topology \
    --transfer-strategy conservative \
    --fisher-beta 0.3 \
    --train-scratch \
    --seed $SEED \
    --output-dir outputs/multiseed/seed_$SEED

  echo "Đã hoàn thành phase với seed: $SEED"
  echo ""
done

echo "Tất cả các phase multiseed đã huấn luyện xong!"