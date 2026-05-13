#!/usr/bin/env bash
# run.sh


DATASET="bace"
CONFIG="./configs/${DATASET}/${DATASET}.yaml"
SEEDS=(2022 2023 2024)


if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi


for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================"
    echo ">>> Running: Seed=${SEED}"
    echo "========================================"

    ${PYTHON_CMD} source/train.py \
        --cfg ${CONFIG} \
        --opts 'SEED' ${SEED} \
        --tag "seed_${SEED}"

    echo ">>> Completed: Seed=${SEED}"
done

echo ""
echo "========================================"
echo ">>> All experiments completed!"
echo ">>> Seeds: ${SEEDS[@]}"
echo "========================================"

declare -a SCORES
for SEED in "${SEEDS[@]}"; do
    LOG_DIR="test/${DATASET}/seed_${SEED}/logs"
    LOG_FILE=$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1)
    if [ -z "${LOG_FILE}" ] || [ ! -f "${LOG_FILE}" ]; then
        echo ">>> Warning: No log file found for seed ${SEED} in ${LOG_DIR}"
        continue
    fi

    SCORE=$(grep -oP 'Mean: \K[0-9.]+' "${LOG_FILE}" | tail -1)
    if [ -n "${SCORE}" ]; then
        SCORES+=("${SCORE}")
        echo ">>> Seed ${SEED} test score: ${SCORE}%"
    else
        echo ">>> Warning: Could not extract score for seed ${SEED}"
    fi
done

if [ ${#SCORES[@]} -gt 0 ]; then
    echo ""
    echo "========================================"
    echo ">>> Aggregated Results (mean ± std)"
    echo "========================================"


    python3 - <<EOF
import statistics, ast
scores_raw = "${SCORES[*]}"
scores_raw = scores_raw.replace(' ', ',')  # 空格分隔转逗号分隔
scores = ast.literal_eval(f"[{scores_raw}]")
mean = statistics.mean(scores)
if len(scores) > 1:
    std = statistics.stdev(scores)  # 样本标准差 (n-1)
else:
    std = 0.0
print(f"  Seeds: {scores}")
print(f"  Mean:  {mean:.2f}%")
print(f"  Std:   {std:.2f}%")
print(f"  => {mean:.2f} ± {std:.2f}%")
EOF
fi
