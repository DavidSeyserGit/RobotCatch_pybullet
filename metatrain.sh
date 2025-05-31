#!/usr/bin/env bash
set -euo pipefail

# Basic configuration
GENERATIONS=5
POPULATION=5
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${WORKSPACE_DIR}/reinforcement_learning/sac+her_training.py"
REFINEMENT_SCRIPT="${WORKSPACE_DIR}/reinforcement_learning/ppo_refinement.py"
OUTPUT_DIR="${WORKSPACE_DIR}/reinforcement_learning/runs/metatrain"

echo "Starting meta-training with $GENERATIONS generations, $POPULATION parallel runs each"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

best_model=""
global_best_model=""
global_best_success_rate=0

for gen in $(seq 1 $GENERATIONS); do
    echo "===== Generation $gen/$GENERATIONS ====="
    
    # Run parallel training
    for i in $(seq 1 $POPULATION); do
        run_dir="gen${gen}_run${i}"
        mkdir -p "$run_dir"
        (
            cd "$run_dir"
            if [ -n "$global_best_model" ]; then
                echo "Run $i: Using global best model $global_best_model (success rate: ${global_best_success_rate}%)"
                export INIT_MODEL="${OUTPUT_DIR}/$global_best_model"
            else
                echo "Run $i: Fresh training"
            fi
            python "$TRAIN_SCRIPT"
        ) &
    done
    wait
    
    # Find best model based on final evaluation success rate
    echo "Finding best model from generation $gen..."
    gen_best_success_rate=0
    gen_best_model=""
    
    for i in $(seq 1 $POPULATION); do
        run_dir="gen${gen}_run${i}"
        final_model="$run_dir/her_sac_robot_final.zip"
        best_model_path="$run_dir/logs/best_model.zip"
        eval_file="$run_dir/logs/evaluations.npz"
        
        # Check if we have a best model from training
        if [ -f "$best_model_path" ] && [ -f "$eval_file" ]; then
            # Get the best success rate from all evaluations
            success_rate=$(python -c "import numpy as np; data=np.load('$eval_file'); print(data['successes'].max().mean() * 100)")
            reward=$(python -c "import numpy as np; data=np.load('$eval_file'); print(data['results'][data['successes'].mean(axis=1).argmax()].mean())")
            echo "Model $best_model_path: peak success rate ${success_rate}%, corresponding reward ${reward}"
            if (( $(echo "$success_rate > $gen_best_success_rate" | bc -l) )); then
                gen_best_success_rate=$success_rate
                gen_best_model=$best_model_path
            fi
        # Fall back to final model if best model not found
        elif [ -f "$final_model" ] && [ -f "$eval_file" ]; then
            success_rate=$(python -c "import numpy as np; data=np.load('$eval_file'); print(data['successes'][-1].mean() * 100)")
            reward=$(python -c "import numpy as np; data=np.load('$eval_file'); print(data['results'][-1].mean())")
            echo "Model $final_model: final success rate ${success_rate}%, reward ${reward}"
            if (( $(echo "$success_rate > $gen_best_success_rate" | bc -l) )); then
                gen_best_success_rate=$success_rate
                gen_best_model=$final_model
            fi
        fi
    done
    
    echo "Generation $gen best model: $gen_best_model (success rate: ${gen_best_success_rate}%)"
    
    # Update global best if this generation was better
    if (( $(echo "$gen_best_success_rate > $global_best_success_rate" | bc -l) )); then
        global_best_success_rate=$gen_best_success_rate
        global_best_model=$gen_best_model
        echo "New global best model found! Success rate: ${global_best_success_rate}%"
    else
        echo "Keeping previous global best model (success rate: ${global_best_success_rate}%)"
    fi
done

# Final PPO refinement
echo "===== Final PPO Refinement ====="
final_dir="${OUTPUT_DIR}/final_ppo"
mkdir -p "$final_dir"
cd "$final_dir"

if [ -n "$global_best_model" ]; then
    cp "${OUTPUT_DIR}/$global_best_model" her_sac_robot.zip
    python "$REFINEMENT_SCRIPT" --model her_sac_robot.zip
    echo "Training complete! Check ${final_dir}/ppo_robot_finetuned_simple.zip for the result"
else
    echo "No successful models found during training"
    exit 1
fi
